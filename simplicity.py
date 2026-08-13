#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Synthetic-data RMT diagnostics for unstructured, TT-based, and TR-based VQCs.

Key design
----------
1. The synthetic dataset is generated once from --data_seed and shared by all
   model seeds.
2. TT and TR use one fixed Gaussian latent vector z, sampled once from
   --latent_seed and stored as a non-trainable buffer.
3. The fixed latent is shared across all TT/TR model seeds, all input samples,
   and all dataset prefixes m.
4. The Gaussian latent is excluded from trainable parameter counts, Jacobians,
   and empirical NTKs.
5. Each model is initialized once per model seed and reused for every m.
6. Pointwise ensemble statistics are computed across model seeds.
7. Functional statistics are computed across inputs for each fixed model.

Example
-------
python synthetic_rmt_tensorhyper_rmt.py \
    --models unstructured_vqc,tt_tensor_hyper_vqc,tr_tensor_hyper_vqc \
    --num_qubits 8 \
    --depth 6 \
    --input_dim 64 \
    --m_list 16,32,64,128 \
    --seeds 0,1,2,3,4,5,6,7,8,9 \
    --data_seed 12345 \
    --latent_seed 2026 \
    --latent_std 1.0
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


# ============================================================================
# Utilities
# ============================================================================

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0, 0.0
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return mean, std


def sample_variance(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size <= 1:
        return 0.0
    return float(array.var(ddof=1))


def pairwise_distinguishability(values: torch.Tensor) -> float:
    values = values.flatten()
    count = values.numel()
    if count <= 1:
        return 0.0

    differences = torch.abs(values[:, None] - values[None, :])
    mask = torch.triu(
        torch.ones(
            count,
            count,
            dtype=torch.bool,
            device=values.device,
        ),
        diagonal=1,
    )
    return float(differences[mask].mean().detach().cpu())


def pairwise_distinguishability_numpy(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size <= 1:
        return 0.0

    differences = np.abs(array[:, None] - array[None, :])
    upper = np.triu_indices(array.size, k=1)
    return float(differences[upper].mean())


def trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]


def count_trainable_parameters(model: nn.Module) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in trainable_parameters(model)
        )
    )


def fixed_gaussian_latent(
    dimension: int,
    seed: int,
    std: float = 1.0,
    normalize_norm: bool = False,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample one Gaussian latent vector using a dedicated CPU generator.

    When normalize_norm=False:
        z ~ N(0, std^2 I).

    When normalize_norm=True:
        z is rescaled to have approximately unit Euclidean norm.

    The returned tensor should be registered as a non-trainable buffer.
    """
    if dimension <= 0:
        raise ValueError("dimension must be positive.")
    if std <= 0:
        raise ValueError("std must be positive.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    latent = std * torch.randn(
        1,
        dimension,
        generator=generator,
        dtype=dtype,
    )

    if normalize_norm:
        latent = latent / torch.linalg.vector_norm(
            latent,
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)

    return latent


def deterministic_base_angle_vector(
    target_dim: int,
    target_rms: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Construct one seed-independent low-amplitude circuit anchor.
    """
    if target_dim <= 0:
        raise ValueError("target_dim must be positive.")
    if target_rms < 0:
        raise ValueError("target_rms must be nonnegative.")

    if target_rms == 0:
        return torch.zeros(1, target_dim, dtype=dtype)

    index = torch.arange(target_dim, dtype=dtype)
    pattern = (
        torch.sin(
            2.0
            * math.pi
            * (index + 0.5)
            / max(1, target_dim)
        )
        + 0.35
        * torch.cos(
            6.0
            * math.pi
            * (index + 0.5)
            / max(1, target_dim)
        )
    )
    pattern = pattern - pattern.mean()

    rms = torch.sqrt(pattern.square().mean()).clamp_min(1e-8)
    return (
        target_rms
        * pattern
        / rms
    ).reshape(1, target_dim)


def add_controlled_anchor_noise(
    anchor: torch.Tensor,
    relative_noise_scale: float,
) -> torch.Tensor:
    """
    Add one seed-dependent perturbation around a deterministic anchor.
    """
    if relative_noise_scale < 0:
        raise ValueError(
            "relative_noise_scale must be nonnegative."
        )

    if relative_noise_scale == 0:
        return anchor.clone()

    noise = torch.randn_like(anchor)
    noise = noise - noise.mean(dim=-1, keepdim=True)
    noise = noise / torch.sqrt(
        noise.square().mean(dim=-1, keepdim=True)
    ).clamp_min(1e-8)

    anchor_rms = torch.sqrt(
        anchor.square().mean(dim=-1, keepdim=True)
    )
    reference_rms = anchor_rms.clamp_min(
        1.0 / math.sqrt(anchor.size(-1))
    )

    return (
        anchor
        + relative_noise_scale
        * reference_rms
        * noise
    )


def partially_center_tensor_residual(
    vector: torch.Tensor,
    depth: int,
    n_wires: int,
    centering_strength: float,
) -> torch.Tensor:
    """
    Partially remove common rotation-channel modes across wires.
    """
    if not 0 <= centering_strength <= 1:
        raise ValueError(
            "centering_strength must lie in [0, 1]."
        )

    shaped = vector.reshape(
        vector.size(0),
        depth,
        n_wires,
        3,
    )
    common_mode = shaped.mean(dim=2, keepdim=True)
    shaped = shaped - centering_strength * common_mode
    return shaped.reshape(vector.size(0), -1)


def fixed_output_calibration(
    raw_vector: torch.Tensor,
    target_rms: float,
    max_gain: float,
    eps: float = 1e-8,
) -> float:
    """
    Compute one scalar initialization calibration factor.

    This factor is fixed after initialization and introduces no
    input-dependent normalization Jacobian.
    """
    if target_rms <= 0:
        raise ValueError("target_rms must be positive.")
    if max_gain <= 0:
        raise ValueError("max_gain must be positive.")

    with torch.no_grad():
        raw_rms = torch.sqrt(
            raw_vector.square().mean() + eps
        )
        gain = target_rms / float(raw_rms.detach().cpu())
        return float(min(max(gain, 0.0), max_gain))


# ============================================================================
# RMT-inspired spectral metrics
# ============================================================================

def marchenko_pastur_density(
    grid: np.ndarray,
    aspect_ratio: float,
) -> np.ndarray:
    gamma = float(aspect_ratio)

    if not 0 < gamma <= 1:
        raise ValueError(
            "aspect_ratio must lie in (0, 1]."
        )

    sqrt_gamma = math.sqrt(gamma)
    lower = (1.0 - sqrt_gamma) ** 2
    upper = (1.0 + sqrt_gamma) ** 2

    density = np.zeros_like(grid, dtype=np.float64)
    mask = (grid > lower) & (grid < upper)
    x = grid[mask]

    density[mask] = np.sqrt(
        (upper - x) * (x - lower)
    ) / (
        2.0 * math.pi * gamma * x
    )
    return density


def marchenko_pastur_ks_distance(
    normalized_eigenvalues: np.ndarray,
    aspect_ratio: float,
    grid_size: int = 4096,
) -> Tuple[float, float, float]:
    values = np.asarray(
        normalized_eigenvalues,
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0, 0.0, 0.0

    gamma = min(
        max(float(aspect_ratio), 1e-8),
        1.0,
    )
    sqrt_gamma = math.sqrt(gamma)
    lower = (1.0 - sqrt_gamma) ** 2
    upper = (1.0 + sqrt_gamma) ** 2

    grid_max = max(
        float(values.max()) * 1.05,
        upper * 1.05,
        1.0,
    )
    grid = np.linspace(
        1e-8,
        grid_max,
        grid_size,
    )
    density = marchenko_pastur_density(
        grid,
        gamma,
    )

    increments = (
        0.5
        * (density[1:] + density[:-1])
        * np.diff(grid)
    )
    cdf = np.concatenate(
        [[0.0], np.cumsum(increments)]
    )

    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]

    sorted_values = np.sort(values)
    empirical_cdf = (
        np.arange(
            1,
            sorted_values.size + 1,
        )
        / sorted_values.size
    )

    theoretical_cdf = np.interp(
        sorted_values,
        grid,
        cdf,
        left=0.0,
        right=1.0,
    )

    ks_distance = float(
        np.max(
            np.abs(
                empirical_cdf
                - theoretical_cdf
            )
        )
    )

    bulk_fraction = float(
        np.mean(
            (values >= lower)
            & (values <= upper)
        )
    )

    upper_edge_ratio = float(
        values.max()
        / max(upper, 1e-12)
    )

    return (
        ks_distance,
        bulk_fraction,
        upper_edge_ratio,
    )


def adjacent_gap_ratio(
    eigenvalues: np.ndarray,
    eps: float = 1e-10,
) -> float:
    values = np.sort(
        np.asarray(
            eigenvalues,
            dtype=np.float64,
        )
    )
    values = values[values > eps]

    if values.size < 3:
        return 0.0

    gaps = np.diff(values)
    if gaps.size < 2:
        return 0.0

    numerator = np.minimum(
        gaps[:-1],
        gaps[1:],
    )
    denominator = np.maximum(
        gaps[:-1],
        gaps[1:],
    )

    valid = denominator > eps
    if not np.any(valid):
        return 0.0

    return float(
        np.mean(
            numerator[valid]
            / denominator[valid]
        )
    )


def kernel_metrics(
    kernel: torch.Tensor,
    parameter_count: int,
    eps: float = 1e-12,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    kernel = 0.5 * (kernel + kernel.T)
    eigenvalues = torch.linalg.eigvalsh(
        kernel
    ).clamp_min(0.0)

    trace = eigenvalues.sum()
    squared_sum = eigenvalues.square().sum()

    operator_norm = (
        eigenvalues.max()
        if eigenvalues.numel()
        else torch.tensor(
            0.0,
            device=kernel.device,
        )
    )

    if (
        trace.item() <= eps
        or squared_sum.item() <= eps
    ):
        effective_rank = 0.0
        stable_rank = 0.0
        spectral_entropy = 0.0
        normalized_entropy = 0.0
        normalized_eigenvalues = torch.zeros_like(
            eigenvalues
        )

    else:
        effective_rank = float(
            (
                trace.square()
                / squared_sum
            ).detach().cpu()
        )

        stable_rank = float(
            (
                trace
                / operator_norm.clamp_min(eps)
            ).detach().cpu()
        )

        probabilities = eigenvalues / trace
        positive = probabilities[
            probabilities > eps
        ]

        spectral_entropy = float(
            (
                -positive
                * torch.log(positive)
            ).sum().detach().cpu()
        )

        normalized_entropy = (
            spectral_entropy
            / max(
                math.log(
                    max(
                        2,
                        eigenvalues.numel(),
                    )
                ),
                eps,
            )
        )

        normalized_eigenvalues = (
            eigenvalues
            / eigenvalues.mean().clamp_min(eps)
        )

    normalizer = max(
        1,
        min(
            kernel.shape[0],
            parameter_count,
        ),
    )

    normalized_effective_rank = (
        effective_rank
        / normalizer
    )

    raw_np = (
        eigenvalues.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    normalized_np = (
        normalized_eigenvalues.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    aspect_ratio = min(
        1.0,
        kernel.shape[0]
        / max(1, parameter_count),
    )

    (
        mp_ks,
        mp_bulk_fraction,
        mp_upper_edge_ratio,
    ) = marchenko_pastur_ks_distance(
        normalized_np,
        aspect_ratio,
    )

    metrics = {
        "ntk_trace_per_sample": float(
            (
                trace
                / max(1, kernel.shape[0])
            ).detach().cpu()
        ),
        "ntk_operator_norm": float(
            operator_norm.detach().cpu()
        ),
        "ntk_effective_rank": effective_rank,
        "ntk_normalized_effective_rank": (
            normalized_effective_rank
        ),
        "ntk_stable_rank": stable_rank,
        "ntk_spectral_entropy": spectral_entropy,
        "ntk_normalized_spectral_entropy": (
            normalized_entropy
        ),
        "ntk_adjacent_gap_ratio": (
            adjacent_gap_ratio(normalized_np)
        ),
        "ntk_mp_ks_distance": mp_ks,
        "ntk_mp_bulk_fraction": (
            mp_bulk_fraction
        ),
        "ntk_mp_upper_edge_ratio": (
            mp_upper_edge_ratio
        ),
        "ntk_aspect_ratio": aspect_ratio,
    }

    return (
        metrics,
        raw_np,
        normalized_np,
    )


# ============================================================================
# Tensor-Train layer
# ============================================================================

class TensorTrainLayer(nn.Module):
    """
    TT linear map from R^{prod(input_dims)} to R^{prod(output_dims)}.
    """

    def __init__(
        self,
        input_dims: List[int],
        output_dims: List[int],
        tt_ranks: List[int],
        core_init_scale: float = 0.15,
        use_bias: bool = True,
    ):
        super().__init__()

        if len(input_dims) != len(output_dims):
            raise ValueError(
                "input_dims and output_dims must have equal length."
            )
        if len(tt_ranks) != len(input_dims) + 1:
            raise ValueError(
                "tt_ranks must have length len(input_dims)+1."
            )
        if tt_ranks[0] != 1 or tt_ranks[-1] != 1:
            raise ValueError(
                "Boundary TT ranks must equal 1."
            )
        if min(*input_dims, *output_dims, *tt_ranks) <= 0:
            raise ValueError(
                "All TT dimensions and ranks must be positive."
            )

        self.input_dims = list(input_dims)
        self.output_dims = list(output_dims)
        self.tt_ranks = list(tt_ranks)

        self.tt_cores = nn.ParameterList()

        for mode in range(len(input_dims)):
            rank_left = tt_ranks[mode]
            rank_right = tt_ranks[mode + 1]
            input_mode = input_dims[mode]
            output_mode = output_dims[mode]

            core = nn.Parameter(
                core_init_scale
                * torch.randn(
                    rank_left,
                    input_mode,
                    output_mode,
                    rank_right,
                )
            )
            self.tt_cores.append(core)

        output_dim = int(math.prod(output_dims))
        if use_bias:
            self.bias = nn.Parameter(
                torch.zeros(output_dim)
            )
        else:
            self.register_parameter(
                "bias",
                None,
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        expected_dim = int(
            math.prod(self.input_dims)
        )

        if (
            x.ndim != 2
            or x.size(1) != expected_dim
        ):
            raise ValueError(
                f"Expected x with shape "
                f"[batch,{expected_dim}], "
                f"got {tuple(x.shape)}."
            )

        tensorized_x = x.reshape(
            batch_size,
            *self.input_dims,
        )
        order = len(self.input_dims)

        labels = [
            character
            for character in (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            )
            if character != "B"
        ]

        if len(labels) < 3 * order + 1:
            raise ValueError(
                "Too many TT modes for einsum labels."
            )

        input_indices = labels[:order]
        output_indices = labels[
            order : 2 * order
        ]
        rank_indices = labels[
            2 * order : 3 * order + 1
        ]

        x_subscript = (
            "B"
            + "".join(input_indices)
        )

        core_subscripts = [
            (
                f"{rank_indices[mode]}"
                f"{input_indices[mode]}"
                f"{output_indices[mode]}"
                f"{rank_indices[mode + 1]}"
            )
            for mode in range(order)
        ]

        output_subscript = (
            "B"
            + "".join(output_indices)
        )

        equation = (
            x_subscript
            + ","
            + ",".join(core_subscripts)
            + "->"
            + output_subscript
        )

        output = torch.einsum(
            equation,
            tensorized_x,
            *self.tt_cores,
        ).reshape(batch_size, -1)

        if self.bias is not None:
            output = output + self.bias

        return output


# ============================================================================
# Tensor-Ring layer
# ============================================================================

class TensorRingLayer(nn.Module):
    """
    TR linear operator from R^{prod(input_dims)} to R^{prod(output_dims)}.
    """

    def __init__(
        self,
        input_dims: List[int],
        output_dims: List[int],
        tr_ranks: List[int],
        core_init_scale: float = 0.10,
        use_bias: bool = True,
    ):
        super().__init__()

        if len(input_dims) != len(output_dims):
            raise ValueError(
                "input_dims and output_dims must have equal length."
            )
        if len(tr_ranks) != len(input_dims):
            raise ValueError(
                "tr_ranks must contain one cyclic rank per mode."
            )
        if min(*input_dims, *output_dims, *tr_ranks) <= 0:
            raise ValueError(
                "All TR dimensions and ranks must be positive."
            )

        self.input_dims = list(input_dims)
        self.output_dims = list(output_dims)
        self.tr_ranks = list(tr_ranks)
        self.order = len(input_dims)

        self.tr_cores = nn.ParameterList()

        for mode in range(self.order):
            rank_left = tr_ranks[mode]
            rank_right = tr_ranks[
                (mode + 1) % self.order
            ]
            input_mode = input_dims[mode]
            output_mode = output_dims[mode]

            fan = max(
                1,
                rank_left * input_mode
                + output_mode * rank_right,
            )
            scale = (
                core_init_scale
                * math.sqrt(2.0 / fan)
            )

            core = nn.Parameter(
                scale
                * torch.randn(
                    rank_left,
                    input_mode,
                    output_mode,
                    rank_right,
                )
            )
            self.tr_cores.append(core)

        output_dim = int(math.prod(output_dims))
        if use_bias:
            self.bias = nn.Parameter(
                torch.zeros(output_dim)
            )
        else:
            self.register_parameter(
                "bias",
                None,
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        expected_dim = int(
            math.prod(self.input_dims)
        )

        if (
            x.ndim != 2
            or x.size(1) != expected_dim
        ):
            raise ValueError(
                f"Expected x with shape "
                f"[batch,{expected_dim}], "
                f"got {tuple(x.shape)}."
            )

        tensorized_x = x.reshape(
            batch_size,
            *self.input_dims,
        )
        order = self.order

        labels = [
            character
            for character in (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            )
            if character != "B"
        ]

        if len(labels) < 3 * order:
            raise ValueError(
                "Too many TR modes for einsum labels."
            )

        input_indices = labels[:order]
        output_indices = labels[
            order : 2 * order
        ]
        rank_indices = labels[
            2 * order : 3 * order
        ]

        x_subscript = (
            "B"
            + "".join(input_indices)
        )

        core_subscripts = []
        for mode in range(order):
            left_rank = rank_indices[mode]
            right_rank = rank_indices[
                (mode + 1) % order
            ]

            core_subscripts.append(
                f"{left_rank}"
                f"{input_indices[mode]}"
                f"{output_indices[mode]}"
                f"{right_rank}"
            )

        output_subscript = (
            "B"
            + "".join(output_indices)
        )

        equation = (
            x_subscript
            + ","
            + ",".join(core_subscripts)
            + "->"
            + output_subscript
        )

        output = torch.einsum(
            equation,
            tensorized_x,
            *self.tr_cores,
        ).reshape(batch_size, -1)

        if self.bias is not None:
            output = output + self.bias

        return output


# ============================================================================
# Quantum circuit
# ============================================================================

class SimpleRYEncoder(tq.QuantumModule):
    def __init__(
        self,
        n_wires: int,
    ):
        super().__init__()
        self.n_wires = n_wires

    @tq.static_support
    def forward(
        self,
        q_device: tq.QuantumDevice,
        encoded_x: torch.Tensor,
    ) -> None:
        for wire in range(self.n_wires):
            tqf.ry(
                q_device,
                wires=wire,
                params=encoded_x[:, wire],
                static=self.static_mode,
                parent_graph=self.graph,
            )


class BaseVQC(tq.QuantumModule):
    """
    Ring-entangled VQC returning <Z_0>.
    """

    def __init__(
        self,
        n_wires: int,
        depth: int,
        trainable_global_angles: bool,
    ):
        super().__init__()

        self.n_wires = n_wires
        self.depth = depth
        self.encoder = SimpleRYEncoder(
            n_wires
        )
        self.measure = tq.MeasureAll(
            tq.PauliZ
        )

        if trainable_global_angles:
            self.global_angles = nn.Parameter(
                0.1
                * torch.randn(
                    depth,
                    n_wires,
                    3,
                )
            )
        else:
            self.register_parameter(
                "global_angles",
                None,
            )

    def _entangle_ring(
        self,
        q_device: tq.QuantumDevice,
    ) -> None:
        for wire in range(
            self.n_wires - 1
        ):
            tqf.cnot(
                q_device,
                wires=[wire, wire + 1],
                static=self.static_mode,
                parent_graph=self.graph,
            )

        if self.n_wires > 1:
            tqf.cnot(
                q_device,
                wires=[
                    self.n_wires - 1,
                    0,
                ],
                static=self.static_mode,
                parent_graph=self.graph,
            )

    @tq.static_support
    def forward(
        self,
        encoded_x: torch.Tensor,
        q_device: tq.QuantumDevice,
        angles_batch: Optional[
            torch.Tensor
        ] = None,
    ) -> torch.Tensor:
        batch_size = encoded_x.size(0)

        q_device.reset_states(
            batch_size
        )
        self.encoder(
            q_device,
            encoded_x,
        )

        if (
            angles_batch is None
            and self.global_angles is None
        ):
            raise RuntimeError(
                "angles_batch is required because "
                "this VQC has no global angles."
            )

        for layer in range(self.depth):
            for wire in range(
                self.n_wires
            ):
                if angles_batch is None:
                    rx_angle = (
                        self.global_angles[
                            layer,
                            wire,
                            0,
                        ]
                    )
                    ry_angle = (
                        self.global_angles[
                            layer,
                            wire,
                            1,
                        ]
                    )
                    rz_angle = (
                        self.global_angles[
                            layer,
                            wire,
                            2,
                        ]
                    )
                else:
                    rx_angle = (
                        angles_batch[
                            :,
                            layer,
                            wire,
                            0,
                        ]
                    )
                    ry_angle = (
                        angles_batch[
                            :,
                            layer,
                            wire,
                            1,
                        ]
                    )
                    rz_angle = (
                        angles_batch[
                            :,
                            layer,
                            wire,
                            2,
                        ]
                    )

                tqf.rx(
                    q_device,
                    wires=wire,
                    params=rx_angle,
                    static=self.static_mode,
                    parent_graph=self.graph,
                )
                tqf.ry(
                    q_device,
                    wires=wire,
                    params=ry_angle,
                    static=self.static_mode,
                    parent_graph=self.graph,
                )
                tqf.rz(
                    q_device,
                    wires=wire,
                    params=rz_angle,
                    static=self.static_mode,
                    parent_graph=self.graph,
                )

            self._entangle_ring(
                q_device
            )

        all_z = self.measure(
            q_device
        )
        return all_z[:, 0]


# ============================================================================
# Model families
# ============================================================================

class UnstructuredVQC(nn.Module):
    def __init__(
        self,
        n_wires: int,
        depth: int,
    ):
        super().__init__()

        self.n_wires = n_wires
        self.vqc = BaseVQC(
            n_wires=n_wires,
            depth=depth,
            trainable_global_angles=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        q_device: tq.QuantumDevice,
    ) -> torch.Tensor:
        encoded_x = (
            torch.tanh(
                x[:, : self.n_wires]
            )
            * math.pi
        )

        return self.vqc(
            encoded_x,
            q_device,
            angles_batch=None,
        )


class TensorTrainHyperVQC(nn.Module):
    """
    TT TensorHyper-VQC with one fixed Gaussian latent buffer:

        theta = theta_base + alpha * H_TT(z_fixed; phi).
    """

    def __init__(
        self,
        noise_dim: int,
        input_dims: List[int],
        output_dims: List[int],
        tt_ranks: List[int],
        n_wires: int,
        depth: int,
        residual_global: bool,
        generated_angle_rms: float,
        max_calibration_gain: float,
        base_angle_rms: float,
        initial_residual_scale: float,
        base_anchor_noise: float,
        residual_centering_strength: float,
        latent_seed: int,
        latent_std: float,
        normalize_latent_norm: bool,
        tt_core_init_scale: float,
        tensor_bias: bool,
    ):
        super().__init__()

        self.noise_dim = noise_dim
        self.n_wires = n_wires
        self.depth = depth
        self.residual_global = (
            residual_global
        )
        self.residual_centering_strength = (
            residual_centering_strength
        )

        if int(math.prod(input_dims)) != noise_dim:
            raise ValueError(
                "prod(input_dims) must equal noise_dim."
            )

        target_dim = (
            depth
            * n_wires
            * 3
        )
        if int(
            math.prod(output_dims)
        ) != target_dim:
            raise ValueError(
                "TensorHyper output dimension "
                f"must equal {target_dim}."
            )

        self.tt = TensorTrainLayer(
            input_dims=input_dims,
            output_dims=output_dims,
            tt_ranks=tt_ranks,
            core_init_scale=tt_core_init_scale,
            use_bias=tensor_bias,
        )

        self.vqc = BaseVQC(
            n_wires=n_wires,
            depth=depth,
            trainable_global_angles=(
                residual_global
            ),
        )

        latent = fixed_gaussian_latent(
            dimension=noise_dim,
            seed=latent_seed,
            std=latent_std,
            normalize_norm=normalize_latent_norm,
        )
        self.register_buffer(
            "fixed_latent",
            latent,
        )

        base_anchor = (
            deterministic_base_angle_vector(
                target_dim=target_dim,
                target_rms=base_angle_rms,
            )
        )
        base_angles = (
            add_controlled_anchor_noise(
                base_anchor,
                base_anchor_noise,
            )
        )
        self.register_buffer(
            "base_angle_vector",
            base_angles,
        )

        with torch.no_grad():
            raw_initial_residual = self.tt(
                self.fixed_latent
            )

        calibration = fixed_output_calibration(
            raw_initial_residual,
            target_rms=generated_angle_rms,
            max_gain=max_calibration_gain,
        )
        self.register_buffer(
            "angle_calibration",
            torch.tensor(
                calibration,
                dtype=torch.float32,
            ),
        )

        inverse_softplus = math.log(
            math.expm1(
                initial_residual_scale
            )
        )
        self.residual_log_scale = nn.Parameter(
            torch.tensor(
                inverse_softplus,
                dtype=torch.float32,
            )
        )

    @property
    def residual_scale(
        self,
    ) -> torch.Tensor:
        return torch.nn.functional.softplus(
            self.residual_log_scale
        )

    def generated_angle_vector(
        self,
    ) -> torch.Tensor:
        residual = self.tt(
            self.fixed_latent
        )

        if (
            self.residual_centering_strength
            > 0
        ):
            residual = (
                partially_center_tensor_residual(
                    residual,
                    depth=self.depth,
                    n_wires=self.n_wires,
                    centering_strength=(
                        self.residual_centering_strength
                    ),
                )
            )

        residual = (
            self.angle_calibration
            * residual
        )

        return (
            self.base_angle_vector
            + self.residual_scale
            * residual
        )

    def forward(
        self,
        x: torch.Tensor,
        q_device: tq.QuantumDevice,
    ) -> torch.Tensor:
        batch_size = x.size(0)

        encoded_x = (
            torch.tanh(
                x[:, : self.n_wires]
            )
            * math.pi
        )

        one_angle_vector = (
            self.generated_angle_vector()
        )

        angle_vectors = (
            one_angle_vector.expand(
                batch_size,
                -1,
            )
        )

        angles = angle_vectors.reshape(
            batch_size,
            self.depth,
            self.n_wires,
            3,
        )
        angles = (
            torch.tanh(angles)
            * math.pi
        )

        if self.residual_global:
            angles = (
                angles
                + self.vqc.global_angles.unsqueeze(0)
            )

        return self.vqc(
            encoded_x,
            q_device,
            angles_batch=angles,
        )


class TensorRingHyperVQC(nn.Module):
    """
    TR TensorHyper-VQC with one fixed Gaussian latent buffer:

        theta = theta_base + alpha * H_TR(z_fixed; phi).
    """

    def __init__(
        self,
        noise_dim: int,
        input_dims: List[int],
        output_dims: List[int],
        tr_ranks: List[int],
        n_wires: int,
        depth: int,
        residual_global: bool,
        generated_angle_rms: float,
        max_calibration_gain: float,
        base_angle_rms: float,
        initial_residual_scale: float,
        tr_core_init_scale: float,
        base_anchor_noise: float,
        residual_centering_strength: float,
        latent_seed: int,
        latent_std: float,
        normalize_latent_norm: bool,
        tensor_bias: bool,
    ):
        super().__init__()

        self.noise_dim = noise_dim
        self.n_wires = n_wires
        self.depth = depth
        self.residual_global = (
            residual_global
        )
        self.residual_centering_strength = (
            residual_centering_strength
        )

        if int(math.prod(input_dims)) != noise_dim:
            raise ValueError(
                "prod(input_dims) must equal noise_dim."
            )

        target_dim = (
            depth
            * n_wires
            * 3
        )
        if int(
            math.prod(output_dims)
        ) != target_dim:
            raise ValueError(
                "TensorHyper output dimension "
                f"must equal {target_dim}."
            )

        self.tr = TensorRingLayer(
            input_dims=input_dims,
            output_dims=output_dims,
            tr_ranks=tr_ranks,
            core_init_scale=tr_core_init_scale,
            use_bias=tensor_bias,
        )

        self.vqc = BaseVQC(
            n_wires=n_wires,
            depth=depth,
            trainable_global_angles=(
                residual_global
            ),
        )

        latent = fixed_gaussian_latent(
            dimension=noise_dim,
            seed=latent_seed,
            std=latent_std,
            normalize_norm=normalize_latent_norm,
        )
        self.register_buffer(
            "fixed_latent",
            latent,
        )

        base_anchor = (
            deterministic_base_angle_vector(
                target_dim=target_dim,
                target_rms=base_angle_rms,
            )
        )
        base_angles = (
            add_controlled_anchor_noise(
                base_anchor,
                base_anchor_noise,
            )
        )
        self.register_buffer(
            "base_angle_vector",
            base_angles,
        )

        with torch.no_grad():
            raw_initial_residual = self.tr(
                self.fixed_latent
            )

        calibration = fixed_output_calibration(
            raw_initial_residual,
            target_rms=generated_angle_rms,
            max_gain=max_calibration_gain,
        )
        self.register_buffer(
            "angle_calibration",
            torch.tensor(
                calibration,
                dtype=torch.float32,
            ),
        )

        inverse_softplus = math.log(
            math.expm1(
                initial_residual_scale
            )
        )
        self.residual_log_scale = nn.Parameter(
            torch.tensor(
                inverse_softplus,
                dtype=torch.float32,
            )
        )

    @property
    def residual_scale(
        self,
    ) -> torch.Tensor:
        return torch.nn.functional.softplus(
            self.residual_log_scale
        )

    def generated_angle_vector(
        self,
    ) -> torch.Tensor:
        residual = self.tr(
            self.fixed_latent
        )

        if (
            self.residual_centering_strength
            > 0
        ):
            residual = (
                partially_center_tensor_residual(
                    residual,
                    depth=self.depth,
                    n_wires=self.n_wires,
                    centering_strength=(
                        self.residual_centering_strength
                    ),
                )
            )

        residual = (
            self.angle_calibration
            * residual
        )

        return (
            self.base_angle_vector
            + self.residual_scale
            * residual
        )

    def forward(
        self,
        x: torch.Tensor,
        q_device: tq.QuantumDevice,
    ) -> torch.Tensor:
        batch_size = x.size(0)

        encoded_x = (
            torch.tanh(
                x[:, : self.n_wires]
            )
            * math.pi
        )

        one_angle_vector = (
            self.generated_angle_vector()
        )

        angle_vectors = (
            one_angle_vector.expand(
                batch_size,
                -1,
            )
        )

        angles = angle_vectors.reshape(
            batch_size,
            self.depth,
            self.n_wires,
            3,
        )
        angles = (
            torch.tanh(angles)
            * math.pi
        )

        if self.residual_global:
            angles = (
                angles
                + self.vqc.global_angles.unsqueeze(0)
            )

        return self.vqc(
            encoded_x,
            q_device,
            angles_batch=angles,
        )


# ============================================================================
# Synthetic data
# ============================================================================

def synthesize_x(
    sample_count: int,
    input_dim: int,
    device: torch.device,
    kind: str,
) -> torch.Tensor:
    if kind == "gaussian":
        return torch.randn(
            sample_count,
            input_dim,
            device=device,
        )

    if kind == "structured":
        t = torch.linspace(
            -1.0,
            1.0,
            sample_count,
            device=device,
        ).unsqueeze(1)

        frequencies = torch.arange(
            1,
            input_dim + 1,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0)

        x = torch.sin(
            math.pi
            * t
            * frequencies
            / input_dim
        )

        x = x + 0.3 * torch.cos(
            2.0
            * math.pi
            * t
            * frequencies
            / input_dim
        )

        x = x + 0.05 * torch.randn_like(
            x
        )
        return x

    if kind == "mixed":
        structured = synthesize_x(
            sample_count,
            input_dim,
            device,
            "structured",
        )

        gaussian = torch.randn(
            sample_count,
            input_dim,
            device=device,
        )

        return (
            0.7 * structured
            + 0.3 * gaussian
        )

    raise ValueError(
        f"Unknown data kind: {kind}"
    )


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_outputs(
    model: nn.Module,
    x: torch.Tensor,
    n_wires: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()

    q_device = tq.QuantumDevice(
        n_wires=n_wires,
        bsz=x.size(0),
    ).to(device)

    with torch.no_grad():
        outputs = model(
            x,
            q_device,
        )

    return outputs.detach()


def evaluate_anchor_state(
    model: nn.Module,
    x_anchor: torch.Tensor,
    n_wires: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    q_device = tq.QuantumDevice(
        n_wires=n_wires,
        bsz=1,
    ).to(device)

    with torch.no_grad():
        _ = model(
            x_anchor[:1],
            q_device,
        )

        state = q_device.states.reshape(
            1,
            -1,
        )[0]

        state = (
            state
            / torch.linalg.vector_norm(
                state
            ).clamp_min(1e-12)
        )

    return (
        state.detach()
        .cpu()
        .numpy()
    )


def state_second_frame_potential(
    states: List[np.ndarray],
) -> float:
    if len(states) < 2:
        return 0.0

    matrix = np.stack(
        states,
        axis=0,
    )
    overlaps = (
        matrix
        @ matrix.conj().T
    )

    values = np.abs(
        overlaps
    ) ** 4

    mask = ~np.eye(
        len(states),
        dtype=bool,
    )

    return float(
        values[mask].mean()
    )


def input_jacobian_sensitivity(
    model: nn.Module,
    x: torch.Tensor,
    n_wires: int,
    device: torch.device,
    max_points: int,
) -> Dict[str, float]:
    model.eval()
    sensitivities: List[float] = []

    point_count = min(
        x.size(0),
        max_points,
    )

    for index in range(point_count):
        one_x = (
            x[index : index + 1]
            .detach()
            .clone()
            .requires_grad_(True)
        )

        q_device = tq.QuantumDevice(
            n_wires=n_wires,
            bsz=1,
        ).to(device)

        output = model(
            one_x,
            q_device,
        ).sum()

        gradient = torch.autograd.grad(
            output,
            one_x,
            retain_graph=False,
            create_graph=False,
        )[0]

        active_gradient = gradient[
            :,
            :n_wires,
        ]

        sensitivities.append(
            float(
                active_gradient.square()
                .sum()
                .detach()
                .cpu()
            )
        )

    if not sensitivities:
        return {
            "input_grad_norm_sq_mean": 0.0,
            "input_grad_norm_sq_median": 0.0,
        }

    array = np.asarray(
        sensitivities,
        dtype=np.float64,
    )

    return {
        "input_grad_norm_sq_mean": float(
            array.mean()
        ),
        "input_grad_norm_sq_median": float(
            np.median(array)
        ),
    }


def jacobian_matrix(
    model: nn.Module,
    x: torch.Tensor,
    n_wires: int,
    device: torch.device,
    max_points: int,
) -> torch.Tensor:
    """
    Return G in R^{N x P}, with rows grad_eta f_eta(x_i).

    The fixed Gaussian latent is a registered buffer, not a trainable
    coordinate, and therefore does not contribute to P.
    """
    model.train()
    x = x[:max_points]
    parameters = trainable_parameters(
        model
    )

    if not parameters:
        return torch.empty(
            0,
            0,
            device=device,
        )

    gradient_rows: List[
        torch.Tensor
    ] = []

    for index in range(
        x.size(0)
    ):
        model.zero_grad(
            set_to_none=True
        )

        q_device = tq.QuantumDevice(
            n_wires=n_wires,
            bsz=1,
        ).to(device)

        output = model(
            x[index : index + 1],
            q_device,
        ).sum()

        gradients = torch.autograd.grad(
            output,
            parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        row_parts: List[
            torch.Tensor
        ] = []

        for parameter, gradient in zip(
            parameters,
            gradients,
        ):
            if gradient is None:
                row_parts.append(
                    torch.zeros_like(
                        parameter,
                        memory_format=(
                            torch.contiguous_format
                        ),
                    ).reshape(-1)
                )
            else:
                row_parts.append(
                    gradient.detach()
                    .reshape(-1)
                )

        gradient_rows.append(
            torch.cat(row_parts)
        )

    return torch.stack(
        gradient_rows,
        dim=0,
    )


def jacobian_and_ntk_metrics(
    model: nn.Module,
    x: torch.Tensor,
    n_wires: int,
    device: torch.device,
    max_points: int,
) -> Tuple[
    Dict[str, float],
    np.ndarray,
    np.ndarray,
]:
    gradients = jacobian_matrix(
        model=model,
        x=x,
        n_wires=n_wires,
        device=device,
        max_points=max_points,
    )

    if (
        gradients.numel() == 0
        or gradients.size(0) <= 1
    ):
        zeros = {
            "grad_entry_mean_sq": 0.0,
            "grad_norm_sq_mean": 0.0,
            "grad_norm_sq_median": 0.0,
            "grad_coordinate_var_mean": 0.0,
            "ntk_trace_per_sample": 0.0,
            "centered_ntk_trace_per_sample": 0.0,
            "ntk_operator_norm": 0.0,
            "ntk_effective_rank": 0.0,
            "ntk_normalized_effective_rank": 0.0,
            "ntk_stable_rank": 0.0,
            "ntk_spectral_entropy": 0.0,
            "ntk_normalized_spectral_entropy": 0.0,
            "ntk_adjacent_gap_ratio": 0.0,
            "ntk_mp_ks_distance": 0.0,
            "ntk_mp_bulk_fraction": 0.0,
            "ntk_mp_upper_edge_ratio": 0.0,
            "ntk_aspect_ratio": 0.0,
        }

        return (
            zeros,
            np.zeros(0),
            np.zeros(0),
        )

    row_norm_sq = gradients.square().sum(
        dim=1
    )

    gradient_metrics = {
        "grad_entry_mean_sq": float(
            gradients.square()
            .mean()
            .detach()
            .cpu()
        ),
        "grad_norm_sq_mean": float(
            row_norm_sq.mean()
            .detach()
            .cpu()
        ),
        "grad_norm_sq_median": float(
            row_norm_sq.median()
            .detach()
            .cpu()
        ),
        "grad_coordinate_var_mean": float(
            gradients.var(
                dim=0,
                unbiased=False,
            ).mean()
            .detach()
            .cpu()
        ),
    }

    kernel = gradients @ gradients.T

    centered_gradients = (
        gradients
        - gradients.mean(
            dim=0,
            keepdim=True,
        )
    )

    centered_kernel = (
        centered_gradients
        @ centered_gradients.T
    )

    centered_trace = (
        torch.trace(centered_kernel)
        / max(
            1,
            centered_kernel.size(0),
        )
    )

    (
        spectral_metrics,
        raw_eigenvalues,
        normalized_eigenvalues,
    ) = kernel_metrics(
        kernel=kernel,
        parameter_count=gradients.size(1),
    )

    spectral_metrics[
        "centered_ntk_trace_per_sample"
    ] = float(
        centered_trace.detach().cpu()
    )

    return (
        {
            **gradient_metrics,
            **spectral_metrics,
        },
        raw_eigenvalues,
        normalized_eigenvalues,
    )


# ============================================================================
# Model builder and validation
# ============================================================================

def build_model(
    args: argparse.Namespace,
    model_name: str,
    device: torch.device,
) -> nn.Module:
    if model_name == "unstructured_vqc":
        model = UnstructuredVQC(
            n_wires=args.num_qubits,
            depth=args.depth,
        )

    elif model_name == "tt_tensor_hyper_vqc":
        model = TensorTrainHyperVQC(
            noise_dim=args.noise_dim,
            input_dims=parse_int_list(
                args.tt_noise_input_dims
            ),
            output_dims=parse_int_list(
                args.theta_output_dims
            ),
            tt_ranks=parse_int_list(
                args.tt_ranks
            ),
            n_wires=args.num_qubits,
            depth=args.depth,
            residual_global=(
                args.residual_global
            ),
            generated_angle_rms=(
                args.generated_angle_rms
            ),
            max_calibration_gain=(
                args.max_calibration_gain
            ),
            base_angle_rms=(
                args.base_angle_rms
            ),
            initial_residual_scale=(
                args.tt_initial_residual_scale
            ),
            base_anchor_noise=(
                args.base_anchor_noise
            ),
            residual_centering_strength=(
                args.tt_residual_centering_strength
            ),
            latent_seed=args.latent_seed,
            latent_std=args.latent_std,
            normalize_latent_norm=(
                args.normalize_latent_norm
            ),
            tt_core_init_scale=(
                args.tt_core_init_scale
            ),
            tensor_bias=args.tensor_bias,
        )

    elif model_name == "tr_tensor_hyper_vqc":
        model = TensorRingHyperVQC(
            noise_dim=args.noise_dim,
            input_dims=parse_int_list(
                args.tr_noise_input_dims
            ),
            output_dims=parse_int_list(
                args.theta_output_dims
            ),
            tr_ranks=parse_int_list(
                args.tr_ranks
            ),
            n_wires=args.num_qubits,
            depth=args.depth,
            residual_global=(
                args.residual_global
            ),
            generated_angle_rms=(
                args.generated_angle_rms
            ),
            max_calibration_gain=(
                args.max_calibration_gain
            ),
            base_angle_rms=(
                args.base_angle_rms
            ),
            initial_residual_scale=(
                args.tr_initial_residual_scale
            ),
            tr_core_init_scale=(
                args.tr_core_init_scale
            ),
            base_anchor_noise=(
                args.base_anchor_noise
            ),
            residual_centering_strength=(
                args.tr_residual_centering_strength
            ),
            latent_seed=args.latent_seed,
            latent_std=args.latent_std,
            normalize_latent_norm=(
                args.normalize_latent_norm
            ),
            tensor_bias=args.tensor_bias,
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model.to(device)


def validate_args(
    args: argparse.Namespace,
) -> None:
    if args.input_dim < args.num_qubits:
        raise ValueError(
            "input_dim must be at least num_qubits."
        )

    if args.noise_dim <= 0:
        raise ValueError(
            "noise_dim must be positive."
        )

    if args.latent_std <= 0:
        raise ValueError(
            "latent_std must be positive."
        )

    tt_input_dims = parse_int_list(
        args.tt_noise_input_dims
    )
    tr_input_dims = parse_int_list(
        args.tr_noise_input_dims
    )
    output_dims = parse_int_list(
        args.theta_output_dims
    )
    tt_ranks = parse_int_list(
        args.tt_ranks
    )
    tr_ranks = parse_int_list(
        args.tr_ranks
    )

    if int(
        math.prod(tt_input_dims)
    ) != args.noise_dim:
        raise ValueError(
            "prod(tt_noise_input_dims) "
            "must equal noise_dim."
        )

    if int(
        math.prod(tr_input_dims)
    ) != args.noise_dim:
        raise ValueError(
            "prod(tr_noise_input_dims) "
            "must equal noise_dim."
        )

    target_dim = (
        args.depth
        * args.num_qubits
        * 3
    )

    if int(
        math.prod(output_dims)
    ) != target_dim:
        raise ValueError(
            "prod(theta_output_dims) must equal "
            "depth*num_qubits*3. "
            f"Got {math.prod(output_dims)} "
            f"versus {target_dim}."
        )

    if len(tt_input_dims) != len(
        output_dims
    ):
        raise ValueError(
            "TT input and output factorizations "
            "must have the same number of modes."
        )

    if len(tr_input_dims) != len(
        output_dims
    ):
        raise ValueError(
            "TR input and output factorizations "
            "must have the same number of modes."
        )

    if len(tt_ranks) != len(
        tt_input_dims
    ) + 1:
        raise ValueError(
            "tt_ranks length must equal "
            "len(tt_noise_input_dims)+1."
        )

    if (
        tt_ranks[0] != 1
        or tt_ranks[-1] != 1
    ):
        raise ValueError(
            "Boundary TT ranks must equal 1."
        )

    if len(tr_ranks) != len(
        tr_input_dims
    ):
        raise ValueError(
            "tr_ranks must contain one cyclic "
            "rank per tensor mode."
        )

    if min(tr_ranks) <= 0:
        raise ValueError(
            "All TR ranks must be positive."
        )

    if args.generated_angle_rms <= 0:
        raise ValueError(
            "generated_angle_rms must be positive."
        )

    if args.max_calibration_gain <= 0:
        raise ValueError(
            "max_calibration_gain must be positive."
        )

    if args.base_angle_rms < 0:
        raise ValueError(
            "base_angle_rms must be nonnegative."
        )

    if args.base_anchor_noise < 0:
        raise ValueError(
            "base_anchor_noise must be nonnegative."
        )

    if args.tt_core_init_scale <= 0:
        raise ValueError(
            "tt_core_init_scale must be positive."
        )

    if args.tr_core_init_scale <= 0:
        raise ValueError(
            "tr_core_init_scale must be positive."
        )

    if not (
        0
        <= args.tt_residual_centering_strength
        <= 1
    ):
        raise ValueError(
            "tt_residual_centering_strength "
            "must lie in [0,1]."
        )

    if not (
        0
        <= args.tr_residual_centering_strength
        <= 1
    ):
        raise ValueError(
            "tr_residual_centering_strength "
            "must lie in [0,1]."
        )

    if (
        args.tt_initial_residual_scale <= 0
        or args.tr_initial_residual_scale <= 0
    ):
        raise ValueError(
            "TT/TR residual scales must be positive."
        )

    if args.ntk_points <= 1:
        raise ValueError(
            "ntk_points must be greater than 1."
        )

    if args.input_grad_points <= 0:
        raise ValueError(
            "input_grad_points must be positive."
        )


# ============================================================================
# Reporting
# ============================================================================

PER_SEED_METRICS = [
    "var_x",
    "pairwise_x",
    "grad_entry_mean_sq",
    "grad_norm_sq_mean",
    "grad_norm_sq_median",
    "grad_coordinate_var_mean",
    "input_grad_norm_sq_mean",
    "input_grad_norm_sq_median",
    "ntk_trace_per_sample",
    "centered_ntk_trace_per_sample",
    "ntk_operator_norm",
    "ntk_effective_rank",
    "ntk_normalized_effective_rank",
    "ntk_stable_rank",
    "ntk_spectral_entropy",
    "ntk_normalized_spectral_entropy",
    "ntk_adjacent_gap_ratio",
    "ntk_mp_ks_distance",
    "ntk_mp_bulk_fraction",
    "ntk_mp_upper_edge_ratio",
]


def print_metric_table(
    title: str,
    metric: str,
    models: List[str],
    m_list: List[int],
    results: Dict[
        str,
        Dict[
            int,
            Dict[
                str,
                List[float],
            ],
        ],
    ],
) -> None:
    print(f"\n--- {title} ---")
    print(
        "m | "
        + " | ".join(models)
    )
    print("-" * 120)

    for sample_count in m_list:
        cells = [str(sample_count)]

        for model_name in models:
            mean, std = mean_std(
                results[
                    model_name
                ][
                    sample_count
                ][
                    metric
                ]
            )

            cells.append(
                f"{mean:.4e} ± {std:.2e}"
            )

        print(
            " | ".join(cells)
        )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic concentration, Jacobian, NTK, and "
            "RMT-inspired diagnostics for unstructured, "
            "TT, and TR variational quantum circuits."
        ),
        allow_abbrev=False,
    )

    parser.add_argument(
        "--models",
        type=str,
        default=(
            "unstructured_vqc,"
            "tt_tensor_hyper_vqc,"
            "tr_tensor_hyper_vqc"
        ),
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9",
    )

    parser.add_argument(
        "--m_list",
        type=str,
        default="16,32,64,128",
    )

    parser.add_argument(
        "--data_seed",
        type=int,
        default=12345,
    )

    parser.add_argument(
        "--latent_seed",
        type=int,
        default=2026,
        help=(
            "Dedicated seed for the fixed Gaussian latent "
            "shared across TT/TR and all model seeds."
        ),
    )

    parser.add_argument(
        "--latent_std",
        type=float,
        default=1.0,
        help=(
            "Standard deviation of the fixed Gaussian latent. "
            "The default gives z ~ N(0,I)."
        ),
    )

    parser.add_argument(
        "--normalize_latent_norm",
        action="store_true",
        help=(
            "Normalize the sampled fixed Gaussian latent to unit norm. "
            "Leave disabled for an exact N(0, latent_std^2 I) sample."
        ),
    )

    parser.add_argument(
        "--input_dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num_qubits",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--data_kind",
        type=str,
        default="structured",
        choices=[
            "structured",
            "gaussian",
            "mixed",
        ],
    )

    parser.add_argument(
        "--noise_dim",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--tt_noise_input_dims",
        type=str,
        default="4,4,1",
    )

    parser.add_argument(
        "--tr_noise_input_dims",
        type=str,
        default="4,4,1",
    )

    parser.add_argument(
        "--theta_output_dims",
        type=str,
        default="4,6,6",
        help=(
            "Product must equal depth*num_qubits*3. "
            "For defaults, 4*6*6=144=6*8*3."
        ),
    )

    parser.add_argument(
        "--tt_ranks",
        type=str,
        default="1,2,2,1",
    )

    parser.add_argument(
        "--tr_ranks",
        type=str,
        default="2,2,2",
    )

    parser.add_argument(
        "--tt_core_init_scale",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--tr_core_init_scale",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--base_angle_rms",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--base_anchor_noise",
        type=float,
        default=0.08,
    )

    parser.add_argument(
        "--generated_angle_rms",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--max_calibration_gain",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--tt_initial_residual_scale",
        type=float,
        default=0.075,
    )

    parser.add_argument(
        "--tr_initial_residual_scale",
        type=float,
        default=0.090,
    )

    parser.add_argument(
        "--tt_residual_centering_strength",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--tr_residual_centering_strength",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--residual_global",
        action="store_true",
        help=(
            "Add unconstrained trainable global VQC angles "
            "to TT/TR-generated angles."
        ),
    )

    parser.add_argument(
        "--tensor_bias",
        action="store_true",
        help=(
            "Enable an unconstrained output bias in the TT/TR operator. "
            "Leave disabled for a stricter tensor-manifold experiment."
        ),
    )

    parser.add_argument(
        "--input_grad_points",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--ntk_points",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--save_csv",
        type=str,
        default=(
            "synthetic_rmt_tensorhyper_rmt_results.csv"
        ),
    )

    parser.add_argument(
        "--save_spectra_csv",
        type=str,
        default=(
            "synthetic_rmt_tensorhyper_rmt_spectra.csv"
        ),
    )

    args = parser.parse_args()
    validate_args(args)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    models = [
        item.strip()
        for item in args.models.split(",")
        if item.strip()
    ]

    valid_models = {
        "unstructured_vqc",
        "tt_tensor_hyper_vqc",
        "tr_tensor_hyper_vqc",
    }

    unknown_models = [
        model
        for model in models
        if model not in valid_models
    ]

    if unknown_models:
        raise ValueError(
            "Unknown models: "
            + ", ".join(unknown_models)
        )

    seeds = parse_int_list(
        args.seeds
    )
    m_list = sorted(
        parse_int_list(
            args.m_list
        )
    )

    if not models:
        raise ValueError(
            "At least one model must be selected."
        )

    if not seeds:
        raise ValueError(
            "At least one seed must be selected."
        )

    if (
        not m_list
        or min(m_list) <= 1
    ):
        raise ValueError(
            "m_list must contain integers greater than 1."
        )

    # Fixed dataset shared across all model seeds.
    set_seed(
        args.data_seed
    )

    x_master = synthesize_x(
        sample_count=max(m_list),
        input_dim=args.input_dim,
        device=device,
        kind=args.data_kind,
    )

    # Fixed latent preview for reproducibility.
    latent_preview = fixed_gaussian_latent(
        dimension=args.noise_dim,
        seed=args.latent_seed,
        std=args.latent_std,
        normalize_norm=args.normalize_latent_norm,
    )

    results: Dict[
        str,
        Dict[
            int,
            Dict[
                str,
                List[float],
            ],
        ],
    ] = {
        model_name: {
            sample_count: {
                metric: []
                for metric in (
                    PER_SEED_METRICS
                    + [
                        "anchor_output",
                        "parameter_count",
                    ]
                )
            }
            for sample_count in m_list
        }
        for model_name in models
    }

    raw_rows: List[
        Dict[str, object]
    ] = []

    spectra_rows: List[
        Dict[str, object]
    ] = []

    anchor_states: Dict[
        str,
        List[np.ndarray],
    ] = {
        model_name: []
        for model_name in models
    }

    for seed in seeds:
        set_seed(seed)

        initialized_models = {
            model_name: build_model(
                args,
                model_name,
                device,
            )
            for model_name in models
        }

        parameter_counts = {
            model_name: (
                count_trainable_parameters(
                    model
                )
            )
            for (
                model_name,
                model,
            ) in initialized_models.items()
        }

        for sample_count in m_list:
            x = x_master[
                :sample_count
            ]

            for (
                model_name,
                model,
            ) in initialized_models.items():

                if sample_count == m_list[0]:
                    anchor_states[
                        model_name
                    ].append(
                        evaluate_anchor_state(
                            model=model,
                            x_anchor=x_master[:1],
                            n_wires=args.num_qubits,
                            device=device,
                        )
                    )

                outputs = evaluate_outputs(
                    model=model,
                    x=x,
                    n_wires=args.num_qubits,
                    device=device,
                )

                if outputs.numel() > 1:
                    var_x = float(
                        outputs.var(
                            unbiased=True
                        )
                        .detach()
                        .cpu()
                    )
                else:
                    var_x = 0.0

                pairwise_x = (
                    pairwise_distinguishability(
                        outputs
                    )
                )

                anchor_output = float(
                    outputs[0]
                    .detach()
                    .cpu()
                )

                input_metrics = (
                    input_jacobian_sensitivity(
                        model=model,
                        x=x,
                        n_wires=(
                            args.num_qubits
                        ),
                        device=device,
                        max_points=min(
                            sample_count,
                            args.input_grad_points,
                        ),
                    )
                )

                (
                    jacobian_metrics,
                    raw_eigenvalues,
                    normalized_eigenvalues,
                ) = jacobian_and_ntk_metrics(
                    model=model,
                    x=x,
                    n_wires=args.num_qubits,
                    device=device,
                    max_points=min(
                        sample_count,
                        args.ntk_points,
                    ),
                )

                for eigen_index, (
                    raw_value,
                    normalized_value,
                ) in enumerate(
                    zip(
                        raw_eigenvalues,
                        normalized_eigenvalues,
                    )
                ):
                    spectra_rows.append(
                        {
                            "seed": seed,
                            "m": sample_count,
                            "model": model_name,
                            "eigen_index": eigen_index,
                            "eigenvalue": float(
                                raw_value
                            ),
                            "normalized_eigenvalue": float(
                                normalized_value
                            ),
                            "mp_aspect_ratio": (
                                jacobian_metrics[
                                    "ntk_aspect_ratio"
                                ]
                            ),
                        }
                    )

                per_seed_values = {
                    "var_x": var_x,
                    "pairwise_x": pairwise_x,
                    "anchor_output": anchor_output,
                    "parameter_count": float(
                        parameter_counts[
                            model_name
                        ]
                    ),
                    **input_metrics,
                    **jacobian_metrics,
                }

                for (
                    metric_name,
                    metric_value,
                ) in per_seed_values.items():
                    results[
                        model_name
                    ][
                        sample_count
                    ].setdefault(
                        metric_name,
                        [],
                    ).append(
                        float(metric_value)
                    )

                raw_rows.append(
                    {
                        "seed": seed,
                        "m": sample_count,
                        "model": model_name,
                        **per_seed_values,
                    }
                )

        del initialized_models

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Ensemble statistics across model seeds at one fixed input.
    for model_name in models:
        frame_potential = (
            state_second_frame_potential(
                anchor_states[
                    model_name
                ]
            )
        )

        hilbert_dim = (
            2 ** args.num_qubits
        )

        haar_state_f2 = (
            2.0
            / (
                hilbert_dim
                * (hilbert_dim + 1)
            )
        )

        for sample_count in m_list:
            anchor_outputs = results[
                model_name
            ][
                sample_count
            ][
                "anchor_output"
            ]

            results[
                model_name
            ][
                sample_count
            ][
                "var_theta_anchor"
            ] = [
                sample_variance(
                    anchor_outputs
                )
            ]

            results[
                model_name
            ][
                sample_count
            ][
                "pairwise_theta_anchor"
            ] = [
                pairwise_distinguishability_numpy(
                    anchor_outputs
                )
            ]

            results[
                model_name
            ][
                sample_count
            ][
                "state_frame_potential_2"
            ] = [
                frame_potential
            ]

            results[
                model_name
            ][
                sample_count
            ][
                "state_frame_potential_2_over_haar"
            ] = [
                frame_potential
                / max(
                    haar_state_f2,
                    1e-16,
                )
            ]

    print(
        "\n=== Synthetic TT/TR TensorHyper-VQC RMT statistics ==="
    )
    print(
        f"Device: {device}"
    )
    print(
        f"Data seed: {args.data_seed}"
    )
    print(
        f"Fixed latent seed: {args.latent_seed}"
    )
    print(
        f"Fixed latent standard deviation: {args.latent_std}"
    )
    print(
        "Fixed latent trainable: False"
    )
    print(
        "Fixed latent shared across model seeds: True"
    )
    print(
        f"Fixed latent norm: "
        f"{float(torch.linalg.vector_norm(latent_preview)):.6f}"
    )
    print(
        "Dataset fixed across model seeds: True"
    )

    print(
        "\n--- Trainable parameter counts ---"
    )
    for model_name in models:
        counts = results[
            model_name
        ][
            m_list[0]
        ][
            "parameter_count"
        ]

        unique_counts = sorted(
            set(
                int(value)
                for value in counts
            )
        )

        print(
            f"{model_name}: {unique_counts}"
        )

    print_metric_table(
        (
            "Pointwise ensemble variance "
            "Var_theta[f_theta(x_anchor)]"
        ),
        "var_theta_anchor",
        models,
        m_list,
        results,
    )

    print_metric_table(
        (
            "Pointwise ensemble pairwise "
            "distinguishability"
        ),
        "pairwise_theta_anchor",
        models,
        m_list,
        results,
    )

    table_titles = {
        "var_x": (
            "Functional output variance "
            "Var_x[f_theta(x)]"
        ),
        "pairwise_x": (
            "Functional pairwise output "
            "distinguishability"
        ),
        "grad_entry_mean_sq": (
            "Mean squared gradient entry"
        ),
        "grad_norm_sq_mean": (
            "Mean squared gradient norm"
        ),
        "grad_norm_sq_median": (
            "Median squared gradient norm"
        ),
        "grad_coordinate_var_mean": (
            "Mean coordinate-wise gradient "
            "variance across inputs"
        ),
        "input_grad_norm_sq_mean": (
            "Mean input-gradient norm squared"
        ),
        "input_grad_norm_sq_median": (
            "Median input-gradient norm squared"
        ),
        "ntk_trace_per_sample": (
            "Empirical NTK trace per sample"
        ),
        "centered_ntk_trace_per_sample": (
            "Centered empirical NTK trace "
            "per sample"
        ),
        "ntk_operator_norm": (
            "Empirical NTK operator norm"
        ),
        "ntk_effective_rank": (
            "Empirical NTK effective rank"
        ),
        "ntk_normalized_effective_rank": (
            "Normalized empirical NTK "
            "effective rank"
        ),
        "ntk_stable_rank": (
            "Empirical NTK stable rank"
        ),
        "ntk_spectral_entropy": (
            "Empirical NTK spectral entropy"
        ),
        "ntk_normalized_spectral_entropy": (
            "Normalized empirical NTK "
            "spectral entropy"
        ),
        "ntk_adjacent_gap_ratio": (
            "NTK adjacent-gap ratio"
        ),
        "ntk_mp_ks_distance": (
            "KS distance to "
            "Marchenko-Pastur law"
        ),
        "ntk_mp_bulk_fraction": (
            "Fraction inside MP bulk support"
        ),
        "ntk_mp_upper_edge_ratio": (
            "Largest eigenvalue / "
            "MP upper edge"
        ),
    }

    for metric_name in PER_SEED_METRICS:
        print_metric_table(
            table_titles[
                metric_name
            ],
            metric_name,
            models,
            m_list,
            results,
        )

    try:
        import pandas as pd

        raw_dataframe = pd.DataFrame(
            raw_rows
        )
        spectra_dataframe = pd.DataFrame(
            spectra_rows
        )

        ensemble_rows: List[
            Dict[str, object]
        ] = []

        for model_name in models:
            for sample_count in m_list:
                ensemble_rows.append(
                    {
                        "model": model_name,
                        "m": sample_count,
                        "var_theta_anchor": (
                            results[
                                model_name
                            ][
                                sample_count
                            ][
                                "var_theta_anchor"
                            ][0]
                        ),
                        "pairwise_theta_anchor": (
                            results[
                                model_name
                            ][
                                sample_count
                            ][
                                "pairwise_theta_anchor"
                            ][0]
                        ),
                        "state_frame_potential_2": (
                            results[
                                model_name
                            ][
                                sample_count
                            ][
                                "state_frame_potential_2"
                            ][0]
                        ),
                        "state_frame_potential_2_over_haar": (
                            results[
                                model_name
                            ][
                                sample_count
                            ][
                                "state_frame_potential_2_over_haar"
                            ][0]
                        ),
                        "data_seed": args.data_seed,
                        "latent_seed": args.latent_seed,
                        "latent_std": args.latent_std,
                        "normalize_latent_norm": (
                            args.normalize_latent_norm
                        ),
                    }
                )

        ensemble_dataframe = pd.DataFrame(
            ensemble_rows
        )

        raw_path = args.save_csv

        if raw_path.lower().endswith(
            ".csv"
        ):
            ensemble_path = (
                raw_path[:-4]
                + "_ensemble_summary.csv"
            )
        else:
            ensemble_path = (
                raw_path
                + "_ensemble_summary.csv"
            )

        raw_dataframe.to_csv(
            raw_path,
            index=False,
        )
        ensemble_dataframe.to_csv(
            ensemble_path,
            index=False,
        )
        spectra_dataframe.to_csv(
            args.save_spectra_csv,
            index=False,
        )

        print(
            f"\nSaved per-seed results to: "
            f"{raw_path}"
        )
        print(
            f"Saved ensemble summaries to: "
            f"{ensemble_path}"
        )
        print(
            f"Saved NTK spectra to: "
            f"{args.save_spectra_csv}"
        )

    except Exception as error:
        print(
            f"\nCSV export failed: {error}"
        )


if __name__ == "__main__":
    main()
