#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RMT-compliant TT-based TensorHyper-VQC for quantum-dot classification.

This implementation enforces two separate input paths:

1. Gaussian latent input to the Tensor-Train hypernetwork

       z ~ N(0, I)
       z -> TensorTrainLayer -> VQC rotation parameters

2. Amplitude encoding of the complete 2500-dimensional image

       x in R^2500
       x -> zero padding + L2 normalization -> AmplitudeEncoder -> VQC

The image is NEVER passed to the TT layer, and the Gaussian latent is NEVER
used as the quantum-state input.

Additional features:
  - Ring, MPS, and tree VQC topologies
  - Optional residual global VQC angles
  - Fixed-per-sample or shared-fixed Gaussian latent variables
  - Composable quantum noise
  - Memory-safe quantum microbatching
  - Accuracy, Macro-F1, Weighted-F1, ECE, and confidence metrics
  - CSV logging and best-checkpoint selection

Requires:
  torch, torchquantum, numpy
"""

import argparse
import csv
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchquantum as tq
import torchquantum.functional as tqf
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Reproducibility
# =============================================================================
DEFAULT_SEED = 1324
torch.manual_seed(DEFAULT_SEED)
np.random.seed(DEFAULT_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(DEFAULT_SEED)


# =============================================================================
# Utilities
# =============================================================================
def parse_int_list(value: str) -> List[int]:
    value = str(value).strip()

    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]

    if not value:
        return []

    return [
        int(token.strip())
        for token in value.split(",")
        if token.strip()
    ]


def build_noise_cfg(args) -> Tuple[Dict[str, float], List[str]]:
    models = [
        model.strip().lower()
        for model in str(args.noise_models).split(",")
        if model.strip()
    ]

    if "none" in models:
        models = []

    cfg = {
        "depol": args.p_depol if "depol" in models else 0.0,
        "dephase": args.p_dephase if "dephase" in models else 0.0,
        "pauli_px": args.pauli_px if "pauli" in models else 0.0,
        "pauli_py": args.pauli_py if "pauli" in models else 0.0,
        "pauli_pz": args.pauli_pz if "pauli" in models else 0.0,
        "overrot_sigma": (
            args.overrot_sigma if "overrot" in models else 0.0
        ),
        "p_twopauli": (
            args.p_twopauli if "twopauli" in models else 0.0
        ),
        "p_readout": (
            args.p_readout if "readout" in models else 0.0
        ),
    }

    return cfg, models


def validate_configuration(args) -> None:
    tt_input_size = math.prod(args.tt_input_dim)
    tt_output_size = math.prod(args.tt_output_dim)
    required_output_size = (
        args.depth_vqc
        * args.num_qubits
        * 3
    )

    if len(args.tt_input_dim) == 0:
        raise ValueError("tt_input_dim cannot be empty.")

    if len(args.tt_output_dim) != len(args.tt_input_dim):
        raise ValueError(
            "tt_input_dim and tt_output_dim must have the same order."
        )

    if len(args.tt_ranks) != len(args.tt_input_dim) + 1:
        raise ValueError(
            "tt_ranks must contain TT order + 1 entries."
        )

    if args.tt_ranks[0] != 1 or args.tt_ranks[-1] != 1:
        raise ValueError(
            "A standard TT layer requires boundary ranks r_0 = r_d = 1."
        )

    if tt_output_size != required_output_size:
        raise ValueError(
            f"TT output size is {tt_output_size}, but the VQC requires "
            f"{required_output_size} = depth_vqc × num_qubits × 3. "
            "Change --tt_output_dim or the VQC dimensions."
        )

    if args.num_qubits < math.ceil(math.log2(args.image_dim)):
        raise ValueError(
            f"{args.image_dim}-dimensional amplitude encoding requires at "
            f"least ceil(log2({args.image_dim})) = "
            f"{math.ceil(math.log2(args.image_dim))} qubits."
        )

    if tt_input_size <= 0:
        raise ValueError("The Gaussian latent dimension must be positive.")

    if args.quantum_batch_size <= 0:
        raise ValueError("quantum_batch_size must be positive.")

    if args.num_classes <= 1:
        raise ValueError("num_classes must be at least 2.")

    if args.ece_bins <= 0:
        raise ValueError("ece_bins must be positive.")

    for name in [
        "p_depol",
        "p_dephase",
        "pauli_px",
        "pauli_py",
        "pauli_pz",
        "p_twopauli",
        "p_readout",
    ]:
        value = float(getattr(args, name))

        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must lie in [0, 1].")

    if args.pauli_px + args.pauli_py + args.pauli_pz > 1.0:
        raise ValueError(
            "pauli_px + pauli_py + pauli_pz must not exceed 1."
        )

    if args.overrot_sigma < 0:
        raise ValueError("overrot_sigma must be non-negative.")


def prepare_amplitude_vectors(
    images: torch.Tensor,
    n_wires: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Convert flattened images into exact 2**n_wires amplitude vectors.

    Steps:
      1. Flatten each image.
      2. Zero-pad to Hilbert-space dimension.
      3. L2-normalize each sample.

    This guarantees that all 2500 image features enter the quantum state.
    """
    images = images.reshape(images.size(0), -1).float()
    state_dim = 2 ** n_wires
    feature_dim = images.size(1)

    if feature_dim > state_dim:
        raise ValueError(
            f"Image dimension {feature_dim} exceeds the {state_dim}-dimensional "
            f"Hilbert space of {n_wires} qubits."
        )

    if feature_dim < state_dim:
        padding = torch.zeros(
            images.size(0),
            state_dim - feature_dim,
            dtype=images.dtype,
            device=images.device,
        )
        images = torch.cat([images, padding], dim=1)

    norms = torch.linalg.vector_norm(
        images,
        ord=2,
        dim=1,
        keepdim=True,
    )

    zero_rows = norms.squeeze(1) <= eps

    if zero_rows.any():
        images = images.clone()
        images[zero_rows, 0] = 1.0
        norms = torch.linalg.vector_norm(
            images,
            ord=2,
            dim=1,
            keepdim=True,
        )

    return images / norms.clamp_min(eps)


# =============================================================================
# Tensor-Train hypernetwork
# =============================================================================
class TensorTrainLayer(nn.Module):
    """
    Tensor-Train linear map from Gaussian latent vectors to VQC parameters.

    Input:
      z: [batch, prod(input_dims)]

    Output:
      theta: [batch, prod(output_dims)]
    """

    def __init__(
        self,
        input_dims: List[int],
        output_dims: List[int],
        tt_ranks: List[int],
        init_scale: float = 0.01,
    ):
        super().__init__()

        order = len(input_dims)

        if len(output_dims) != order:
            raise ValueError(
                "input_dims and output_dims must have identical order."
            )

        if len(tt_ranks) != order + 1:
            raise ValueError(
                "tt_ranks must have length order + 1."
            )

        self.input_dims = list(input_dims)
        self.output_dims = list(output_dims)
        self.tt_ranks = list(tt_ranks)
        self.input_size = math.prod(input_dims)
        self.output_size = math.prod(output_dims)

        self.tt_cores = nn.ParameterList()

        for mode in range(order):
            rank_left = tt_ranks[mode]
            rank_right = tt_ranks[mode + 1]
            input_mode = input_dims[mode]
            output_mode = output_dims[mode]

            core = nn.Parameter(
                torch.empty(
                    rank_left,
                    input_mode,
                    output_mode,
                    rank_right,
                )
            )

            nn.init.xavier_uniform_(core)
            core.data.mul_(init_scale / max(core.data.std().item(), 1e-12))
            self.tt_cores.append(core)

        self.bias = nn.Parameter(
            torch.zeros(self.output_size)
        )

    def forward(self, gaussian_latent: torch.Tensor) -> torch.Tensor:
        if gaussian_latent.dim() != 2:
            raise ValueError(
                "Gaussian TT input must have shape [batch, latent_dim]."
            )

        if gaussian_latent.size(1) != self.input_size:
            raise ValueError(
                f"TT expected Gaussian latent dimension {self.input_size}, "
                f"but received {gaussian_latent.size(1)}."
            )

        batch_size = gaussian_latent.size(0)
        latent_tensor = gaussian_latent.reshape(
            batch_size,
            *self.input_dims,
        )

        batch_symbol = "B"
        symbols = [
            character
            for character in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            if character != batch_symbol
        ]

        order = len(self.input_dims)
        required_symbols = 3 * order + 1

        if required_symbols > len(symbols):
            raise ValueError(
                "TT order is too large for the einsum symbol implementation."
            )

        input_symbols = symbols[:order]
        output_symbols = symbols[order:2 * order]
        rank_symbols = symbols[2 * order:3 * order + 1]

        input_term = batch_symbol + "".join(input_symbols)

        core_terms = [
            (
                rank_symbols[mode]
                + input_symbols[mode]
                + output_symbols[mode]
                + rank_symbols[mode + 1]
            )
            for mode in range(order)
        ]

        output_term = batch_symbol + "".join(output_symbols)

        equation = (
            input_term
            + ","
            + ",".join(core_terms)
            + "->"
            + output_term
        )

        output = torch.einsum(
            equation,
            latent_tensor,
            *self.tt_cores,
        )

        return output.reshape(batch_size, -1) + self.bias


# =============================================================================
# Quantum circuits
# =============================================================================
class BaseAmplitudeVQC(tq.QuantumModule):
    """
    Base VQC that always uses amplitude encoding.

    The image input must be an already prepared normalized amplitude vector
    with shape [batch, 2**n_wires].
    """

    def __init__(
        self,
        n_wires: int = 12,
        n_qlayers: int = 6,
        topology: str = "ring",
        add_fc: bool = True,
        out_features: int = 2,
        use_residual_global_angles: bool = True,
    ):
        super().__init__()

        if topology not in {"ring", "mps", "tree"}:
            raise ValueError(
                f"Unsupported topology: {topology}."
            )

        self.n_wires = n_wires
        self.n_qlayers = n_qlayers
        self.topology = topology
        self.add_fc = add_fc
        self.use_residual_global_angles = use_residual_global_angles
        self.noise_cfg: Dict[str, float] = {}

        self.encoder = tq.AmplitudeEncoder()

        self.global_angles = nn.Parameter(
            torch.randn(
                n_qlayers,
                n_wires,
                3,
            ) * 0.1
        )

        self.measure = tq.MeasureAll(tq.PauliZ)

        if add_fc:
            self.fc = nn.Linear(
                n_wires,
                out_features,
            )

    def reset_quantum_device(self, batch_size: int) -> None:
        self.q_device.reset_states(batch_size)

    # -------------------------------------------------------------------------
    # Noise
    # -------------------------------------------------------------------------
    def _apply_single_qubit_depolarizing(self, probability: float) -> None:
        if probability <= 0:
            return

        for wire in range(self.n_wires):
            if torch.rand((), device=self.q_device.device) < probability:
                error_id = torch.randint(
                    0,
                    3,
                    (),
                    device=self.q_device.device,
                ).item()

                operation = (
                    tqf.x
                    if error_id == 0
                    else tqf.y
                    if error_id == 1
                    else tqf.z
                )

                operation(
                    self.q_device,
                    wires=wire,
                    static=self.static_mode,
                    parent_graph=self.graph,
                )

    def _apply_single_qubit_dephasing(self, probability: float) -> None:
        if probability <= 0:
            return

        for wire in range(self.n_wires):
            if torch.rand((), device=self.q_device.device) < probability:
                tqf.z(
                    self.q_device,
                    wires=wire,
                    static=self.static_mode,
                    parent_graph=self.graph,
                )

    def _apply_single_qubit_pauli(
        self,
        px: float,
        py: float,
        pz: float,
    ) -> None:
        if px + py + pz <= 0:
            return

        for wire in range(self.n_wires):
            random_value = torch.rand(
                (),
                device=self.q_device.device,
            )

            if random_value < px:
                tqf.x(
                    self.q_device,
                    wires=wire,
                    static=self.static_mode,
                    parent_graph=self.graph,
                )
            elif random_value < px + py:
                tqf.y(
                    self.q_device,
                    wires=wire,
                    static=self.static_mode,
                    parent_graph=self.graph,
                )
            elif random_value < px + py + pz:
                tqf.z(
                    self.q_device,
                    wires=wire,
                    static=self.static_mode,
                    parent_graph=self.graph,
                )

    def _apply_two_qubit_pauli(
        self,
        wires: List[int],
        probability: float,
    ) -> None:
        if probability <= 0:
            return

        if torch.rand((), device=self.q_device.device) < probability:
            operations = [tqf.i, tqf.x, tqf.y, tqf.z]

            operation_a = torch.randint(
                0,
                4,
                (),
                device=self.q_device.device,
            ).item()

            operation_b = torch.randint(
                0,
                4,
                (),
                device=self.q_device.device,
            ).item()

            operations[operation_a](
                self.q_device,
                wires=wires[0],
                static=self.static_mode,
                parent_graph=self.graph,
            )

            operations[operation_b](
                self.q_device,
                wires=wires[1],
                static=self.static_mode,
                parent_graph=self.graph,
            )

    def _apply_readout_error(
        self,
        measurements: torch.Tensor,
        probability: float,
    ) -> torch.Tensor:
        if probability <= 0:
            return measurements

        sign_flip_mask = (
            torch.rand_like(measurements) < probability
        ).to(measurements.dtype)

        return measurements * (
            1.0 - 2.0 * sign_flip_mask
        )

    def _inject_single_qubit_noise(self) -> None:
        self._apply_single_qubit_depolarizing(
            float(self.noise_cfg.get("depol", 0.0))
        )

        self._apply_single_qubit_dephasing(
            float(self.noise_cfg.get("dephase", 0.0))
        )

        self._apply_single_qubit_pauli(
            float(self.noise_cfg.get("pauli_px", 0.0)),
            float(self.noise_cfg.get("pauli_py", 0.0)),
            float(self.noise_cfg.get("pauli_pz", 0.0)),
        )

    # -------------------------------------------------------------------------
    # Entanglement
    # -------------------------------------------------------------------------
    def _apply_cnot(
        self,
        control: int,
        target: int,
        two_qubit_error_probability: float,
    ) -> None:
        wires = [control, target]

        tqf.cnot(
            self.q_device,
            wires=wires,
            static=self.static_mode,
            parent_graph=self.graph,
        )

        self._apply_two_qubit_pauli(
            wires,
            two_qubit_error_probability,
        )

    def _entangle_ring(
        self,
        two_qubit_error_probability: float,
    ) -> None:
        for wire in range(self.n_wires - 1):
            self._apply_cnot(
                wire,
                wire + 1,
                two_qubit_error_probability,
            )

        self._apply_cnot(
            self.n_wires - 1,
            0,
            two_qubit_error_probability,
        )

    def _entangle_mps(
        self,
        two_qubit_error_probability: float,
    ) -> None:
        for wire in range(self.n_wires - 1):
            self._apply_cnot(
                wire,
                wire + 1,
                two_qubit_error_probability,
            )

    def _entangle_tree(
        self,
        two_qubit_error_probability: float,
    ) -> None:
        stride = 1

        while stride < self.n_wires:
            for control in range(0, self.n_wires, 2 * stride):
                target = control + stride

                if target < self.n_wires:
                    self._apply_cnot(
                        control,
                        target,
                        two_qubit_error_probability,
                    )

            stride *= 2

    def _apply_entanglement(
        self,
        two_qubit_error_probability: float,
    ) -> None:
        if self.topology == "ring":
            self._entangle_ring(two_qubit_error_probability)
        elif self.topology == "mps":
            self._entangle_mps(two_qubit_error_probability)
        else:
            self._entangle_tree(two_qubit_error_probability)

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    @tq.static_support
    def forward(
        self,
        amplitude_vectors: torch.Tensor,
        q_device: tq.QuantumDevice,
        generated_angles: torch.Tensor,
    ) -> torch.Tensor:
        """
        amplitude_vectors:
          [batch, 2**n_wires], padded and L2-normalized image amplitudes.

        generated_angles:
          [batch, n_qlayers, n_wires, 3], produced only from Gaussian latent z.
        """
        if generated_angles.dim() != 4:
            raise ValueError(
                "generated_angles must have shape "
                "[batch, layers, wires, 3]."
            )

        expected_shape = (
            amplitude_vectors.size(0),
            self.n_qlayers,
            self.n_wires,
            3,
        )

        if tuple(generated_angles.shape) != expected_shape:
            raise ValueError(
                f"Expected generated angles with shape {expected_shape}, "
                f"but received {tuple(generated_angles.shape)}."
            )

        self.q_device = q_device
        batch_size = amplitude_vectors.size(0)
        self.reset_quantum_device(batch_size)

        # Requirement 2: full image enters the VQC through amplitude encoding.
        self.encoder(
            self.q_device,
            amplitude_vectors,
        )

        if self.use_residual_global_angles:
            total_angles = (
                generated_angles
                + self.global_angles.unsqueeze(0)
            )
        else:
            total_angles = generated_angles

        overrotation_sigma = float(
            self.noise_cfg.get("overrot_sigma", 0.0)
        )

        two_qubit_error_probability = float(
            self.noise_cfg.get("p_twopauli", 0.0)
        )

        for layer in range(self.n_qlayers):
            for wire in range(self.n_wires):
                rx_angle = total_angles[:, layer, wire, 0]
                ry_angle = total_angles[:, layer, wire, 1]
                rz_angle = total_angles[:, layer, wire, 2]

                if overrotation_sigma > 0:
                    rx_angle = (
                        rx_angle
                        + torch.randn_like(rx_angle)
                        * overrotation_sigma
                    )
                    ry_angle = (
                        ry_angle
                        + torch.randn_like(ry_angle)
                        * overrotation_sigma
                    )
                    rz_angle = (
                        rz_angle
                        + torch.randn_like(rz_angle)
                        * overrotation_sigma
                    )

                tqf.rx(
                    self.q_device,
                    wires=wire,
                    params=rx_angle,
                    static=self.static_mode,
                    parent_graph=self.graph,
                )

                tqf.ry(
                    self.q_device,
                    wires=wire,
                    params=ry_angle,
                    static=self.static_mode,
                    parent_graph=self.graph,
                )

                tqf.rz(
                    self.q_device,
                    wires=wire,
                    params=rz_angle,
                    static=self.static_mode,
                    parent_graph=self.graph,
                )

            self._inject_single_qubit_noise()
            self._apply_entanglement(
                two_qubit_error_probability
            )

        measurements = self.measure(
            self.q_device
        )

        measurements = self._apply_readout_error(
            measurements,
            float(self.noise_cfg.get("p_readout", 0.0)),
        )

        if self.add_fc:
            return self.fc(measurements)

        return measurements[:, :2]


# =============================================================================
# TT-based TensorHyper-VQC wrapper
# =============================================================================
class TTTensorHyperVQC(nn.Module):
    """
    Gaussian latent -> TT -> VQC parameters
    Image -> amplitude encoding -> parameterized VQC

    The two data paths are explicitly separated.
    """

    def __init__(
        self,
        tt_input_dims: List[int],
        tt_output_dims: List[int],
        tt_ranks: List[int],
        n_wires: int,
        n_qlayers: int,
        topology: str,
        num_classes: int,
        add_fc: bool,
        use_residual_global_angles: bool,
    ):
        super().__init__()

        self.tt = TensorTrainLayer(
            input_dims=tt_input_dims,
            output_dims=tt_output_dims,
            tt_ranks=tt_ranks,
        )

        self.vqc = BaseAmplitudeVQC(
            n_wires=n_wires,
            n_qlayers=n_qlayers,
            topology=topology,
            add_fc=add_fc,
            out_features=num_classes,
            use_residual_global_angles=use_residual_global_angles,
        )

    def forward(
        self,
        images: torch.Tensor,
        gaussian_latents: torch.Tensor,
        q_device: tq.QuantumDevice,
    ) -> torch.Tensor:
        batch_size = images.size(0)

        if gaussian_latents.size(0) != batch_size:
            raise ValueError(
                "Images and Gaussian latents must have the same batch size."
            )

        # Requirement 1: only Gaussian noise is passed to the TT layer.
        generated_angles = self.tt(
            gaussian_latents
        ).reshape(
            batch_size,
            self.vqc.n_qlayers,
            self.vqc.n_wires,
            3,
        )

        # Requirement 2: the complete image is transformed into amplitudes.
        amplitude_vectors = prepare_amplitude_vectors(
            images,
            n_wires=self.vqc.n_wires,
        )

        return self.vqc(
            amplitude_vectors=amplitude_vectors,
            q_device=q_device,
            generated_angles=generated_angles,
        )


# =============================================================================
# Dataset
# =============================================================================
def make_gaussian_latents(
    num_samples: int,
    latent_dim: int,
    mode: str,
    seed: int,
) -> torch.Tensor:
    """
    Produce Gaussian TT inputs.

    fixed_per_sample:
      Each image receives one independently sampled z_i that is fixed across
      all epochs.

    shared_fixed:
      One z_0 is sampled and shared by every image, giving one common generated
      circuit across the dataset.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    if mode == "fixed_per_sample":
        return torch.randn(
            num_samples,
            latent_dim,
            generator=generator,
        )

    if mode == "shared_fixed":
        shared_latent = torch.randn(
            1,
            latent_dim,
            generator=generator,
        )

        return shared_latent.expand(
            num_samples,
            -1,
        ).clone()

    raise ValueError(
        f"Unsupported Gaussian mode: {mode}."
    )


def load_quantum_dot_data(
    data_root: str,
    image_dim: int,
    gaussian_dim: int,
    gaussian_mode: str,
    gaussian_seed: int,
) -> Tuple[TensorDataset, TensorDataset]:
    noisy_path = os.path.join(
        data_root,
        "csds.npy",
    )

    labels_path = os.path.join(
        data_root,
        "labels.npy",
    )

    images = np.load(noisy_path)
    labels = np.load(labels_path)

    images = images.reshape(
        -1,
        image_dim,
    ).astype(np.float32)

    labels = labels.astype(np.int64)

    gaussian_latents = make_gaussian_latents(
        num_samples=len(labels),
        latent_dim=gaussian_dim,
        mode=gaussian_mode,
        seed=gaussian_seed,
    )

    split_index = int(
        0.9 * len(labels)
    )

    train_dataset = TensorDataset(
        torch.from_numpy(images[:split_index]),
        gaussian_latents[:split_index],
        torch.from_numpy(labels[:split_index]),
    )

    test_dataset = TensorDataset(
        torch.from_numpy(images[split_index:]),
        gaussian_latents[split_index:],
        torch.from_numpy(labels[split_index:]),
    )

    return train_dataset, test_dataset


# =============================================================================
# Metrics
# =============================================================================
def multiclass_f1_scores(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    eps: float = 1e-12,
) -> Tuple[float, float, List[float]]:
    predictions = predictions.detach().view(-1).long().cpu()
    targets = targets.detach().view(-1).long().cpu()

    per_class_f1 = []
    supports = []

    for class_id in range(num_classes):
        predicted_positive = predictions == class_id
        actual_positive = targets == class_id

        true_positive = torch.sum(
            predicted_positive & actual_positive
        ).item()

        false_positive = torch.sum(
            predicted_positive & (~actual_positive)
        ).item()

        false_negative = torch.sum(
            (~predicted_positive) & actual_positive
        ).item()

        support = torch.sum(
            actual_positive
        ).item()

        precision = (
            true_positive
            / max(true_positive + false_positive, eps)
        )

        recall = (
            true_positive
            / max(true_positive + false_negative, eps)
        )

        if precision + recall <= eps:
            f1 = 0.0
        else:
            f1 = (
                2.0
                * precision
                * recall
                / (precision + recall)
            )

        per_class_f1.append(float(f1))
        supports.append(int(support))

    macro_f1 = float(
        np.mean(per_class_f1)
    )

    total_support = max(
        sum(supports),
        1,
    )

    weighted_f1 = float(
        sum(
            f1 * support
            for f1, support in zip(per_class_f1, supports)
        )
        / total_support
    )

    return macro_f1, weighted_f1, per_class_f1


def expected_calibration_error(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int,
) -> float:
    probabilities = probabilities.detach().float().cpu()
    targets = targets.detach().long().cpu()

    confidences, predictions = torch.max(
        probabilities,
        dim=1,
    )

    correctness = predictions.eq(
        targets
    ).float()

    bin_boundaries = torch.linspace(
        0.0,
        1.0,
        num_bins + 1,
    )

    ece = torch.zeros(
        (),
        dtype=torch.float32,
    )

    for bin_index in range(num_bins):
        lower = bin_boundaries[bin_index]
        upper = bin_boundaries[bin_index + 1]

        if bin_index == 0:
            in_bin = (
                (confidences >= lower)
                & (confidences <= upper)
            )
        else:
            in_bin = (
                (confidences > lower)
                & (confidences <= upper)
            )

        fraction = in_bin.float().mean()

        if fraction.item() > 0:
            bin_accuracy = correctness[in_bin].mean()
            bin_confidence = confidences[in_bin].mean()

            ece += (
                fraction
                * torch.abs(
                    bin_accuracy
                    - bin_confidence
                )
            )

    return float(ece.item())


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ece_bins: int,
) -> Dict[str, object]:
    probabilities = torch.softmax(
        logits.float(),
        dim=1,
    )

    predictions = torch.argmax(
        probabilities,
        dim=1,
    )

    accuracy = float(
        predictions.eq(targets).float().mean().item()
    )

    macro_f1, weighted_f1, per_class_f1 = multiclass_f1_scores(
        predictions,
        targets,
        num_classes,
    )

    ece = expected_calibration_error(
        probabilities,
        targets,
        ece_bins,
    )

    mean_confidence = float(
        probabilities.max(dim=1).values.mean().item()
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "ece": ece,
        "mean_confidence": mean_confidence,
        "per_class_f1": per_class_f1,
    }


# =============================================================================
# Training
# =============================================================================
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_qubits: int,
    num_classes: int,
    ece_bins: int,
    quantum_batch_size: int,
    optimizer=None,
) -> Dict[str, object]:
    """
    Run one epoch using quantum microbatches.

    The loss of each microbatch is weighted by its fraction of the outer
    DataLoader batch, preserving the gradient of the original batch mean.
    """
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_targets = []

    for images_outer, gaussian_outer, targets_outer in loader:
        outer_batch_size = images_outer.size(0)

        if training:
            optimizer.zero_grad(set_to_none=True)

        for start in range(
            0,
            outer_batch_size,
            quantum_batch_size,
        ):
            end = min(
                start + quantum_batch_size,
                outer_batch_size,
            )

            images = images_outer[start:end].to(
                device,
                non_blocking=True,
            )

            gaussian_latents = gaussian_outer[start:end].to(
                device,
                non_blocking=True,
            )

            targets = targets_outer[start:end].to(
                device,
                non_blocking=True,
            )

            microbatch_size = images.size(0)

            q_device = tq.QuantumDevice(
                n_wires=num_qubits,
                bsz=microbatch_size,
            ).to(device)

            if training:
                logits = model(
                    images,
                    gaussian_latents,
                    q_device,
                )

                loss = criterion(
                    logits,
                    targets,
                )

                scaled_loss = (
                    loss
                    * microbatch_size
                    / outer_batch_size
                )

                scaled_loss.backward()
            else:
                with torch.no_grad():
                    logits = model(
                        images,
                        gaussian_latents,
                        q_device,
                    )

                    loss = criterion(
                        logits,
                        targets,
                    )

            total_loss += (
                loss.item()
                * microbatch_size
            )

            total_samples += microbatch_size

            all_logits.append(
                logits.detach().cpu()
            )

            all_targets.append(
                targets.detach().cpu()
            )

            del (
                q_device,
                logits,
                loss,
                images,
                gaussian_latents,
                targets,
            )

        if training:
            optimizer.step()

    logits_full = torch.cat(
        all_logits,
        dim=0,
    )

    targets_full = torch.cat(
        all_targets,
        dim=0,
    )

    metrics = compute_metrics(
        logits_full,
        targets_full,
        num_classes,
        ece_bins,
    )

    average_loss = (
        total_loss
        / max(total_samples, 1)
    )

    metrics["loss"] = average_loss
    metrics["nll"] = average_loss

    return metrics


def metric_is_better(
    current: float,
    best: float,
    metric_name: str,
) -> bool:
    if metric_name in {
        "test_loss",
        "test_nll",
        "test_ece",
    }:
        return current < best

    return current > best


def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    args,
    train_metrics: Dict[str, object],
    test_metrics: Dict[str, object],
) -> None:
    os.makedirs(
        os.path.dirname(path) or ".",
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": vars(args),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        },
        path,
    )


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RMT-compliant TT-based TensorHyper-VQC with Gaussian TT input "
            "and amplitude-encoded quantum-dot images."
        )
    )

    parser.add_argument(
        "--data_root",
        default="./mlqe_2023_edx/week1/dataset",
        type=str,
    )

    parser.add_argument(
        "--save_path",
        default="models",
        type=str,
    )

    parser.add_argument(
        "--image_dim",
        default=2500,
        type=int,
    )

    parser.add_argument(
        "--num_qubits",
        default=12,
        type=int,
    )

    parser.add_argument(
        "--depth_vqc",
        default=6,
        type=int,
    )

    parser.add_argument(
        "--num_classes",
        default=2,
        type=int,
    )

    parser.add_argument(
        "--model_kind",
        default="mps",
        choices=[
            "ring",
            "mps",
            "tree",
        ],
    )

    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Outer optimization batch size.",
    )

    parser.add_argument(
        "--quantum_batch_size",
        default=2,
        type=int,
        help=(
            "Statevector simulator microbatch size. Use 1 or 2 for large "
            "qubit counts."
        ),
    )

    parser.add_argument(
        "--num_epochs",
        default=21,
        type=int,
    )

    parser.add_argument(
        "--lr",
        default=3e-3,
        type=float,
    )

    parser.add_argument(
        "--tt_input_dim",
        type=parse_int_list,
        default="5,10,5,10",
        help=(
            "TT input modes. Their product is the Gaussian latent dimension."
        ),
    )

    parser.add_argument(
        "--tt_output_dim",
        type=parse_int_list,
        default="4,2,3,9",
        help=(
            "TT output modes. Product must equal depth_vqc*num_qubits*3."
        ),
    )

    parser.add_argument(
        "--tt_ranks",
        type=parse_int_list,
        default="1,2,2,2,1",
    )

    parser.add_argument(
        "--gaussian_mode",
        default="fixed_per_sample",
        choices=[
            "fixed_per_sample",
            "shared_fixed",
        ],
        help=(
            "fixed_per_sample assigns one fixed Gaussian z_i to each image; "
            "shared_fixed uses one fixed z for the entire dataset."
        ),
    )

    parser.add_argument(
        "--gaussian_seed",
        default=1234,
        type=int,
    )

    parser.add_argument(
        "--no_residual_global_angles",
        action="store_true",
        help=(
            "Disable residual trainable global angles and use only TT-generated "
            "angles."
        ),
    )

    parser.add_argument(
        "--no_fc",
        action="store_true",
        help=(
            "Do not use a classical output layer; use the first two measured "
            "expectations as logits."
        ),
    )

    parser.add_argument(
        "--ece_bins",
        default=15,
        type=int,
    )

    parser.add_argument(
        "--selection_metric",
        default="test_macro_f1",
        choices=[
            "test_accuracy",
            "test_macro_f1",
            "test_weighted_f1",
            "test_ece",
            "test_loss",
            "test_nll",
        ],
    )

    parser.add_argument(
        "--metrics_filename",
        default="tt_rmt_metrics.csv",
        type=str,
    )

    parser.add_argument(
        "--noise_models",
        default="none",
        type=str,
        help=(
            "Comma-separated: depol,dephase,pauli,overrot,twopauli,"
            "readout,none"
        ),
    )

    parser.add_argument(
        "--p_depol",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--p_dephase",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--pauli_px",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--pauli_py",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--pauli_pz",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--overrot_sigma",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--p_twopauli",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--p_readout",
        default=0.0,
        type=float,
    )

    args = parser.parse_args()
    validate_configuration(args)

    torch.manual_seed(args.gaussian_seed)
    np.random.seed(args.gaussian_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.gaussian_seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    gaussian_dim = math.prod(
        args.tt_input_dim
    )

    train_dataset, test_dataset = load_quantum_dot_data(
        data_root=args.data_root,
        image_dim=args.image_dim,
        gaussian_dim=gaussian_dim,
        gaussian_mode=args.gaussian_mode,
        gaussian_seed=args.gaussian_seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = TTTensorHyperVQC(
        tt_input_dims=args.tt_input_dim,
        tt_output_dims=args.tt_output_dim,
        tt_ranks=args.tt_ranks,
        n_wires=args.num_qubits,
        n_qlayers=args.depth_vqc,
        topology=args.model_kind,
        num_classes=args.num_classes,
        add_fc=not args.no_fc,
        use_residual_global_angles=(
            not args.no_residual_global_angles
        ),
    ).to(device)

    noise_cfg, noise_models = build_noise_cfg(args)
    model.vqc.noise_cfg = noise_cfg

    print("Architecture: RMT-compliant TT-based TensorHyper-VQC")
    print("TT input: Gaussian latent only")
    print(
        "Gaussian mode:",
        args.gaussian_mode,
    )
    print(
        "Gaussian dimension:",
        gaussian_dim,
    )
    print(
        "Image path: 2500-D image -> zero padding -> L2 normalization "
        "-> amplitude encoding"
    )
    print(
        "Amplitude state dimension:",
        2 ** args.num_qubits,
    )
    print(
        "Topology:",
        args.model_kind,
    )
    print(
        "Residual global angles:",
        not args.no_residual_global_angles,
    )
    print(
        "Noise models:",
        ",".join(noise_models) if noise_models else "none",
    )
    print(
        "Noise cfg:",
        noise_cfg,
    )
    print(
        "Total parameters:",
        sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
    )
    print(
        "Quantum microbatch size:",
        args.quantum_batch_size,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1,
    )

    os.makedirs(
        args.save_path,
        exist_ok=True,
    )

    metrics_path = os.path.join(
        args.save_path,
        args.metrics_filename,
    )

    checkpoint_path = os.path.join(
        args.save_path,
        f"best_tt_rmt_{args.model_kind}.pt",
    )

    csv_fields = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_nll",
        "train_accuracy",
        "train_macro_f1",
        "train_weighted_f1",
        "train_ece",
        "train_mean_confidence",
        "test_loss",
        "test_nll",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "test_ece",
        "test_mean_confidence",
    ]

    if args.selection_metric in {
        "test_loss",
        "test_nll",
        "test_ece",
    }:
        best_metric = float("inf")
    else:
        best_metric = -float("inf")

    with open(
        metrics_path,
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        for epoch in range(
            1,
            args.num_epochs + 1,
        ):
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                num_qubits=args.num_qubits,
                num_classes=args.num_classes,
                ece_bins=args.ece_bins,
                quantum_batch_size=args.quantum_batch_size,
                optimizer=optimizer,
            )

            test_metrics = run_epoch(
                model=model,
                loader=test_loader,
                criterion=criterion,
                device=device,
                num_qubits=args.num_qubits,
                num_classes=args.num_classes,
                ece_bins=args.ece_bins,
                quantum_batch_size=args.quantum_batch_size,
                optimizer=None,
            )

            current_learning_rate = optimizer.param_groups[0]["lr"]

            row = {
                "epoch": epoch,
                "learning_rate": current_learning_rate,
                "train_loss": train_metrics["loss"],
                "train_nll": train_metrics["nll"],
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "train_weighted_f1": train_metrics["weighted_f1"],
                "train_ece": train_metrics["ece"],
                "train_mean_confidence": train_metrics["mean_confidence"],
                "test_loss": test_metrics["loss"],
                "test_nll": test_metrics["nll"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_weighted_f1": test_metrics["weighted_f1"],
                "test_ece": test_metrics["ece"],
                "test_mean_confidence": test_metrics["mean_confidence"],
            }

            writer.writerow(row)
            csv_file.flush()

            selected_value = float(
                row[args.selection_metric]
            )

            if metric_is_better(
                selected_value,
                best_metric,
                args.selection_metric,
            ):
                best_metric = selected_value

                save_checkpoint(
                    path=checkpoint_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    train_metrics=train_metrics,
                    test_metrics=test_metrics,
                )

            scheduler.step()

            print(
                f"Epoch {epoch:02d} | "
                f"Train loss {train_metrics['loss']:.4f}, "
                f"acc {train_metrics['accuracy']:.4f}, "
                f"macro-F1 {train_metrics['macro_f1']:.4f}, "
                f"ECE {train_metrics['ece']:.4f} | "
                f"Test loss {test_metrics['loss']:.4f}, "
                f"acc {test_metrics['accuracy']:.4f}, "
                f"macro-F1 {test_metrics['macro_f1']:.4f}, "
                f"ECE {test_metrics['ece']:.4f}"
            )

    print(
        f"Metrics saved to: {metrics_path}"
    )
    print(
        f"Best checkpoint saved to: {checkpoint_path}"
    )
    print(
        f"Best {args.selection_metric}: {best_metric:.6f}"
    )


if __name__ == "__main__":
    main()
