#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Synthetic-data simulation for random-matrix-induced simplicity bias in VQCs.

Models:
  1. unstructured_vqc
  2. tensor_operator_vqc: theta(x) = T(x; phi)
  3. tensor_hyper_vqc: theta = H(z; phi), z ~ N(0,I)

Metrics:
  - Var_x[f_theta(x)]
  - Pairwise output distinguishability
  - Initialization gradient variance
  - Empirical NTK effective rank

Example:
  python synthetic_rmt_vqc.py \
    --models unstructured_vqc,tensor_operator_vqc,tensor_hyper_vqc \
    --num_qubits 8 --depth 6 --input_dim 64 \
    --m_list 16,32,64,128 \
    --seeds 0,1,2,3,4
"""

import math
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def mean_std(xs: List[float]) -> Tuple[float, float]:
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def pairwise_distinguishability(f: torch.Tensor) -> float:
    f = f.flatten()
    if f.numel() <= 1:
        return 0.0
    diff = torch.abs(f[:, None] - f[None, :])
    return float(diff.mean().detach().cpu())


def effective_rank_from_kernel(K: torch.Tensor, eps: float = 1e-12) -> float:
    eigvals = torch.linalg.eigvalsh(K).clamp_min(0.0)
    s1 = eigvals.sum()
    s2 = (eigvals ** 2).sum()
    if s1.item() < eps or s2.item() < eps:
        return 0.0
    return float((s1 ** 2 / s2).detach().cpu())


# -------------------------
# Tensor-Train layer
# -------------------------
class TensorTrainLayer(nn.Module):
    """
    TT layer mapping x in R^{prod(input_dims)}
    to y in R^{prod(output_dims)}.
    """

    def __init__(self, input_dims: List[int], output_dims: List[int], tt_ranks: List[int]):
        super().__init__()

        if len(input_dims) != len(output_dims):
            raise ValueError("input_dims and output_dims must have the same length.")
        if len(tt_ranks) != len(input_dims) + 1:
            raise ValueError("tt_ranks must have length len(input_dims)+1.")
        if tt_ranks[0] != 1 or tt_ranks[-1] != 1:
            raise ValueError("Boundary TT ranks must be 1.")

        self.input_dims = list(input_dims)
        self.output_dims = list(output_dims)
        self.tt_ranks = list(tt_ranks)

        self.tt_cores = nn.ParameterList()
        for k in range(len(input_dims)):
            r0, r1 = tt_ranks[k], tt_ranks[k + 1]
            n_k, m_k = input_dims[k], output_dims[k]
            core = nn.Parameter(0.02 * torch.randn(r0, n_k, m_k, r1))
            self.tt_cores.append(core)

        self.bias = nn.Parameter(torch.zeros(int(math.prod(output_dims))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)

        expected_dim = int(math.prod(self.input_dims))
        if x.size(1) != expected_dim:
            raise ValueError(f"Input dimension mismatch: got {x.size(1)}, expected {expected_dim}.")

        x = x.view(bsz, *self.input_dims)
        d = len(self.input_dims)

        # Reserve "B" for batch and avoid reusing it in TT indices.
        labels = [c for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" if c != "B"]

        if len(labels) < 3 * d + 1:
            raise ValueError("Too many TT dimensions for einsum labels.")

        i_idx = labels[:d]
        o_idx = labels[d:2 * d]
        r_idx = labels[2 * d:2 * d + d + 1]

        x_sub = "B" + "".join(i_idx)
        core_subs = [
            f"{r_idx[k]}{i_idx[k]}{o_idx[k]}{r_idx[k + 1]}"
            for k in range(d)
        ]
        out_sub = "B" + "".join(o_idx)

        eq = x_sub + "," + ",".join(core_subs) + "->" + out_sub
        out = torch.einsum(eq, x, *self.tt_cores)
        return out.reshape(bsz, -1) + self.bias


# -------------------------
# Quantum circuit modules
# -------------------------
class SimpleRYEncoder(tq.QuantumModule):
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x_enc: torch.Tensor):
        for w in range(self.n_wires):
            tqf.ry(
                q_device,
                wires=w,
                params=x_enc[:, w],
                static=self.static_mode,
                parent_graph=self.graph,
            )


class BaseVQC(tq.QuantumModule):
    """
    Returns local observable <Z_0>.
    """

    def __init__(self, n_wires: int, depth: int):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.encoder = SimpleRYEncoder(n_wires)
        self.global_angles = nn.Parameter(0.1 * torch.randn(depth, n_wires, 3))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def _entangle_ring(self, q_device: tq.QuantumDevice):
        for i in range(self.n_wires - 1):
            tqf.cnot(
                q_device,
                wires=[i, i + 1],
                static=self.static_mode,
                parent_graph=self.graph,
            )
        tqf.cnot(
            q_device,
            wires=[self.n_wires - 1, 0],
            static=self.static_mode,
            parent_graph=self.graph,
        )

    @tq.static_support
    def forward(
        self,
        x_enc: torch.Tensor,
        q_device: tq.QuantumDevice,
        angles_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x_enc: [batch, n_wires]
        angles_batch:
          None or [batch, depth, n_wires, 3]
        """
        bsz = x_enc.size(0)
        q_device.reset_states(bsz)
        self.encoder(q_device, x_enc)

        use_batch = angles_batch is not None

        for layer in range(self.depth):
            for wire in range(self.n_wires):
                if use_batch:
                    rx = angles_batch[:, layer, wire, 0]
                    ry = angles_batch[:, layer, wire, 1]
                    rz = angles_batch[:, layer, wire, 2]
                else:
                    rx = self.global_angles[layer, wire, 0]
                    ry = self.global_angles[layer, wire, 1]
                    rz = self.global_angles[layer, wire, 2]

                tqf.rx(q_device, wires=wire, params=rx, static=self.static_mode, parent_graph=self.graph)
                tqf.ry(q_device, wires=wire, params=ry, static=self.static_mode, parent_graph=self.graph)
                tqf.rz(q_device, wires=wire, params=rz, static=self.static_mode, parent_graph=self.graph)

            self._entangle_ring(q_device)

        z_all = self.measure(q_device)
        return z_all[:, 0]


# -------------------------
# Model families
# -------------------------
class UnstructuredVQC(nn.Module):
    def __init__(self, n_wires: int, depth: int):
        super().__init__()
        self.n_wires = n_wires
        self.vqc = BaseVQC(n_wires, depth)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        x_enc = torch.tanh(x[:, : self.n_wires]) * math.pi
        return self.vqc(x_enc, q_device, angles_batch=None)


class TensorOperatorVQC(nn.Module):
    """
    Input-conditioned tensor operator:
        theta(x) = T(x; phi)
    """

    def __init__(
        self,
        input_dims: List[int],
        output_dims: List[int],
        tt_ranks: List[int],
        n_wires: int,
        depth: int,
        residual_global: bool = False,
    ):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.residual_global = residual_global

        target_dim = depth * n_wires * 3
        if int(math.prod(output_dims)) != target_dim:
            raise ValueError(f"TensorOperator output dim must be {target_dim}.")

        self.tt = TensorTrainLayer(input_dims, output_dims, tt_ranks)
        self.vqc = BaseVQC(n_wires, depth)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        bsz = x.size(0)
        x_enc = torch.tanh(x[:, : self.n_wires]) * math.pi

        theta_x = self.tt(x).reshape(bsz, self.depth, self.n_wires, 3)
        theta_x = torch.tanh(theta_x) * math.pi

        if self.residual_global:
            theta_x = theta_x + self.vqc.global_angles.unsqueeze(0)

        return self.vqc(x_enc, q_device, angles_batch=theta_x)


class TensorHyperVQC(nn.Module):
    """
    Noise-conditioned tensor hypernetwork:
        theta = H(z; phi), z ~ N(0,I)
    """

    def __init__(
        self,
        noise_dim: int,
        input_dims: List[int],
        output_dims: List[int],
        tt_ranks: List[int],
        n_wires: int,
        depth: int,
        residual_global: bool = False,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.n_wires = n_wires
        self.depth = depth
        self.residual_global = residual_global

        if int(math.prod(input_dims)) != noise_dim:
            raise ValueError("prod(noise input dims) must equal noise_dim.")

        target_dim = depth * n_wires * 3
        if int(math.prod(output_dims)) != target_dim:
            raise ValueError(f"TensorHyper output dim must be {target_dim}.")

        self.tt = TensorTrainLayer(input_dims, output_dims, tt_ranks)
        self.vqc = BaseVQC(n_wires, depth)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        bsz = x.size(0)
        x_enc = torch.tanh(x[:, : self.n_wires]) * math.pi

        z = torch.randn(bsz, self.noise_dim, device=x.device)
        theta = self.tt(z).reshape(bsz, self.depth, self.n_wires, 3)
        theta = torch.tanh(theta) * math.pi

        if self.residual_global:
            theta = theta + self.vqc.global_angles.unsqueeze(0)

        return self.vqc(x_enc, q_device, angles_batch=theta)


# -------------------------
# Synthetic data
# -------------------------
def synthesize_x(
    m: int,
    input_dim: int,
    device: torch.device,
    kind: str = "structured",
) -> torch.Tensor:
    if kind == "gaussian":
        return torch.randn(m, input_dim, device=device)

    if kind == "structured":
        t = torch.linspace(-1.0, 1.0, m, device=device).unsqueeze(1)
        freqs = torch.arange(1, input_dim + 1, device=device).float().unsqueeze(0)
        x = torch.sin(math.pi * t * freqs / input_dim)
        x += 0.3 * torch.cos(2.0 * math.pi * t * freqs / input_dim)
        x += 0.05 * torch.randn_like(x)
        return x

    if kind == "mixed":
        x1 = synthesize_x(m, input_dim, device, "structured")
        x2 = torch.randn(m, input_dim, device=device)
        return 0.7 * x1 + 0.3 * x2

    raise ValueError(f"Unknown data kind: {kind}")


# -------------------------
# Metrics
# -------------------------
def evaluate_outputs(
    model: nn.Module,
    x: torch.Tensor,
    n_wires: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    q_dev = tq.QuantumDevice(n_wires=n_wires, bsz=x.size(0)).to(device)
    with torch.no_grad():
        return model(x, q_dev)


def gradient_variance(
    model: nn.Module,
    x: torch.Tensor,
    n_wires: int,
    device: torch.device,
) -> float:
    model.train()
    q_dev = tq.QuantumDevice(n_wires=n_wires, bsz=x.size(0)).to(device)
    f = model(x, q_dev).mean()

    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(
        f,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    flat = [g.detach().reshape(-1) for g in grads if g is not None]
    if not flat:
        return 0.0

    gvec = torch.cat(flat)
    if gvec.numel() <= 1:
        return 0.0
    return float(torch.var(gvec, unbiased=True).detach().cpu())


def empirical_ntk_effective_rank(
    model: nn.Module,
    x: torch.Tensor,
    n_wires: int,
    device: torch.device,
    max_points: int = 32,
) -> float:
    model.train()
    x = x[:max_points]
    params = [p for p in model.parameters() if p.requires_grad]

    grads = []
    for i in range(x.size(0)):
        q_dev = tq.QuantumDevice(n_wires=n_wires, bsz=1).to(device)
        f_i = model(x[i : i + 1], q_dev).sum()

        g_i = torch.autograd.grad(
            f_i,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        flat = [g.detach().reshape(-1) for g in g_i if g is not None]
        if not flat:
            continue
        grads.append(torch.cat(flat))

    if len(grads) <= 1:
        return 0.0

    G = torch.stack(grads, dim=0)
    K = G @ G.T
    return effective_rank_from_kernel(K)


# -------------------------
# Model builder
# -------------------------
def build_model(args, model_name: str, device: torch.device) -> nn.Module:
    if model_name == "unstructured_vqc":
        return UnstructuredVQC(args.num_qubits, args.depth).to(device)

    if model_name == "tensor_operator_vqc":
        return TensorOperatorVQC(
            input_dims=parse_int_list(args.tt_input_dims),
            output_dims=parse_int_list(args.theta_output_dims),
            tt_ranks=parse_int_list(args.tt_ranks),
            n_wires=args.num_qubits,
            depth=args.depth,
            residual_global=args.residual_global,
        ).to(device)

    if model_name == "tensor_hyper_vqc":
        return TensorHyperVQC(
            noise_dim=args.noise_dim,
            input_dims=parse_int_list(args.noise_input_dims),
            output_dims=parse_int_list(args.theta_output_dims),
            tt_ranks=parse_int_list(args.noise_tt_ranks),
            n_wires=args.num_qubits,
            depth=args.depth,
            residual_global=args.residual_global,
        ).to(device)

    raise ValueError(f"Unknown model: {model_name}")


def validate_args(args):
    if int(math.prod(parse_int_list(args.tt_input_dims))) != args.input_dim:
        raise ValueError(
            f"prod(tt_input_dims) must equal input_dim. "
            f"Got {math.prod(parse_int_list(args.tt_input_dims))} vs {args.input_dim}."
        )

    if int(math.prod(parse_int_list(args.noise_input_dims))) != args.noise_dim:
        raise ValueError(
            f"prod(noise_input_dims) must equal noise_dim. "
            f"Got {math.prod(parse_int_list(args.noise_input_dims))} vs {args.noise_dim}."
        )

    target_dim = args.depth * args.num_qubits * 3
    if int(math.prod(parse_int_list(args.theta_output_dims))) != target_dim:
        raise ValueError(
            f"prod(theta_output_dims) must equal depth*num_qubits*3. "
            f"Got {math.prod(parse_int_list(args.theta_output_dims))} vs {target_dim}."
        )

    if len(parse_int_list(args.tt_ranks)) != len(parse_int_list(args.tt_input_dims)) + 1:
        raise ValueError("tt_ranks length must equal len(tt_input_dims)+1.")

    if len(parse_int_list(args.noise_tt_ranks)) != len(parse_int_list(args.noise_input_dims)) + 1:
        raise ValueError("noise_tt_ranks length must equal len(noise_input_dims)+1.")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        type=str,
        default="unstructured_vqc,tensor_operator_vqc,tensor_hyper_vqc",
    )
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--m_list", type=str, default="16,32,64,128")

    parser.add_argument("--input_dim", type=int, default=64)
    parser.add_argument("--num_qubits", type=int, default=20)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument(
        "--data_kind",
        type=str,
        default="structured",
        choices=["structured", "gaussian", "mixed"],
    )

    # TensorOperator: prod(tt_input_dims) must equal input_dim.
    parser.add_argument("--tt_input_dims", type=str, default="4,4,4")

    # TensorHyper: prod(noise_input_dims) must equal noise_dim.
    parser.add_argument("--noise_dim", type=int, default=16)
    parser.add_argument("--noise_input_dims", type=str, default="4,4,1")

    # Output dimension must equal depth * num_qubits * 3.
    # Default: 6*8*3 = 144 = 4*6*6.
    parser.add_argument("--theta_output_dims", type=str, default="10,6,6")

    parser.add_argument("--tt_ranks", type=str, default="1,4,4,1")
    parser.add_argument("--noise_tt_ranks", type=str, default="1,4,4,1")

    parser.add_argument("--ntk_points", type=int, default=32)
    parser.add_argument("--residual_global", action="store_true")
    parser.add_argument("--save_csv", type=str, default="synthetic_rmt_vqc_results.csv")

    args = parser.parse_args()
    validate_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    m_list = [int(m.strip()) for m in args.m_list.split(",") if m.strip()]

    results: Dict[str, Dict[int, Dict[str, List[float]]]] = {
        model: {
            m: {
                "var_x": [],
                "pairwise": [],
                "grad_var": [],
                "ntk_reff": [],
            }
            for m in m_list
        }
        for model in models
    }

    rows = []

    for seed in seeds:
        set_seed(seed)

        for m in m_list:
            x = synthesize_x(m, args.input_dim, device, args.data_kind)

            for model_name in models:
                model = build_model(args, model_name, device)

                f = evaluate_outputs(model, x, args.num_qubits, device)
                var_x = float(torch.var(f, unbiased=True).detach().cpu()) if f.numel() > 1 else 0.0
                pairwise = pairwise_distinguishability(f)

                x_small = x[: min(m, args.ntk_points)]
                grad_var = gradient_variance(model, x_small, args.num_qubits, device)
                ntk_reff = empirical_ntk_effective_rank(
                    model,
                    x,
                    args.num_qubits,
                    device,
                    max_points=min(m, args.ntk_points),
                )

                results[model_name][m]["var_x"].append(var_x)
                results[model_name][m]["pairwise"].append(pairwise)
                results[model_name][m]["grad_var"].append(grad_var)
                results[model_name][m]["ntk_reff"].append(ntk_reff)

                rows.append(
                    {
                        "seed": seed,
                        "m": m,
                        "model": model_name,
                        "var_x": var_x,
                        "pairwise": pairwise,
                        "grad_var": grad_var,
                        "ntk_reff": ntk_reff,
                    }
                )

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print("\n=== Synthetic RMT-VQC statistics ===")
    for metric in ["var_x", "pairwise", "grad_var", "ntk_reff"]:
        print(f"\n--- {metric} ---")
        header = ["m"] + models
        print(" | ".join(header))
        print("-" * 100)

        for m in m_list:
            row = [str(m)]
            for model_name in models:
                mu, sd = mean_std(results[model_name][m][metric])
                row.append(f"{mu:.4e} ± {sd:.2e}")
            print(" | ".join(row))

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_csv(args.save_csv, index=False)
        print(f"\nSaved raw results to: {args.save_csv}")
    except Exception as e:
        print(f"\nCould not save CSV because pandas failed: {e}")


if __name__ == "__main__":
    main()
