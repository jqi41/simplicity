#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finite-m empirical verification of (Theorem 1 / Theorem 3 proxies):

We compare three circuit families on a *synthesized dataset* of size m:
  (1) naive VQC (deep, unstructured baseline)
  (2) TT-assisted encoding (TN-VQC): TT maps x -> phi(x) used for data encoding
  (3) TensorHyper-VQC: TT maps x -> circuit angles theta(x)

Goal:
  For each dataset size m, estimate the *input-variance* of a local observable output:
      Var_x [ f_theta(x) ]   where f_theta(x) = <psi_{theta,x} | O | psi_{theta,x} >
  averaged over random initializations (seeds).

Interpretation:
  - Haar-like typicality / concentration predicts Var_x[f_theta(x)] becomes very small
    for sufficiently deep unstructured VQCs (even at finite n).
  - Tensor-structured models should retain larger Var_x[f_theta(x)] ("anti-concentration"/non-collapse).

This script:
  - Uses torchquantum (for fast statevector simulation)
  - Uses a simple local observable: PauliZ on wire 0 (via MeasureAll and take index 0)
  - Sweeps dataset size m and reports mean±std over seeds for each model.

Requirements:
  pip install torch torchquantum numpy

Run:
  python verify_variance_vs_m.py --m_list 8,16,32,64,128 --num_qubits 12 --depth 6 --seeds 0,1,2,3,4
"""

import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_int_list(s: str) -> List[int]:
    s = s.strip()
    return [] if not s else [int(x.strip()) for x in s.split(",")]


def mean_std(xs: List[float]) -> Tuple[float, float]:
    arr = np.asarray(xs, dtype=np.float64)
    mu = float(arr.mean()) if arr.size else 0.0
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mu, sd


# -------------------------
# Tensor-Train layer
# -------------------------
TT_INIT_SCALE = 0.01


class TensorTrainLayer(nn.Module):
    """
    TT maps x in R^{prod(input_dims)} to y in R^{prod(output_dims)}.
    """
    def __init__(self, input_dims: List[int], output_dims: List[int], tt_ranks: List[int]):
        super().__init__()
        d = len(input_dims)
        assert len(output_dims) == d
        assert len(tt_ranks) == d + 1

        self.input_dims = list(input_dims)
        self.output_dims = list(output_dims)
        self.tt_ranks = list(tt_ranks)

        self.tt_cores = nn.ParameterList()
        for k in range(d):
            r0, r1 = tt_ranks[k], tt_ranks[k + 1]
            n_k, m_k = input_dims[k], output_dims[k]
            core = nn.Parameter(torch.randn(r0, n_k, m_k, r1) * TT_INIT_SCALE)
            nn.init.xavier_uniform_(core)
            self.tt_cores.append(core)

        self.bias = nn.Parameter(torch.zeros(int(math.prod(output_dims))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [bsz, prod(input_dims)]
        bsz = x.size(0)
        x_rs = x.view(bsz, *self.input_dims)

        batch = "b"
        letters = [chr(i) for i in range(ord("a"), ord("z") + 1) if chr(i) != batch]
        d = len(self.input_dims)
        iL = letters[:d]
        oL = letters[d:2 * d]
        rL = letters[2 * d:2 * d + d + 1]

        inp = batch + "".join(iL)
        cores = [f"{rL[k]}{iL[k]}{oL[k]}{rL[k + 1]}" for k in range(d)]
        outp = batch + "".join(oL)
        eins = inp + "," + ",".join(cores) + "->" + outp

        out = torch.einsum(eins, x_rs, *self.tt_cores)
        return out.reshape(bsz, -1) + self.bias


# -------------------------
# Simple encoder: RY on each wire using first n_wires features
# -------------------------
class SimpleRYEncoder(tq.QuantumModule):
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x_enc: torch.Tensor):
        # x_enc: [bsz, n_wires] -> use as angles
        for w in range(self.n_wires):
            tqf.ry(
                q_device,
                wires=w,
                params=x_enc[:, w],
                static=self.static_mode,
                parent_graph=self.graph,
            )


# -------------------------
# Base VQC returning local observable expectation (PauliZ on wire 0)
# -------------------------
class BaseVQC(tq.QuantumModule):
    """
    Output:
      f_theta(x) := <psi_{theta,x} | Z_0 | psi_{theta,x}>
    We implement measure-all-Z and take index 0.
    """
    def __init__(self, n_wires: int, n_qlayers: int, noise_prob: float = 0.0):
        super().__init__()
        self.n_wires = n_wires
        self.n_qlayers = n_qlayers
        self.noise_prob = float(noise_prob)

        self.angles = nn.Parameter(torch.randn(n_qlayers, n_wires, 3) * 0.1)
        self.encoder = SimpleRYEncoder(n_wires=n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def _reset(self, q_device: tq.QuantumDevice, bsz: int):
        q_device.reset_states(bsz)

    def _depolarize(self, q_device: tq.QuantumDevice):
        if self.noise_prob <= 0:
            return
        for i in range(self.n_wires):
            if torch.rand((), device=q_device.device) < self.noise_prob:
                err = torch.randint(0, 3, (), device=q_device.device).item()
                op = tqf.x if err == 0 else (tqf.y if err == 1 else tqf.z)
                op(q_device, wires=i, static=self.static_mode, parent_graph=self.graph)

    def _entangle_ring(self, q_device: tq.QuantumDevice):
        for i in range(self.n_wires - 1):
            tqf.cnot(q_device, wires=[i, i + 1], static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(q_device, wires=[self.n_wires - 1, 0], static=self.static_mode, parent_graph=self.graph)

    @tq.static_support
    def forward(
        self,
        x_enc: torch.Tensor,
        q_device: tq.QuantumDevice,
        angles_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x_enc: [bsz, n_wires] encoding angles for RY
        angles_batch: [bsz, L, n_wires, 3] if provided, else use global self.angles
        returns: f(x) = <Z_0> as [bsz]
        """
        bsz = x_enc.size(0)
        self._reset(q_device, bsz)
        self.encoder(q_device, x_enc)

        use_batch = angles_batch is not None
        for k in range(self.n_qlayers):
            for w in range(self.n_wires):
                if use_batch:
                    r = angles_batch[:, k, w, 0]
                    y = angles_batch[:, k, w, 1]
                    z = angles_batch[:, k, w, 2]
                else:
                    r, y, z = self.angles[k, w]
                tqf.rx(q_device, wires=w, params=r, static=self.static_mode, parent_graph=self.graph)
                tqf.ry(q_device, wires=w, params=y, static=self.static_mode, parent_graph=self.graph)
                tqf.rz(q_device, wires=w, params=z, static=self.static_mode, parent_graph=self.graph)

            self._entangle_ring(q_device)
            self._depolarize(q_device)

        z_all = self.measure(q_device)  # [bsz, n_wires]
        return z_all[:, 0]              # local observable on wire 0


# -------------------------
# Three model families
# -------------------------
class NaiveVQC(nn.Module):
    """
    naive: encoding is directly from x (first n_wires dims) and angles are global trainable params
    Here we are not training; we just sample random init and evaluate variance.
    """
    def __init__(self, n_wires: int, n_qlayers: int, noise_prob: float = 0.0):
        super().__init__()
        self.n_wires = n_wires
        self.vqc = BaseVQC(n_wires=n_wires, n_qlayers=n_qlayers, noise_prob=noise_prob)

    def forward(self, x_raw: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        # map raw features to bounded encoding angles
        x_enc = torch.tanh(x_raw[:, : self.n_wires]) * math.pi
        return self.vqc(x_enc, q_device, angles_batch=None)


class TNVQC(nn.Module):
    """
    TT-assisted encoding (TN-VQC):
      phi(x) = TT(x)[:n_wires], then used as encoder angles; VQC angles are global.
    """
    def __init__(
        self,
        input_dim: int,
        tt_input_dims: List[int],
        tt_output_dims: List[int],
        tt_ranks: List[int],
        n_wires: int,
        n_qlayers: int,
        noise_prob: float = 0.0,
    ):
        super().__init__()
        assert int(math.prod(tt_input_dims)) == input_dim, \
            "prod(tt_input_dims) must equal input_dim for this script."
        assert int(math.prod(tt_output_dims)) >= n_wires, \
            "prod(tt_output_dims) must be >= n_wires to produce phi(x)."
        self.n_wires = n_wires
        self.tt = TensorTrainLayer(tt_input_dims, tt_output_dims, tt_ranks)
        self.vqc = BaseVQC(n_wires=n_wires, n_qlayers=n_qlayers, noise_prob=noise_prob)

    def forward(self, x_raw: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        phi = self.tt(x_raw)[:, : self.n_wires]
        x_enc = torch.tanh(phi) * math.pi
        return self.vqc(x_enc, q_device, angles_batch=None)


class TensorHyperVQC(nn.Module):
    """
    TensorHyper-VQC:
      theta(x) = TT(x) reshaped into [L, n_wires, 3], and used as batch-wise angles.
      encoding is fixed from x itself (first n_wires dims).
    """
    def __init__(
        self,
        input_dim: int,
        tt_input_dims: List[int],
        tt_output_dims: List[int],
        tt_ranks: List[int],
        n_wires: int,
        n_qlayers: int,
        noise_prob: float = 0.0,
        residual_global: bool = True,
    ):
        super().__init__()
        assert int(math.prod(tt_input_dims)) == input_dim, \
            "prod(tt_input_dims) must equal input_dim for this script."
        target = n_qlayers * n_wires * 3
        assert int(math.prod(tt_output_dims)) == target, \
            "For TensorHyper-VQC, prod(tt_output_dims) must equal L*n_wires*3."
        self.n_wires = n_wires
        self.n_qlayers = n_qlayers
        self.residual_global = residual_global

        self.tt = TensorTrainLayer(tt_input_dims, tt_output_dims, tt_ranks)
        self.vqc = BaseVQC(n_wires=n_wires, n_qlayers=n_qlayers, noise_prob=noise_prob)

    def forward(self, x_raw: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        bsz = x_raw.size(0)
        x_enc = torch.tanh(x_raw[:, : self.n_wires]) * math.pi

        ang = self.tt(x_raw).reshape(bsz, self.n_qlayers, self.n_wires, 3)
        # keep angles bounded-ish for stability (optional but helps)
        ang = torch.tanh(ang) * math.pi

        if self.residual_global:
            ang = ang + self.vqc.angles.unsqueeze(0)

        return self.vqc(x_enc, q_device, angles_batch=ang)


# -------------------------
# Synthesized dataset
# -------------------------
def synthesize_x(m: int, input_dim: int, device: torch.device) -> torch.Tensor:
    """
    Synthetic inputs x ~ N(0,1), optionally could be structured.
    """
    return torch.randn(m, input_dim, device=device)


# -------------------------
# Metric: Var_x f_theta(x) at random init
# -------------------------
@torch.no_grad()
def estimate_var_over_x(
    model: nn.Module,
    x: torch.Tensor,
    n_wires: int,
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """
    Returns empirical variance over x of f(x) = <Z_0>.
    """
    model.eval()
    vals = []

    m = x.size(0)
    for i in range(0, m, batch_size):
        xb = x[i:i + batch_size]
        q_dev = tq.QuantumDevice(n_wires=n_wires, bsz=xb.size(0)).to(device)
        fb = model(xb, q_dev)  # [bsz]
        vals.append(fb.detach().cpu())

    f = torch.cat(vals, dim=0).numpy()
    return float(np.var(f, ddof=1)) if len(f) > 1 else 0.0


@dataclass
class OneRun:
    var_x: float
    n_params: int


def build_model(
    model_name: str,
    input_dim: int,
    n_wires: int,
    depth: int,
    noise_prob: float,
    tt_input_dims: List[int],
    tn_tt_output_dims: List[int],
    th_tt_output_dims: List[int],
    tt_ranks: List[int],
    residual_global: bool,
    device: torch.device,
) -> nn.Module:
    if model_name == "naive_vqc":
        return NaiveVQC(n_wires=n_wires, n_qlayers=depth, noise_prob=noise_prob).to(device)
    if model_name == "tn_vqc":
        return TNVQC(
            input_dim=input_dim,
            tt_input_dims=tt_input_dims,
            tt_output_dims=tn_tt_output_dims,
            tt_ranks=tt_ranks,
            n_wires=n_wires,
            n_qlayers=depth,
            noise_prob=noise_prob,
        ).to(device)
    if model_name == "tensorhyper_vqc":
        return TensorHyperVQC(
            input_dim=input_dim,
            tt_input_dims=tt_input_dims,
            tt_output_dims=th_tt_output_dims,
            tt_ranks=tt_ranks,
            n_wires=n_wires,
            n_qlayers=depth,
            noise_prob=noise_prob,
            residual_global=residual_global,
        ).to(device)
    raise ValueError(f"Unknown model_name: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="naive_vqc,tn_vqc,tensorhyper_vqc")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--m_list", type=str, default="8,16,32,64,128,256")

    parser.add_argument("--input_dim", type=int, default=2500,
                        help="Dim of synthetic x. For TT models, must equal prod(tt_input_dims).")
    parser.add_argument("--num_qubits", type=int, default=12)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--noise_prob", type=float, default=0.0)
    parser.add_argument("--batch_size_eval", type=int, default=64)

    # TT specs:
    parser.add_argument("--tt_input_dims", type=str, default="5,10,5,10",
                        help="Must satisfy prod == input_dim")
    # TN-VQC: prod >= num_qubits (phi dimension)
    parser.add_argument("--tn_tt_output_dims", type=str, default="16,16,1,1")
    # TensorHyper-VQC: prod == depth*num_qubits*3
    parser.add_argument("--th_tt_output_dims", type=str, default="4,2,3,9")
    parser.add_argument("--tt_ranks", type=str, default="1,2,5,2,1")
    parser.add_argument("--residual_global", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = [s.strip() for s in args.models.split(",") if s.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    m_list = [int(s.strip()) for s in args.m_list.split(",") if s.strip()]

    tt_input_dims = parse_int_list(args.tt_input_dims)
    tn_tt_output_dims = parse_int_list(args.tn_tt_output_dims)
    th_tt_output_dims = parse_int_list(args.th_tt_output_dims)
    tt_ranks = parse_int_list(args.tt_ranks)

    # Consistency checks (important for TT)
    if models and ("tn_vqc" in models or "tensorhyper_vqc" in models):
        if int(math.prod(tt_input_dims)) != args.input_dim:
            raise ValueError(
                f"Need prod(tt_input_dims)==input_dim, got {math.prod(tt_input_dims)} vs {args.input_dim}. "
                "Either change --input_dim or --tt_input_dims."
            )
    if "tensorhyper_vqc" in models:
        need = args.depth * args.num_qubits * 3
        got = int(math.prod(th_tt_output_dims))
        if got != need:
            raise ValueError(
                f"Need prod(th_tt_output_dims)==depth*num_qubits*3, got {got} vs {need}. "
                "Adjust --th_tt_output_dims."
            )

    # Storage: results[model][m] = list of var_x over seeds
    results: Dict[str, Dict[int, List[float]]] = {mn: {m: [] for m in m_list} for mn in models}
    nparams: Dict[str, int] = {}

    for seed in seeds:
        set_seed(seed)

        # For each m, use an independent synthetic dataset
        for m in m_list:
            x = synthesize_x(m=m, input_dim=args.input_dim, device=device)

            for mn in models:
                model = build_model(
                    model_name=mn,
                    input_dim=args.input_dim,
                    n_wires=args.num_qubits,
                    depth=args.depth,
                    noise_prob=args.noise_prob,
                    tt_input_dims=tt_input_dims,
                    tn_tt_output_dims=tn_tt_output_dims,
                    th_tt_output_dims=th_tt_output_dims,
                    tt_ranks=tt_ranks,
                    residual_global=args.residual_global,
                    device=device,
                )
                if mn not in nparams:
                    nparams[mn] = sum(p.numel() for p in model.parameters() if p.requires_grad)

                var_x = estimate_var_over_x(
                    model=model,
                    x=x,
                    n_wires=args.num_qubits,
                    device=device,
                    batch_size=args.batch_size_eval,
                )
                results[mn][m].append(var_x)

                # free memory
                del model
                torch.cuda.empty_cache()

    # -------------------------
    # Print paper-ready table (mean±std over seeds) vs m
    # -------------------------
    print("\n=== Var_x[f_theta(x)] vs dataset size m (mean±std over seeds) ===")
    header = ["m"] + [f"{mn}(#p={nparams[mn]})" for mn in models]
    print(" | ".join(header))
    print("-" * (len(" | ".join(header)) + 10))

    for m in m_list:
        row = [str(m)]
        for mn in models:
            mu, sd = mean_std(results[mn][m])
            row.append(f"{mu:.4e}±{sd:.2e}")
        print(" | ".join(row))

    # Also dump “exact numbers to paste” in a compact block per model
    print("\n=== Paste-ready blocks (per model) ===")
    for mn in models:
        print(f"\n[{mn}] (n={args.num_qubits}, depth={args.depth})")
        for m in m_list:
            mu, sd = mean_std(results[mn][m])
            print(f"  m={m:4d}: Var_x = {mu:.6e} ± {sd:.2e}")


if __name__ == "__main__":
    main()

