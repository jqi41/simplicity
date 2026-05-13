#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quantum-dot classification for random-matrix-induced simplicity bias.

Models:
  1. unstructured
  2. tensor_operator: theta(x) = T(x; phi)
  3. tensor_hyper: theta = H(z; phi), z ~ N(0,I)

Key update:
  The VQC outputs are fed directly to softmax using the first two quantum
  observable channels. No classical fully connected readout layer is used.

Default sweep:
  qubits = 8, 12, 16, 20
  depths = 4, 6, 8
  ranks  = 2, 4, 6
"""

from __future__ import annotations

import os
import csv
import math
import argparse
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Utility
# ============================================================

def parse_int_list(x: str) -> List[int]:
    x = str(x).strip()
    if x.startswith("[") and x.endswith("]"):
        x = x[1:-1]
    if not x:
        return []
    return [int(t.strip()) for t in x.split(",") if t.strip()]


def parse_str_list(x: str) -> List[str]:
    return [t.strip() for t in str(x).split(",") if t.strip()]


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def factorize_to_n_dims(n: int, num_dims: int = 4) -> Tuple[int, ...]:
    dims = [1] * num_dims
    remaining = int(n)
    p = 2

    while p * p <= remaining:
        while remaining % p == 0:
            idx = min(range(num_dims), key=lambda i: dims[i])
            dims[idx] *= p
            remaining //= p
        p += 1

    if remaining > 1:
        idx = min(range(num_dims), key=lambda i: dims[i])
        dims[idx] *= remaining

    return tuple(sorted(dims))


def build_noise_cfg(args) -> Dict[str, float]:
    models = parse_str_list(args.noise_models)
    if "none" in models:
        models = []

    return {
        "depol": args.p_depol if "depol" in models else 0.0,
        "dephase": args.p_dephase if "dephase" in models else 0.0,
        "p_readout": args.p_readout if "readout" in models else 0.0,
        "overrot_sigma": args.overrot_sigma if "overrot" in models else 0.0,
    }


# ============================================================
# Data
# ============================================================

def load_quantum_dot_data(
    dataset_root: str,
    batch_size: int,
    test_kind: str = "gen",
):
    x_clean = np.load(f"{dataset_root}/csds_noiseless.npy")
    x_noisy = np.load(f"{dataset_root}/csds.npy")
    y = np.load(f"{dataset_root}/labels.npy").astype(np.int64)

    x_clean = x_clean.reshape(-1, 2500).astype(np.float32)
    x_noisy = x_noisy.reshape(-1, 2500).astype(np.float32)

    x_mean = x_noisy.mean()
    x_std = x_noisy.std() + 1e-8

    x_clean = (x_clean - x_mean) / x_std
    x_noisy = (x_noisy - x_mean) / x_std

    split = int(0.9 * len(y))

    x_train, y_train = x_noisy[:split], y[:split]

    if test_kind == "rep":
        x_test, y_test = x_clean[split:], y[split:]
    else:
        x_test, y_test = x_noisy[split:], y[split:]

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_ds, test_ds, train_loader, test_loader


# ============================================================
# Tensor-Train Linear Layer
# ============================================================

class TTLinear(nn.Module):
    def __init__(self, in_dims, out_dims, tt_rank: int, bias: bool = True):
        super().__init__()

        if len(in_dims) != len(out_dims):
            raise ValueError(
                f"in_dims and out_dims must have the same length. "
                f"Got {len(in_dims)} and {len(out_dims)}."
            )

        self.in_dims = list(in_dims)
        self.out_dims = list(out_dims)
        self.input_dim = int(np.prod(in_dims))
        self.output_dim = int(np.prod(out_dims))

        ranks = [1] + [tt_rank] * (len(in_dims) - 1) + [1]

        self.cores = nn.ParameterList()
        for k in range(len(in_dims)):
            core = nn.Parameter(
                torch.empty(ranks[k], in_dims[k], out_dims[k], ranks[k + 1])
            )
            nn.init.xavier_uniform_(core)
            self.cores.append(core)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.output_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        bsz = x.size(0)

        if x.size(1) != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch: got {x.size(1)}, expected {self.input_dim}."
            )

        x = x.view(bsz, *self.in_dims)

        d = len(self.in_dims)
        labels = [c for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" if c != "B"]

        if len(labels) < 3 * d + 1:
            raise ValueError("Too many TT dimensions for einsum labels.")

        iL = labels[:d]
        oL = labels[d:2 * d]
        rL = labels[2 * d:2 * d + d + 1]

        inp = "B" + "".join(iL)
        cores = [
            f"{rL[k]}{iL[k]}{oL[k]}{rL[k + 1]}"
            for k in range(d)
        ]
        outp = "B" + "".join(oL)

        eins = inp + "," + ",".join(cores) + "->" + outp
        out = torch.einsum(eins, x, *self.cores)
        out = out.reshape(bsz, -1)

        if self.bias is not None:
            out = out + self.bias

        return out


# ============================================================
# Encoders and Tensor Parameter Generators
# ============================================================

class DenseEncoder(nn.Module):
    def __init__(self, input_dim=2500, hidden_dim=256, n_qubits=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_qubits),
        )

    def forward(self, x):
        return math.pi * torch.tanh(self.net(x))


class TensorOperatorGenerator(nn.Module):
    """
    Input-conditioned tensor operator:
        theta(x) = T(x; phi)
    """

    def __init__(self, input_dim=2500, hidden_dim=256, param_dim=216, rank=4):
        super().__init__()
        self.feature = TTLinear(
            factorize_to_n_dims(input_dim),
            factorize_to_n_dims(hidden_dim),
            rank,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.operator = TTLinear(
            factorize_to_n_dims(hidden_dim),
            factorize_to_n_dims(param_dim),
            rank,
        )

    def forward(self, x):
        h = F.gelu(self.norm(self.feature(x)))
        theta = self.operator(h)
        return theta, h


class TensorHyperGenerator(nn.Module):
    """
    Noise-conditioned tensor hypernetwork:
        theta = H(z; phi), z ~ N(0,I)
    """

    def __init__(self, noise_dim=64, hidden_dim=128, param_dim=216, rank=4):
        super().__init__()
        self.noise_dim = noise_dim
        self.feature = TTLinear(
            factorize_to_n_dims(noise_dim),
            factorize_to_n_dims(hidden_dim),
            rank,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.hyper = TTLinear(
            factorize_to_n_dims(hidden_dim),
            factorize_to_n_dims(param_dim),
            rank,
        )

    def forward(self, batch_size: int, device: torch.device):
        z = torch.randn(batch_size, self.noise_dim, device=device)
        h = F.gelu(self.norm(self.feature(z)))
        theta = self.hyper(h)
        return theta, h


# ============================================================
# VQC Surrogate
# ============================================================

class NoiseAwareVQC(nn.Module):
    """
    Differentiable VQC surrogate.

    This is not a full statevector simulator. It is designed for scalable
    empirical testing of concentration, distinguishability, and trainability
    trends on quantum-dot data.
    """

    def __init__(self, n_qubits=12, layers=6, noise_cfg=None):
        super().__init__()
        self.n_qubits = n_qubits
        self.layers = layers
        self.noise_cfg = {} if noise_cfg is None else noise_cfg

        self.theta = nn.Parameter(0.05 * torch.randn(1, layers, 3, n_qubits))

    def forward(self, angles, theta_override: Optional[torch.Tensor] = None):
        bsz = angles.size(0)

        if theta_override is None:
            theta = self.theta.expand(bsz, -1, -1, -1)
        else:
            theta = theta_override.view(bsz, self.layers, 3, self.n_qubits)

        overrot = float(self.noise_cfg.get("overrot_sigma", 0.0))
        if self.training and overrot > 0:
            angles = angles + overrot * torch.randn_like(angles)
            theta = theta + overrot * torch.randn_like(theta)

        rx = theta[:, :, 0, :].mean(dim=1)
        ry = theta[:, :, 1, :].mean(dim=1)
        rz = theta[:, :, 2, :].mean(dim=1)

        qfeat = torch.sin(angles + rx)
        qfeat = qfeat + 0.5 * torch.cos(ry)
        qfeat = qfeat + 0.5 * torch.sin(rz)

        for _ in range(self.layers):
            qfeat = 0.75 * qfeat + 0.25 * torch.roll(qfeat, shifts=1, dims=1)
            qfeat = torch.tanh(qfeat)

        depol = float(self.noise_cfg.get("depol", 0.0))
        dephase = float(self.noise_cfg.get("dephase", 0.0))
        readout = float(self.noise_cfg.get("p_readout", 0.0))

        qfeat = (1.0 - depol) * qfeat
        qfeat = (1.0 - dephase) * qfeat
        qfeat = (1.0 - 2.0 * readout) * qfeat

        return torch.clamp(qfeat, -1.0, 1.0)


# ============================================================
# Direct Quantum Readout
# ============================================================

class DirectQuantumReadout(nn.Module):
    """
    Directly maps the first two quantum observable channels to logits.

        logits = scale * [qfeat[:, 0], qfeat[:, 1]]

    This removes the classical fully connected readout layer.
    """

    def __init__(self, num_classes=2, learnable_scale=True, init_scale=3.0):
        super().__init__()

        if num_classes != 2:
            raise ValueError("DirectQuantumReadout supports binary classification only.")

        if learnable_scale:
            self.logit_scale = nn.Parameter(torch.tensor(float(init_scale)))
        else:
            self.register_buffer("logit_scale", torch.tensor(float(init_scale)))

    def forward(self, qfeat):
        if qfeat.size(1) < 2:
            raise ValueError("Need at least two quantum channels for binary direct readout.")
        return self.logit_scale * qfeat[:, :2]


# ============================================================
# Model Families
# ============================================================

class VQCClassifier(nn.Module):
    def __init__(
        self,
        model_type: str,
        input_dim=2500,
        num_classes=2,
        n_qubits=12,
        depth=6,
        rank=4,
        hidden_dim=256,
        noise_dim=64,
        residual_global=False,
        noise_cfg=None,
        learnable_logit_scale=True,
    ):
        super().__init__()

        if n_qubits < 2:
            raise ValueError("Direct quantum readout requires n_qubits >= 2.")

        self.model_type = model_type
        self.n_qubits = n_qubits
        self.depth = depth
        self.rank = rank
        self.param_dim = 3 * n_qubits * depth
        self.residual_global = residual_global

        self.encoder = DenseEncoder(input_dim, hidden_dim, n_qubits)

        if model_type == "unstructured":
            self.param_generator = None

        elif model_type == "tensor_operator":
            self.param_generator = TensorOperatorGenerator(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                param_dim=self.param_dim,
                rank=rank,
            )

        elif model_type == "tensor_hyper":
            self.param_generator = TensorHyperGenerator(
                noise_dim=noise_dim,
                hidden_dim=hidden_dim,
                param_dim=self.param_dim,
                rank=rank,
            )

        else:
            raise ValueError(
                f"Unknown model_type={model_type}. "
                "Use unstructured, tensor_operator, or tensor_hyper."
            )

        self.vqc = NoiseAwareVQC(n_qubits, depth, noise_cfg)

        # No classical FC layer: direct quantum channels -> logits.
        self.head = DirectQuantumReadout(
            num_classes=num_classes,
            learnable_scale=learnable_logit_scale,
            init_scale=3.0,
        )

    def forward(self, x):
        angles = self.encoder(x)
        aux = {"angles": angles}

        theta_override = None

        if self.model_type == "tensor_operator":
            theta_override, h = self.param_generator(x)
            aux["theta_generated"] = theta_override
            aux["h"] = h

        elif self.model_type == "tensor_hyper":
            theta_override, h = self.param_generator(
                batch_size=x.size(0),
                device=x.device,
            )
            aux["theta_generated"] = theta_override
            aux["h"] = h

        if theta_override is not None and self.residual_global:
            base = self.vqc.theta.expand(x.size(0), -1, -1, -1).reshape(x.size(0), -1)
            theta_override = theta_override + base

        qfeat = self.vqc(angles, theta_override)

        # Directly use qfeat[:, 0] and qfeat[:, 1] as the two logits.
        logits = self.head(qfeat)

        aux["qfeat"] = qfeat
        aux["observable"] = logits[:, 1] - logits[:, 0]

        return logits, aux


# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)

    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def output_variance_and_pairwise(model, loader, device, max_batches=4):
    model.eval()

    outputs = []
    for b, (x, _) in enumerate(loader):
        if b >= max_batches:
            break

        x = x.to(device)
        logits, aux = model(x)
        f = aux["observable"]
        outputs.append(f.detach().cpu())

    f_all = torch.cat(outputs, dim=0)

    output_var = f_all.var(unbiased=False).item()

    if len(f_all) > 1:
        diffs = torch.cdist(f_all.view(-1, 1), f_all.view(-1, 1), p=1)
        pairwise = diffs.mean().item()
    else:
        pairwise = 0.0

    return output_var, pairwise


def gradient_variance(model, loader, device, max_batches=2):
    model.train()

    grads = []

    for b, (x, y) in enumerate(loader):
        if b >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)

        model.zero_grad()
        loss.backward()

        g_list = []
        for name, p in model.named_parameters():
            if p.grad is not None and (
                "vqc" in name or "encoder" in name or "param_generator" in name
            ):
                g_list.append(p.grad.detach().flatten())

        if g_list:
            grads.append(torch.cat(g_list).cpu())

    if not grads:
        return 0.0

    g = torch.cat(grads)
    return g.var(unbiased=False).item()


def initialization_statistics(
    model_ctor,
    loader,
    device,
    num_inits=5,
):
    output_vars = []
    pairwise_vals = []
    grad_vars = []

    for seed in range(num_inits):
        set_seed(1000 + seed)
        model = model_ctor().to(device)

        out_var, pairwise = output_variance_and_pairwise(model, loader, device)
        grad_var = gradient_variance(model, loader, device)

        output_vars.append(out_var)
        pairwise_vals.append(pairwise)
        grad_vars.append(grad_var)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "init_output_var_mean": float(np.mean(output_vars)),
        "init_output_var_std": float(np.std(output_vars)),
        "init_pairwise_mean": float(np.mean(pairwise_vals)),
        "init_pairwise_std": float(np.std(pairwise_vals)),
        "init_grad_var_mean": float(np.mean(grad_vars)),
        "init_grad_var_std": float(np.std(grad_vars)),
    }


# ============================================================
# Training
# ============================================================

def train_model(model, train_loader, test_loader, device, epochs=20, lr=3e-3):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_acc = 0.0
    best_loss = float("inf")

    for _ in range(1, epochs + 1):
        model.train()

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits, aux = model(x)
            loss = F.cross_entropy(logits, y)

            if "theta_generated" in aux:
                loss = loss + 1e-5 * aux["theta_generated"].pow(2).mean()
            if "qfeat" in aux:
                loss = loss + 1e-5 * aux["qfeat"].pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        scheduler.step()

        test_loss, test_acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, test_acc)
        best_loss = min(best_loss, test_loss)

    final_loss, final_acc = evaluate(model, test_loader, device)

    return {
        "best_test_acc": best_acc,
        "best_test_loss": best_loss,
        "final_test_acc": final_acc,
        "final_test_loss": final_loss,
    }


# ============================================================
# Single Experiment
# ============================================================

def run_single_experiment(
    args,
    model_type: str,
    n_qubits: int,
    depth: int,
    rank: int,
    train_loader,
    test_loader,
    device,
):
    noise_cfg = build_noise_cfg(args)

    def model_ctor():
        return VQCClassifier(
            model_type=model_type,
            input_dim=2500,
            num_classes=2,
            n_qubits=n_qubits,
            depth=depth,
            rank=rank,
            hidden_dim=args.hidden_dim,
            noise_dim=args.noise_dim,
            residual_global=args.residual_global,
            noise_cfg=noise_cfg,
            learnable_logit_scale=not args.fixed_logit_scale,
        )

    print("\n====================================================")
    print(f"Model={model_type} | qubits={n_qubits} | depth={depth} | rank={rank}")
    print("====================================================")

    init_stats = initialization_statistics(
        model_ctor=model_ctor,
        loader=test_loader,
        device=device,
        num_inits=args.num_inits,
    )

    set_seed(args.seed)
    model = model_ctor().to(device)

    train_stats = train_model(
        model,
        train_loader,
        test_loader,
        device,
        epochs=args.num_epochs,
        lr=args.lr,
    )

    trained_out_var, trained_pairwise = output_variance_and_pairwise(
        model,
        test_loader,
        device,
    )
    trained_grad_var = gradient_variance(model, test_loader, device)

    result = {
        "model": model_type,
        "n_qubits": n_qubits,
        "depth": depth,
        "rank": rank,
        "trainable_params": count_trainable_params(model),
        **init_stats,
        **train_stats,
        "trained_output_var": trained_out_var,
        "trained_pairwise": trained_pairwise,
        "trained_grad_var": trained_grad_var,
    }

    print(result)
    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", type=str, default="./mlqe_2023_edx/week1/dataset")
    parser.add_argument("--results_csv", type=str, default="quantum_dot_rmt_3241_readout_q8_20.csv")

    parser.add_argument("--models", type=str, default="unstructured,tensor_operator,tensor_hyper")
    parser.add_argument("--qubits_list", type=str, default="8,12,16,20")
    parser.add_argument("--depths_list", type=str, default="4,6,8")
    parser.add_argument("--ranks_list", type=str, default="2,4,6")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_inits", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--noise_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--test_kind", type=str, default="gen", choices=["gen", "rep"])

    parser.add_argument("--seed", type=int, default=3241)
    parser.add_argument("--residual_global", action="store_true")
    parser.add_argument("--fixed_logit_scale", action="store_true")

    parser.add_argument("--noise_models", type=str, default="depol,dephase,readout")
    parser.add_argument("--p_depol", type=float, default=0.01)
    parser.add_argument("--p_dephase", type=float, default=0.01)
    parser.add_argument("--p_readout", type=float, default=0.05)
    parser.add_argument("--overrot_sigma", type=float, default=0.002)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, train_loader, test_loader = load_quantum_dot_data(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        test_kind=args.test_kind,
    )

    models = parse_str_list(args.models)
    qubits_list = parse_int_list(args.qubits_list)
    depths_list = parse_int_list(args.depths_list)
    ranks_list = parse_int_list(args.ranks_list)

    rows = []

    for model_type in models:
        for n_qubits in qubits_list:
            for depth in depths_list:

                if model_type == "unstructured":
                    rank_values = [0]
                else:
                    rank_values = ranks_list

                for rank in rank_values:
                    row = run_single_experiment(
                        args=args,
                        model_type=model_type,
                        n_qubits=n_qubits,
                        depth=depth,
                        rank=rank,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        device=device,
                    )
                    rows.append(row)

                    os.makedirs(os.path.dirname(args.results_csv) or ".", exist_ok=True)
                    with open(args.results_csv, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                        writer.writeheader()
                        writer.writerows(rows)

    print(f"\nSaved results to: {args.results_csv}")


if __name__ == "__main__":
    main()
