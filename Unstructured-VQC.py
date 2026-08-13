#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unstructured VQC baseline for Quantum-Dot Classification
(with composable quantum noise and RMT-oriented evaluation metrics)

Purpose:
  This script provides the unstructured variational-quantum-circuit baseline
  used to corroborate comparisons against TT-based and TR-based TensorHyper-VQC
  models in the RMT paper.

Controlled comparison:
  – Same quantum-dot dataset and 90/10 split
  – Same ring, MPS, and tree entanglement choices
  – Same rotation blocks: RX-RY-RZ on every qubit in every layer
  – Same composable noise models
  – Same optimizer and learning-rate scheduler
  – Same train/test metrics and CSV logging
  – Same best-checkpoint selection criteria
  – Memory-safe quantum microbatching for large statevectors

Unlike TensorHyper-VQC:
  – No Tensor Train or Tensor Ring parameter generator is used
  – No input-conditioned circuit parameters are generated
  – A single unstructured parameter tensor

        theta ∈ R^[n_qlayers, n_wires, 3]

    is trained directly and shared across all input samples

Metrics reported for both training and testing:
  – Cross-entropy loss / NLL
  – Accuracy
  – Macro-F1
  – Weighted-F1
  – Expected Calibration Error (ECE)
  – Mean prediction confidence

Composable noise models:
  depol, dephase, pauli(px,py,pz), overrot, twopauli, readout

Requires:
  torch, torchquantum, numpy
"""

import argparse
import csv
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchquantum as tq
import torchquantum.functional as tqf
from torch.utils.data import DataLoader, TensorDataset


# -------------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------------
seed = 1324
torch.manual_seed(seed)
np.random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------------------
def build_noise_cfg(args):
    models = [
        m.strip().lower()
        for m in str(args.noise_models).split(',')
        if m.strip()
    ]

    if 'none' in models:
        models = []

    cfg = dict(
        depol=args.p_depol if 'depol' in models else 0.0,
        dephase=args.p_dephase if 'dephase' in models else 0.0,
        pauli_px=args.pauli_px if 'pauli' in models else 0.0,
        pauli_py=args.pauli_py if 'pauli' in models else 0.0,
        pauli_pz=args.pauli_pz if 'pauli' in models else 0.0,
        overrot_sigma=args.overrot_sigma if 'overrot' in models else 0.0,
        p_twopauli=args.p_twopauli if 'twopauli' in models else 0.0,
        p_readout=args.p_readout if 'readout' in models else 0.0,
    )

    return cfg, models


# -------------------------------------------------------------------------
# Classification metrics
# -------------------------------------------------------------------------
def multiclass_f1_scores(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    eps: float = 1e-12,
) -> Tuple[float, float, List[float]]:
    """
    Compute macro-F1, weighted-F1, and per-class F1 without scikit-learn.
    """
    predictions = predictions.detach().view(-1).to(torch.long).cpu()
    targets = targets.detach().view(-1).to(torch.long).cpu()

    per_class_f1 = []
    supports = []

    for class_id in range(num_classes):
        pred_positive = predictions == class_id
        true_positive = targets == class_id

        tp = torch.sum(pred_positive & true_positive).item()
        fp = torch.sum(pred_positive & (~true_positive)).item()
        fn = torch.sum((~pred_positive) & true_positive).item()
        support = torch.sum(true_positive).item()

        precision = tp / max(tp + fp, eps)
        recall = tp / max(tp + fn, eps)

        if precision + recall <= eps:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)

        per_class_f1.append(float(f1))
        supports.append(int(support))

    macro_f1 = float(np.mean(per_class_f1))

    total_support = max(sum(supports), 1)
    weighted_f1 = float(
        sum(f1 * support for f1, support in zip(per_class_f1, supports))
        / total_support
    )

    return macro_f1, weighted_f1, per_class_f1


def expected_calibration_error(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 15,
) -> float:
    """
    Standard top-label Expected Calibration Error.

    ECE = sum_b (|B_b| / N) *
          |accuracy(B_b) - confidence(B_b)|
    """
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")

    probabilities = probabilities.detach().float().cpu()
    targets = targets.detach().view(-1).to(torch.long).cpu()

    confidences, predictions = torch.max(probabilities, dim=1)
    correctness = predictions.eq(targets).float()

    bin_boundaries = torch.linspace(0.0, 1.0, num_bins + 1)
    ece = torch.zeros((), dtype=torch.float32)

    for bin_idx in range(num_bins):
        lower = bin_boundaries[bin_idx]
        upper = bin_boundaries[bin_idx + 1]

        if bin_idx == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)

        bin_fraction = in_bin.float().mean()

        if bin_fraction.item() > 0:
            bin_accuracy = correctness[in_bin].mean()
            bin_confidence = confidences[in_bin].mean()

            ece += bin_fraction * torch.abs(
                bin_accuracy - bin_confidence
            )

    return float(ece.item())


def compute_classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ece_bins: int,
) -> Dict[str, object]:
    """
    Compute metrics using logits concatenated over an entire dataset.
    """
    probabilities = torch.softmax(logits.float(), dim=1)
    predictions = torch.argmax(probabilities, dim=1)

    accuracy = float(
        predictions.eq(targets).float().mean().item()
    )

    macro_f1, weighted_f1, per_class_f1 = multiclass_f1_scores(
        predictions=predictions,
        targets=targets,
        num_classes=num_classes,
    )

    ece = expected_calibration_error(
        probabilities=probabilities,
        targets=targets,
        num_bins=ece_bins,
    )

    mean_confidence = float(
        probabilities.max(dim=1).values.mean().item()
    )

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'ece': ece,
        'mean_confidence': mean_confidence,
        'per_class_f1': per_class_f1,
    }


# -------------------------------------------------------------------------
# Unstructured VQC
# -------------------------------------------------------------------------
class UnstructuredVQC(tq.QuantumModule):
    """
    Standard unstructured variational quantum circuit.

    The trainable parameter tensor is:
        angles[layer, wire, rotation_axis]

    It is independent of the input sample and is optimized directly.
    """

    def __init__(
        self,
        n_wires=12,
        n_qlayers=6,
        topology='ring',
        tensor_product_enc=True,
        add_fc=False,
        out_features=2,
        angle_init_scale=0.1,
    ):
        super().__init__()

        if topology not in {'ring', 'mps', 'tree'}:
            raise ValueError(
                f"Unsupported topology: {topology}."
            )

        self.n_wires = n_wires
        self.n_qlayers = n_qlayers
        self.topology = topology
        self.add_fc = add_fc
        self.noise_cfg = dict()

        self.angles = nn.Parameter(
            torch.randn(n_qlayers, n_wires, 3)
            * angle_init_scale
        )

        if tensor_product_enc:
            cfg = [
                {
                    'input_idx': [i],
                    'func': 'ry',
                    'wires': [i],
                }
                for i in range(n_wires)
            ]
            self.encoder = tq.GeneralEncoder(cfg)
        else:
            self.encoder = tq.AmplitudeEncoder()

        self.measure = tq.MeasureAll(tq.PauliZ)

        if add_fc:
            self.fc = nn.Linear(
                n_wires,
                out_features,
            )

    def reset_quantum_device(self, bsz):
        self.q_device.reset_states(bsz)

    # ------------------------------------------------------------------
    # Noise primitives
    # ------------------------------------------------------------------
    def _apply_single_qubit_depolarizing(self, p):
        if p <= 0:
            return

        for wire in range(self.n_wires):
            if torch.rand((), device=self.q_device.device) < p:
                error_id = torch.randint(
                    0,
                    3,
                    (),
                    device=self.q_device.device,
                ).item()

                op = (
                    tqf.x if error_id == 0
                    else tqf.y if error_id == 1
                    else tqf.z
                )

                op(
                    self.q_device,
                    wires=wire,
                    static=self.static_mode,
                    parent_graph=self.graph,
                )

    def _apply_single_qubit_dephasing(self, p):
        if p <= 0:
            return

        for wire in range(self.n_wires):
            if torch.rand((), device=self.q_device.device) < p:
                tqf.z(
                    self.q_device,
                    wires=wire,
                    static=self.static_mode,
                    parent_graph=self.graph,
                )

    def _apply_single_qubit_pauli(self, px, py, pz):
        if px + py + pz <= 0:
            return

        if px < 0 or py < 0 or pz < 0:
            raise ValueError(
                "Pauli probabilities must be non-negative."
            )

        if px + py + pz > 1:
            raise ValueError(
                "pauli_px + pauli_py + pauli_pz must not exceed 1."
            )

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

    def _apply_two_qubit_pauli_after_cnot(self, wires, p):
        if p <= 0:
            return

        if torch.rand((), device=self.q_device.device) < p:
            ops = [tqf.i, tqf.x, tqf.y, tqf.z]

            op_a = torch.randint(
                0,
                4,
                (),
                device=self.q_device.device,
            ).item()

            op_b = torch.randint(
                0,
                4,
                (),
                device=self.q_device.device,
            ).item()

            ops[op_a](
                self.q_device,
                wires=wires[0],
                static=self.static_mode,
                parent_graph=self.graph,
            )

            ops[op_b](
                self.q_device,
                wires=wires[1],
                static=self.static_mode,
                parent_graph=self.graph,
            )

    def _apply_readout_error(self, output, p):
        if p <= 0:
            return output

        sign_flip_mask = (
            torch.rand_like(output) < p
        ).float()

        return output * (
            1.0 - 2.0 * sign_flip_mask
        )

    def _inject_single_qubit_noise(self):
        self._apply_single_qubit_depolarizing(
            float(self.noise_cfg.get('depol', 0.0))
        )

        self._apply_single_qubit_dephasing(
            float(self.noise_cfg.get('dephase', 0.0))
        )

        self._apply_single_qubit_pauli(
            float(self.noise_cfg.get('pauli_px', 0.0)),
            float(self.noise_cfg.get('pauli_py', 0.0)),
            float(self.noise_cfg.get('pauli_pz', 0.0)),
        )

    # ------------------------------------------------------------------
    # Entanglement topologies
    # ------------------------------------------------------------------
    def _entangle_ring(self, two_qubit_error_probability):
        for wire in range(self.n_wires - 1):
            pair = [wire, wire + 1]

            tqf.cnot(
                self.q_device,
                wires=pair,
                static=self.static_mode,
                parent_graph=self.graph,
            )

            self._apply_two_qubit_pauli_after_cnot(
                pair,
                two_qubit_error_probability,
            )

        final_pair = [
            self.n_wires - 1,
            0,
        ]

        tqf.cnot(
            self.q_device,
            wires=final_pair,
            static=self.static_mode,
            parent_graph=self.graph,
        )

        self._apply_two_qubit_pauli_after_cnot(
            final_pair,
            two_qubit_error_probability,
        )

    def _entangle_mps(self, two_qubit_error_probability):
        for wire in range(self.n_wires - 1):
            pair = [wire, wire + 1]

            tqf.cnot(
                self.q_device,
                wires=pair,
                static=self.static_mode,
                parent_graph=self.graph,
            )

            self._apply_two_qubit_pauli_after_cnot(
                pair,
                two_qubit_error_probability,
            )

    def _entangle_tree(self, two_qubit_error_probability):
        half = self.n_wires // 2

        for wire in range(half):
            pair = [
                wire,
                wire + half,
            ]

            tqf.cnot(
                self.q_device,
                wires=pair,
                static=self.static_mode,
                parent_graph=self.graph,
            )

            self._apply_two_qubit_pauli_after_cnot(
                pair,
                two_qubit_error_probability,
            )

    def _apply_entanglement(self, two_qubit_error_probability):
        if self.topology == 'ring':
            self._entangle_ring(
                two_qubit_error_probability
            )
        elif self.topology == 'mps':
            self._entangle_mps(
                two_qubit_error_probability
            )
        else:
            self._entangle_tree(
                two_qubit_error_probability
            )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    @tq.static_support
    def forward(self, x, q_device):
        self.q_device = q_device

        batch_size = x.size(0)
        self.reset_quantum_device(batch_size)
        self.encoder(self.q_device, x)

        overrot_sigma = float(
            self.noise_cfg.get(
                'overrot_sigma',
                0.0,
            )
        )

        two_qubit_error_probability = float(
            self.noise_cfg.get(
                'p_twopauli',
                0.0,
            )
        )

        for layer in range(self.n_qlayers):
            for wire in range(self.n_wires):
                rx_angle = self.angles[layer, wire, 0]
                ry_angle = self.angles[layer, wire, 1]
                rz_angle = self.angles[layer, wire, 2]

                if overrot_sigma > 0:
                    rx_angle = (
                        rx_angle
                        + torch.randn(
                            (),
                            device=self.q_device.device,
                        )
                        * overrot_sigma
                    )

                    ry_angle = (
                        ry_angle
                        + torch.randn(
                            (),
                            device=self.q_device.device,
                        )
                        * overrot_sigma
                    )

                    rz_angle = (
                        rz_angle
                        + torch.randn(
                            (),
                            device=self.q_device.device,
                        )
                        * overrot_sigma
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

        output = self.measure(
            self.q_device
        )

        output = self._apply_readout_error(
            output,
            float(
                self.noise_cfg.get(
                    'p_readout',
                    0.0,
                )
            ),
        )

        if self.add_fc:
            return self.fc(output)

        return output


# -------------------------------------------------------------------------
# Quantum-dot dataset
# -------------------------------------------------------------------------
def load_quantum_dot_data():
    """
    Load quantum-dot charge-stability diagrams.

    The noiseless array is loaded for compatibility with the TT/TR scripts,
    while the noisy diagrams are used for both training and testing.
    """
    X_clean = np.load(
        "./mlqe_2023_edx/week1/dataset/csds_noiseless.npy"
    )

    X_noisy = np.load(
        "./mlqe_2023_edx/week1/dataset/csds.npy"
    )

    y = np.load(
        "./mlqe_2023_edx/week1/dataset/labels.npy"
    )

    _ = X_clean

    X = X_noisy.reshape(
        -1,
        2500,
    ).astype(np.float32)

    y = y.astype(np.int64)

    split = int(
        0.9 * len(y)
    )

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )

    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
    )

    return train_dataset, test_dataset


# -------------------------------------------------------------------------
# Training and evaluation
# -------------------------------------------------------------------------
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
    Run one training or evaluation epoch using quantum microbatches.

    A statevector simulator stores O(batch_size * 2**num_qubits) complex
    amplitudes, while gradient-based training additionally retains many
    intermediate states. Consequently, a conventional DataLoader batch can
    easily exhaust memory for 18--20 qubits.

    The outer DataLoader batch is therefore divided into quantum microbatches.
    During training, each microbatch loss is weighted by

        microbatch_size / outer_batch_size

    before backward(), so the accumulated gradient is exactly the gradient of
    the mean loss over the original outer batch.
    """
    if quantum_batch_size <= 0:
        raise ValueError("quantum_batch_size must be positive.")

    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_examples = 0
    all_logits = []
    all_targets = []

    for X_outer, y_outer in loader:
        outer_batch_size = X_outer.size(0)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        for start_idx in range(0, outer_batch_size, quantum_batch_size):
            end_idx = min(
                start_idx + quantum_batch_size,
                outer_batch_size,
            )

            X_batch = X_outer[start_idx:end_idx].to(
                device,
                non_blocking=True,
            )
            y_batch = y_outer[start_idx:end_idx].to(
                device,
                non_blocking=True,
            )

            microbatch_size = X_batch.size(0)

            q_device = tq.QuantumDevice(
                n_wires=num_qubits,
                bsz=microbatch_size,
            ).to(device)

            if is_training:
                logits = model(X_batch, q_device)
                loss = criterion(logits, y_batch)

                # Preserve the gradient of the mean loss over the outer batch.
                scaled_loss = (
                    loss
                    * microbatch_size
                    / outer_batch_size
                )
                scaled_loss.backward()
            else:
                with torch.no_grad():
                    logits = model(X_batch, q_device)
                    loss = criterion(logits, y_batch)

            total_loss += loss.item() * microbatch_size
            total_examples += microbatch_size

            all_logits.append(logits.detach().cpu())
            all_targets.append(y_batch.detach().cpu())

            # Release the exponentially large quantum state as early as possible.
            del q_device, logits, loss, X_batch, y_batch

        if is_training:
            optimizer.step()

    logits_full = torch.cat(all_logits, dim=0)
    targets_full = torch.cat(all_targets, dim=0)

    metrics = compute_classification_metrics(
        logits=logits_full,
        targets=targets_full,
        num_classes=num_classes,
        ece_bins=ece_bins,
    )

    average_loss = total_loss / max(total_examples, 1)
    metrics['loss'] = average_loss
    metrics['nll'] = average_loss

    return metrics

def metric_is_better(
    current_value: float,
    best_value: float,
    metric_name: str,
) -> bool:
    """
    Accuracy and F1 are maximized; ECE and loss are minimized.
    """
    if metric_name in {
        'test_ece',
        'test_loss',
        'test_nll',
    }:
        return current_value < best_value

    return current_value > best_value


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
        os.path.dirname(path) or '.',
        exist_ok=True,
    )

    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'args': vars(args),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
        },
        path,
    )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train an unstructured VQC baseline with composable quantum "
            "noise and RMT-oriented evaluation metrics."
        )
    )

    parser.add_argument(
        '--save_path',
        metavar='DIR',
        default='models',
        help='Directory for checkpoints and metric logs',
    )

    parser.add_argument(
        '--num_qubits',
        default=12,
        type=int,
        help='Number of qubits in the quantum circuit',
    )

    parser.add_argument(
        '--batch_size',
        default=64,
        type=int,
        help='Training and testing batch size',
    )

    parser.add_argument(
        '--quantum_batch_size',
        default=2,
        type=int,
        help=(
            'Microbatch size used by the quantum statevector simulator. '
            'For 20 qubits, start with 1 or 2; this avoids kernel death '
            'while preserving the effective DataLoader batch size.'
        ),
    )

    parser.add_argument(
        '--num_epochs',
        default=21,
        type=int,
        help='Number of training epochs',
    )

    parser.add_argument(
        '--depth_vqc',
        default=6,
        type=int,
        help='Number of variational layers',
    )

    parser.add_argument(
        '--lr',
        default=3e-3,
        type=float,
        help='Learning rate',
    )

    parser.add_argument(
        '--model_kind',
        default='mps',
        choices=[
            'ring',
            'mps',
            'tree',
        ],
        help='Unstructured VQC entanglement topology',
    )

    parser.add_argument(
        '--angle_init_scale',
        default=0.1,
        type=float,
        help='Standard deviation of initial VQC angles',
    )

    parser.add_argument(
        '--num_classes',
        default=2,
        type=int,
        help='Number of target classes',
    )

    parser.add_argument(
        '--add_fc',
        action='store_true',
        help=(
            'Apply a trainable linear classifier to all measured '
            'Pauli-Z expectations'
        ),
    )

    parser.add_argument(
        '--ece_bins',
        default=15,
        type=int,
        help='Number of confidence bins for ECE',
    )

    parser.add_argument(
        '--selection_metric',
        default='test_macro_f1',
        choices=[
            'test_accuracy',
            'test_macro_f1',
            'test_weighted_f1',
            'test_ece',
            'test_loss',
            'test_nll',
        ],
        help='Metric used for best-checkpoint selection',
    )

    parser.add_argument(
        '--metrics_filename',
        default='unstructured_vqc_metrics.csv',
        type=str,
        help='CSV filename for per-epoch metrics',
    )

    parser.add_argument(
        '--noise_models',
        type=str,
        default='depol,dephase,readout',
        help=(
            'Comma-separated: depol,dephase,pauli,overrot,'
            'twopauli,readout,none'
        ),
    )

    parser.add_argument(
        '--p_depol',
        type=float,
        default=0.000,
        help='Probability per single-qubit depolarizing error',
    )

    parser.add_argument(
        '--p_dephase',
        type=float,
        default=0.000,
        help='Probability per single-qubit dephasing error',
    )

    parser.add_argument(
        '--pauli_px',
        type=float,
        default=0.000,
        help='Pauli-X probability',
    )

    parser.add_argument(
        '--pauli_py',
        type=float,
        default=0.000,
        help='Pauli-Y probability',
    )

    parser.add_argument(
        '--pauli_pz',
        type=float,
        default=0.000,
        help='Pauli-Z probability',
    )

    parser.add_argument(
        '--overrot_sigma',
        type=float,
        default=0.00,
        help='Std. dev. of coherent angle jitter in radians',
    )

    parser.add_argument(
        '--p_twopauli',
        type=float,
        default=0.00,
        help='Probability per CNOT of a random two-qubit Pauli',
    )

    parser.add_argument(
        '--p_readout',
        type=float,
        default=0.00,
        help='Probability of flipping each Z expectation sign',
    )

    args = parser.parse_args()

    if args.num_classes <= 1:
        raise ValueError(
            "num_classes must be at least 2."
        )

    if not args.add_fc and args.num_classes > args.num_qubits:
        raise ValueError(
            "With add_fc=False, num_classes cannot exceed num_qubits."
        )

    if args.ece_bins <= 0:
        raise ValueError(
            "ece_bins must be positive."
        )

    if args.quantum_batch_size <= 0:
        raise ValueError(
            "quantum_batch_size must be positive."
        )

    if args.p_depol < 0 or args.p_dephase < 0:
        raise ValueError(
            "Noise probabilities must be non-negative."
        )

    if args.p_twopauli < 0 or args.p_readout < 0:
        raise ValueError(
            "Noise probabilities must be non-negative."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_dataset, test_dataset = load_quantum_dot_data()

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

    model = UnstructuredVQC(
        n_wires=args.num_qubits,
        n_qlayers=args.depth_vqc,
        topology=args.model_kind,
        tensor_product_enc=True,
        add_fc=args.add_fc,
        out_features=args.num_classes,
        angle_init_scale=args.angle_init_scale,
    ).to(device)

    noise_cfg, noise_models = build_noise_cfg(args)
    model.noise_cfg = noise_cfg

    print(
        "Baseline: unstructured VQC"
    )

    print(
        "Topology:",
        args.model_kind,
    )

    print(
        "Noise models:",
        ','.join(noise_models)
        if noise_models
        else 'none',
    )

    print(
        "Noise cfg:",
        noise_cfg,
    )

    print(
        "Parameters:",
        sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
    )

    print(
        "Unstructured angle parameters:",
        model.angles.numel(),
    )

    print(
        "DataLoader batch size:",
        args.batch_size,
    )

    print(
        "Quantum microbatch size:",
        args.quantum_batch_size,
    )

    estimated_state_mb = (
        args.quantum_batch_size
        * (2 ** args.num_qubits)
        * 8
        / (1024 ** 2)
    )

    print(
        "Approx. complex64 statevector memory per quantum microbatch "
        f"(excluding gradients/intermediates): {estimated_state_mb:.1f} MB"
    )

    print(
        f"ECE bins: {args.ece_bins}"
    )

    print(
        f"Best-checkpoint criterion: "
        f"{args.selection_metric}"
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
        (
            f"best_unstructured_vqc_"
            f"{args.model_kind}.pt"
        ),
    )

    csv_fields = [
        'epoch',
        'learning_rate',
        'train_loss',
        'train_nll',
        'train_accuracy',
        'train_macro_f1',
        'train_weighted_f1',
        'train_ece',
        'train_mean_confidence',
        'test_loss',
        'test_nll',
        'test_accuracy',
        'test_macro_f1',
        'test_weighted_f1',
        'test_ece',
        'test_mean_confidence',
    ]

    if args.selection_metric in {
        'test_ece',
        'test_loss',
        'test_nll',
    }:
        best_metric_value = float('inf')
    else:
        best_metric_value = -float('inf')

    with open(
        metrics_path,
        mode='w',
        newline='',
        encoding='utf-8',
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

            current_lr = optimizer.param_groups[0]['lr']

            row = {
                'epoch': epoch,
                'learning_rate': current_lr,
                'train_loss': train_metrics['loss'],
                'train_nll': train_metrics['nll'],
                'train_accuracy': train_metrics['accuracy'],
                'train_macro_f1': train_metrics['macro_f1'],
                'train_weighted_f1': train_metrics['weighted_f1'],
                'train_ece': train_metrics['ece'],
                'train_mean_confidence': train_metrics['mean_confidence'],
                'test_loss': test_metrics['loss'],
                'test_nll': test_metrics['nll'],
                'test_accuracy': test_metrics['accuracy'],
                'test_macro_f1': test_metrics['macro_f1'],
                'test_weighted_f1': test_metrics['weighted_f1'],
                'test_ece': test_metrics['ece'],
                'test_mean_confidence': test_metrics['mean_confidence'],
            }

            writer.writerow(row)
            csv_file.flush()

            selected_value = float(
                row[args.selection_metric]
            )

            if metric_is_better(
                current_value=selected_value,
                best_value=best_metric_value,
                metric_name=args.selection_metric,
            ):
                best_metric_value = selected_value

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
                f"Train loss: {train_metrics['loss']:.4f}, "
                f"acc: {train_metrics['accuracy']:.4f}, "
                f"macro-F1: {train_metrics['macro_f1']:.4f}, "
                f"weighted-F1: {train_metrics['weighted_f1']:.4f}, "
                f"ECE: {train_metrics['ece']:.4f} | "
                f"Test loss: {test_metrics['loss']:.4f}, "
                f"acc: {test_metrics['accuracy']:.4f}, "
                f"macro-F1: {test_metrics['macro_f1']:.4f}, "
                f"weighted-F1: {test_metrics['weighted_f1']:.4f}, "
                f"ECE: {test_metrics['ece']:.4f}"
            )

    print(
        f"Metrics saved to: {metrics_path}"
    )

    print(
        f"Best checkpoint saved to: "
        f"{checkpoint_path}"
    )

    print(
        f"Best {args.selection_metric}: "
        f"{best_metric_value:.6f}"
    )
