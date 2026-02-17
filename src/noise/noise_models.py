# src/noise/noise_models.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import stim

NoiseArg = Optional[Union[float, Sequence[float]]]
NoiseSpec = Tuple[str, List[Any], NoiseArg]


# ---------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------
META_GATES = {
    "DETECTOR",
    "OBSERVABLE_INCLUDE",
    "SHIFT_COORDS",
    "QUBIT_COORDS",
    "TICK",
    "MPAD",
}

MEASUREMENT_GATES = {
    "M",
    "MX",
    "MY",
    "MZ",
    "MR",
    "MRX",
    "MRY",
    "MRZ",
    "MPP",
}

RESET_GATES = {
    "R",
    "RX",
    "RY",
    "RZ",
}

# Reasonable set of 2q gates to process targets by pairs
TWO_QUBIT_GATES = {
    "CX",
    "CY",
    "CZ",
    "XCX",
    "XCY",
    "XCZ",
    "YCX",
    "YCY",
    "YCZ",
    "SWAP",
    "ISWAP",
    "SQRT_XX",
    "SQRT_YY",
    "SQRT_ZZ",
    "CNOT",
}


def _validate_prob(name: str, value: float) -> None:
    if not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{name} must be in [0,1]. Received: {value}")


def _extract_qubit_targets(targets: Sequence[Any]) -> List[int]:
    """Extract qubit indices from a stim target list."""
    out: List[int] = []
    seen = set()

    for t in targets:
        q: Optional[int] = None

        # stim GateTarget
        if hasattr(t, "is_qubit_target") and getattr(t, "is_qubit_target"):
            if hasattr(t, "value"):
                q = int(getattr(t, "value"))
            else:
                try:
                    q = int(t)
                except Exception:
                    q = None
        else:
            # Raw integer target (for compatibility)
            if isinstance(t, int) and t >= 0:
                q = int(t)

        if q is not None and q not in seen:
            out.append(q)
            seen.add(q)

    return out


def _collect_qubits(circuit: stim.Circuit) -> List[int]:
    qubits = set()
    for op in circuit:
        if hasattr(op, "body_copy"):  # CircuitRepeatBlock
            body = op.body_copy()
            qubits.update(_collect_qubits(body))
        else:  # CircuitInstruction
            qubits.update(_extract_qubit_targets(op.targets_copy()))
    return sorted(qubits)


def _paired_qubits_for_two_qubit_gate(gate: str, qubits: List[int]) -> List[Tuple[int, int]]:
    """Return pairs (q0,q1), (q2,q3), ... when a 2q gate is broadcast."""
    if len(qubits) < 2:
        return []

    if gate in TWO_QUBIT_GATES and len(qubits) % 2 == 0:
        return [(qubits[i], qubits[i + 1]) for i in range(0, len(qubits), 2)]

    if len(qubits) == 2:
        return [(qubits[0], qubits[1])]

    return []


def _is_measurement_gate(gate: str) -> bool:
    if gate in MEASUREMENT_GATES:
        return True
    # Extra support by prefix
    return gate.startswith("M")


def _is_reset_gate(gate: str) -> bool:
    return gate in RESET_GATES


def _append_noise_spec(out: stim.Circuit, spec: NoiseSpec) -> None:
    name, targets, arg = spec
    if not targets:
        return

    if arg is None:
        out.append(name, targets)
    elif isinstance(arg, (list, tuple)):
        out.append(name, targets, [float(x) for x in arg])
    else:
        out.append(name, targets, float(arg))


# ---------------------------------------------------------------------
# Noise model API
# ---------------------------------------------------------------------
@dataclass
class NoiseModel:
    """
    Base class for noise models.

    Contrato principal:
      - apply_to_circuit(circuit) -> stim.Circuit
      - subclases sobreescriben pre_noise_specs / post_noise_specs
    """

    p: float = 0.001
    seed: Optional[int] = None  # reservado (si en futuro se usa noise dependiente de seed)

    def __post_init__(self) -> None:
        _validate_prob("p", self.p)

    def pre_noise_specs(
        self,
        *,
        gate: str,
        qubits: List[int],
        gate_args: List[float],
        all_qubits: List[int],
    ) -> List[NoiseSpec]:
        return []

    def post_noise_specs(
        self,
        *,
        gate: str,
        qubits: List[int],
        gate_args: List[float],
        all_qubits: List[int],
    ) -> List[NoiseSpec]:
        return []

    def apply_to_circuit(self, circuit: stim.Circuit) -> stim.Circuit:
        if not isinstance(circuit, stim.Circuit):
            raise TypeError(f"circuit must be stim.Circuit. Received: {type(circuit)}")
        all_qubits = _collect_qubits(circuit)
        return self._transform_circuit(circuit, all_qubits=all_qubits)

    def _transform_circuit(self, circuit: stim.Circuit, *, all_qubits: List[int]) -> stim.Circuit:
        out = stim.Circuit()

        for op in circuit:
            # Repeat block (recursive)
            if hasattr(op, "body_copy"):
                repeat_count = int(op.repeat_count)
                body = op.body_copy()
                body_noisy = self._transform_circuit(body, all_qubits=all_qubits)
                out.append(stim.CircuitRepeatBlock(repeat_count, body_noisy))
                continue

            # Regular instruction
            gate = op.name
            targets = op.targets_copy()
            gate_args = list(op.gate_args_copy())
            qubits = _extract_qubit_targets(targets)

            # Noise pre
            for spec in self.pre_noise_specs(
                gate=gate,
                qubits=qubits,
                gate_args=gate_args,
                all_qubits=all_qubits,
            ):
                _append_noise_spec(out, spec)

            # Original instruction
            out.append(gate, targets, gate_args)

            # Noise post
            for spec in self.post_noise_specs(
                gate=gate,
                qubits=qubits,
                gate_args=gate_args,
                all_qubits=all_qubits,
            ):
                _append_noise_spec(out, spec)

        return out


@dataclass
class DepolarizingNoise(NoiseModel):
    """Uniform depolarizing noise after data operations."""

    def post_noise_specs(
        self,
        *,
        gate: str,
        qubits: List[int],
        gate_args: List[float],
        all_qubits: List[int],
    ) -> List[NoiseSpec]:
        if not qubits:
            return []
        if gate in META_GATES or _is_measurement_gate(gate):
            return []

        specs: List[NoiseSpec] = []
        pairs = _paired_qubits_for_two_qubit_gate(gate, qubits)

        if pairs:
            for a, b in pairs:
                specs.append(("DEPOLARIZE2", [a, b], self.p))
        else:
            for q in qubits:
                specs.append(("DEPOLARIZE1", [q], self.p))
        return specs


@dataclass
class BiasedNoise(NoiseModel):
    """
    Z-biased noise.
    eta >= 1 is typical (larger eta => more Z weight).
    """

    eta: float = 100.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.eta <= 0:
            raise ValueError(f"eta must be > 0. Received: {self.eta}")

    @property
    def px(self) -> float:
        return float(self.p / (self.eta + 2.0))

    @property
    def py(self) -> float:
        return float(self.p / (self.eta + 2.0))

    @property
    def pz(self) -> float:
        return float(self.p * self.eta / (self.eta + 2.0))

    def post_noise_specs(
        self,
        *,
        gate: str,
        qubits: List[int],
        gate_args: List[float],
        all_qubits: List[int],
    ) -> List[NoiseSpec]:
        if not qubits:
            return []
        if gate in META_GATES or _is_measurement_gate(gate):
            return []

        args = [self.px, self.py, self.pz]
        specs: List[NoiseSpec] = []
        for q in qubits:
            specs.append(("PAULI_CHANNEL_1", [q], args))
        return specs


@dataclass
class CircuitLevelNoise(NoiseModel):
    """
    Circuit-level noise:
      - p1 for gates 1q
      - p2 for gates 2q
      - pm for readout error (X_ERROR before measurement)
      - pr for post-reset error
    """

    p1: Optional[float] = None
    p2: Optional[float] = None
    pm: Optional[float] = None
    pr: Optional[float] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        base = float(self.p)
        self.p1 = base if self.p1 is None else float(self.p1)
        self.p2 = min(1.0, 2.0 * base) if self.p2 is None else float(self.p2)
        self.pm = base if self.pm is None else float(self.pm)
        self.pr = base if self.pr is None else float(self.pr)

        _validate_prob("p1", self.p1)
        _validate_prob("p2", self.p2)
        _validate_prob("pm", self.pm)
        _validate_prob("pr", self.pr)

    def pre_noise_specs(
        self,
        *,
        gate: str,
        qubits: List[int],
        gate_args: List[float],
        all_qubits: List[int],
    ) -> List[NoiseSpec]:
        # Measurement error: approximated as X_ERROR before measurement
        if _is_measurement_gate(gate) and qubits and self.pm > 0:
            return [("X_ERROR", [q], self.pm) for q in qubits]
        return []

    def post_noise_specs(
        self,
        *,
        gate: str,
        qubits: List[int],
        gate_args: List[float],
        all_qubits: List[int],
    ) -> List[NoiseSpec]:
        if not qubits:
            return []
        if gate in META_GATES or _is_measurement_gate(gate):
            return []

        if _is_reset_gate(gate):
            if self.pr <= 0:
                return []
            return [("X_ERROR", [q], self.pr) for q in qubits]

        pairs = _paired_qubits_for_two_qubit_gate(gate, qubits)
        if pairs:
            if self.p2 <= 0:
                return []
            return [("DEPOLARIZE2", [a, b], self.p2) for a, b in pairs]

        if self.p1 <= 0:
            return []
        return [("DEPOLARIZE1", [q], self.p1) for q in qubits]


@dataclass
class PhenomenologicalNoise(NoiseModel):
    """
    Simplified phenomenological noise:
      - p_meas: X_ERROR antes de medir
      - p_idle: Z_ERROR tras TICK (sobre qubits del circuit)
      - p_reset: X_ERROR tras reset
      - p_data: Z_ERROR tras operaciones de datos (opcional)
    """

    p_meas: Optional[float] = None
    p_idle: Optional[float] = None
    p_reset: float = 0.0
    p_data: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        base = float(self.p)
        self.p_meas = base if self.p_meas is None else float(self.p_meas)
        self.p_idle = (0.1 * base) if self.p_idle is None else float(self.p_idle)

        _validate_prob("p_meas", self.p_meas)
        _validate_prob("p_idle", self.p_idle)
        _validate_prob("p_reset", self.p_reset)
        _validate_prob("p_data", self.p_data)

    def pre_noise_specs(
        self,
        *,
        gate: str,
        qubits: List[int],
        gate_args: List[float],
        all_qubits: List[int],
    ) -> List[NoiseSpec]:
        if _is_measurement_gate(gate) and qubits and self.p_meas > 0:
            return [("X_ERROR", [q], self.p_meas) for q in qubits]
        return []

    def post_noise_specs(
        self,
        *,
        gate: str,
        qubits: List[int],
        gate_args: List[float],
        all_qubits: List[int],
    ) -> List[NoiseSpec]:
        specs: List[NoiseSpec] = []

        # Idle dephasing en TICK
        if gate == "TICK":
            if self.p_idle > 0 and all_qubits:
                for q in all_qubits:
                    specs.append(("Z_ERROR", [q], self.p_idle))
            return specs

        if not qubits:
            return specs
        if gate in META_GATES or _is_measurement_gate(gate):
            return specs

        if _is_reset_gate(gate) and self.p_reset > 0:
            for q in qubits:
                specs.append(("X_ERROR", [q], self.p_reset))
            return specs

        if self.p_data > 0:
            for q in qubits:
                specs.append(("Z_ERROR", [q], self.p_data))

        return specs


@dataclass
class CorrelatedNoise(NoiseModel):
    """
    Approximate correlated noise:
      1) Base independent error on active qubits: <PAULI>_ERROR(p)
      2) Correlated errors on neighboring pairs: CORRELATED_ERROR(p_pair) P(q_i) P(q_j)

    p_pair = p * correlation_strength / distance, truncado a [0,1].
    """

    correlation_length: int = 1
    correlation_strength: float = 0.3
    topology: str = "line"  # "line" | "grid"
    pauli: str = "Z"        # "X" | "Y" | "Z"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.correlation_length < 1:
            raise ValueError(
                f"correlation_length must be >= 1. Received: {self.correlation_length}"
            )
        _validate_prob("correlation_strength", self.correlation_strength)
        self.topology = self.topology.lower().strip()
        if self.topology not in {"line", "grid"}:
            raise ValueError(f"topology must be 'line' or 'grid'. Received: {self.topology}")

        self.pauli = self.pauli.upper().strip()
        if self.pauli not in {"X", "Y", "Z"}:
            raise ValueError(f"pauli must be X/Y/Z. Received: {self.pauli}")

    @property
    def pauli_error_gate(self) -> str:
        return f"{self.pauli}_ERROR"

    def _pauli_target(self, q: int) -> stim.GateTarget:
        if self.pauli == "X":
            return stim.target_x(q)
        if self.pauli == "Y":
            return stim.target_y(q)
        return stim.target_z(q)

    def get_neighbors(
        self,
        qubit_idx: int,
        max_distance: int,
        *,
        n_qubits: int,
    ) -> List[Tuple[int, int]]:
        """
        Return a list of (neighbor, distance) up to max_distance.
        """
        if n_qubits <= 0 or max_distance < 1:
            return []

        neighbors: List[Tuple[int, int]] = []

        if self.topology == "line":
            for d in range(1, max_distance + 1):
                ql = qubit_idx - d
                qr = qubit_idx + d
                if 0 <= ql < n_qubits:
                    neighbors.append((ql, d))
                if 0 <= qr < n_qubits:
                    neighbors.append((qr, d))
            return neighbors

        # topology == "grid"
        width = int(math.ceil(math.sqrt(n_qubits)))
        r0, c0 = divmod(qubit_idx, width)

        for q in range(n_qubits):
            if q == qubit_idx:
                continue
            r, c = divmod(q, width)
            dist = abs(r - r0) + abs(c - c0)
            if 1 <= dist <= max_distance:
                neighbors.append((q, dist))

        return neighbors

    def post_noise_specs(
        self,
        *,
        gate: str,
        qubits: List[int],
        gate_args: List[float],
        all_qubits: List[int],
    ) -> List[NoiseSpec]:
        if not qubits:
            return []
        if gate in META_GATES or _is_measurement_gate(gate):
            return []

        specs: List[NoiseSpec] = []

        # 1) Base independent error on active qubits
        if self.p > 0:
            for q in qubits:
                specs.append((self.pauli_error_gate, [q], self.p))

        # 2) Correlation by nearby pairs
        if self.correlation_strength <= 0:
            return specs

        n_qubits = (max(all_qubits) + 1) if all_qubits else (max(qubits) + 1)

        # map pair -> minimum observed distance
        pair_dist: Dict[Tuple[int, int], int] = {}
        for q in qubits:
            for nb, dist in self.get_neighbors(
                qubit_idx=q,
                max_distance=self.correlation_length,
                n_qubits=n_qubits,
            ):
                if nb == q:
                    continue
                a, b = (q, nb) if q < nb else (nb, q)
                prev = pair_dist.get((a, b))
                if prev is None or dist < prev:
                    pair_dist[(a, b)] = dist

        for (a, b), dist in pair_dist.items():
            p_pair = min(1.0, self.p * self.correlation_strength / float(dist))
            if p_pair <= 0:
                continue

            # CORRELATED_ERROR requires Pauli targets (X/Y/Z target)
            targets = [self._pauli_target(a), self._pauli_target(b)]
            specs.append(("CORRELATED_ERROR", targets, p_pair))

        return specs


# ---------------------------------------------------------------------
# Public factory / helpers
# ---------------------------------------------------------------------
NoiseModelLike = Union[str, Dict[str, Any], NoiseModel]


def build_noise_model(model: NoiseModelLike, **kwargs: Any) -> NoiseModel:
    """
    Build a NoiseModel from:
      - instancia NoiseModel
      - string ("depolarizing", "biased", "circuit_level", "phenomenological", "correlated")
      - dict {"type": "...", ...params...}
    """
    if isinstance(model, NoiseModel):
        return model

    params: Dict[str, Any] = dict(kwargs)

    if isinstance(model, dict):
        if "type" not in model:
            raise ValueError("If model is dict, it must include key 'type'.")
        mtype = str(model["type"]).strip().lower()
        model_params = {k: v for k, v in model.items() if k != "type"}
        params = {**model_params, **params}
    elif isinstance(model, str):
        mtype = model.strip().lower()
    else:
        raise TypeError(
            "model must be NoiseModel | str | dict. "
            f"Received: {type(model)}"
        )

    aliases = {
        "depolarizing": "depolarizing",
        "depolarising": "depolarizing",
        "depol": "depolarizing",
        "biased": "biased",
        "biased_z": "biased",
        "circuit_level": "circuit_level",
        "circuit": "circuit_level",
        "phenomenological": "phenomenological",
        "phenom": "phenomenological",
        "correlated": "correlated",
    }
    key = aliases.get(mtype, mtype)

    if key == "depolarizing":
        return DepolarizingNoise(**params)
    if key == "biased":
        return BiasedNoise(**params)
    if key == "circuit_level":
        return CircuitLevelNoise(**params)
    if key == "phenomenological":
        return PhenomenologicalNoise(**params)
    if key == "correlated":
        return CorrelatedNoise(**params)

    valid = ", ".join(sorted(set(aliases.values())))
    raise ValueError(f"Unknown noise model: '{mtype}'. Valid options: {valid}")


def apply_noise_model(
    circuit: stim.Circuit,
    model: NoiseModelLike = "depolarizing",
    **kwargs: Any,
) -> stim.Circuit:
    """
    Convenience helper:
      noisy_circuit = apply_noise_model(circuit, model="biased", p=0.01, eta=100)
    """
    noise_model = build_noise_model(model, **kwargs)
    return noise_model.apply_to_circuit(circuit)


__all__ = [
    "NoiseModel",
    "DepolarizingNoise",
    "BiasedNoise",
    "CircuitLevelNoise",
    "PhenomenologicalNoise",
    "CorrelatedNoise",
    "build_noise_model",
    "apply_noise_model",
]
