from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


DecoderDistanceKey = Tuple[str, int]
LatencyTable = Mapping[DecoderDistanceKey, float]


def normalize_decoder_label(decoder: str) -> str:
    if not isinstance(decoder, str) or not decoder.strip():
        raise ValueError("decoder label must be a non-empty string")
    return decoder.strip().lower()


def _validate_distance(distance: int) -> int:
    if not isinstance(distance, int) or distance <= 0:
        raise ValueError("distance must be a positive integer")
    return int(distance)


def lookup_latency_sec(
    decoder: str,
    distance: int,
    latency_table: LatencyTable,
) -> float:
    key = (normalize_decoder_label(decoder), _validate_distance(distance))
    if key not in latency_table:
        raise KeyError(f"missing latency entry for decoder={key[0]!r}, distance={key[1]}")
    latency = float(latency_table[key])
    if latency < 0.0:
        raise ValueError(f"latency must be >= 0, got {latency} for key={key}")
    return latency


@dataclass(frozen=True)
class HardwareArchitecture:
    name: str
    cycle_time_sec: float
    classical_budget_sec: float

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("name must be a non-empty string")
        if float(self.cycle_time_sec) <= 0.0:
            raise ValueError("cycle_time_sec must be > 0")
        if float(self.classical_budget_sec) <= 0.0:
            raise ValueError("classical_budget_sec must be > 0")

    def is_compatible(
        self,
        decoder: str,
        distance: int,
        latency_table: LatencyTable,
        *,
        safety_factor: float = 1.0,
    ) -> bool:
        if float(safety_factor) <= 0.0:
            raise ValueError("safety_factor must be > 0")
        latency = lookup_latency_sec(decoder, distance, latency_table)
        return bool(latency * float(safety_factor) <= float(self.classical_budget_sec))

    def budget_utilization(
        self,
        decoder: str,
        distance: int,
        latency_table: LatencyTable,
    ) -> float:
        latency = lookup_latency_sec(decoder, distance, latency_table)
        return float(latency / float(self.classical_budget_sec))


class GoogleSuperconducting(HardwareArchitecture):
    def __init__(self) -> None:
        super().__init__(
            name="GoogleSuperconducting",
            cycle_time_sec=1e-6,
            classical_budget_sec=200e-9,
        )


class IBMEagle(HardwareArchitecture):
    def __init__(self) -> None:
        super().__init__(
            name="IBMEagle",
            cycle_time_sec=1.5e-6,
            classical_budget_sec=300e-9,
        )


class IonQForte(HardwareArchitecture):
    def __init__(self) -> None:
        super().__init__(
            name="IonQForte",
            cycle_time_sec=10e-6,
            classical_budget_sec=2e-6,
        )


class PsiQuantumPhotonic(HardwareArchitecture):
    def __init__(self) -> None:
        super().__init__(
            name="PsiQuantumPhotonic",
            cycle_time_sec=10e-9,
            classical_budget_sec=2e-9,
        )


def default_architectures() -> List[HardwareArchitecture]:
    return [
        GoogleSuperconducting(),
        IBMEagle(),
        IonQForte(),
        PsiQuantumPhotonic(),
    ]


def build_latency_table(
    rows: Iterable[Mapping[str, Any]],
    *,
    decoder_key: str = "decoder_label",
    distance_key: str = "distance",
    latency_key: str = "predicted_decode_time_sec",
) -> Dict[DecoderDistanceKey, float]:
    out: Dict[DecoderDistanceKey, float] = {}
    for row in rows:
        decoder = normalize_decoder_label(str(row[decoder_key]))
        distance = _validate_distance(int(row[distance_key]))
        latency = float(row[latency_key])
        if latency < 0.0:
            raise ValueError(f"latency must be >= 0, got {latency} for row={row}")
        out[(decoder, distance)] = latency
    if not out:
        raise ValueError("empty latency table")
    return out


def compatibility_matrix(
    architecture: HardwareArchitecture,
    decoder_labels: Sequence[str],
    distances: Sequence[int],
    latency_table: LatencyTable,
    *,
    safety_factor: float = 1.0,
) -> List[List[int]]:
    matrix: List[List[int]] = []
    for decoder in decoder_labels:
        row: List[int] = []
        for distance in distances:
            ok = architecture.is_compatible(
                decoder=decoder,
                distance=int(distance),
                latency_table=latency_table,
                safety_factor=safety_factor,
            )
            row.append(1 if ok else 0)
        matrix.append(row)
    return matrix


__all__ = [
    "DecoderDistanceKey",
    "LatencyTable",
    "HardwareArchitecture",
    "GoogleSuperconducting",
    "IBMEagle",
    "IonQForte",
    "PsiQuantumPhotonic",
    "normalize_decoder_label",
    "lookup_latency_sec",
    "default_architectures",
    "build_latency_table",
    "compatibility_matrix",
]

