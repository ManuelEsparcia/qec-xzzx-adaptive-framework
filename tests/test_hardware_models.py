from __future__ import annotations

import math

from src.hardware.hardware_models import (
    GoogleSuperconducting,
    IBMEagle,
    IonQForte,
    PsiQuantumPhotonic,
    build_latency_table,
    compatibility_matrix,
    default_architectures,
    lookup_latency_sec,
)


def test_default_architectures_have_expected_budgets() -> None:
    archs = default_architectures()
    names = [a.name for a in archs]
    assert names == [
        "GoogleSuperconducting",
        "IBMEagle",
        "IonQForte",
        "PsiQuantumPhotonic",
    ]

    by_name = {a.name: a for a in archs}
    assert math.isclose(by_name["GoogleSuperconducting"].classical_budget_sec, 200e-9, rel_tol=0.0, abs_tol=1e-18)
    assert math.isclose(by_name["IBMEagle"].classical_budget_sec, 300e-9, rel_tol=0.0, abs_tol=1e-18)
    assert math.isclose(by_name["IonQForte"].classical_budget_sec, 2e-6, rel_tol=0.0, abs_tol=1e-18)
    assert math.isclose(by_name["PsiQuantumPhotonic"].classical_budget_sec, 2e-9, rel_tol=0.0, abs_tol=1e-18)


def test_lookup_latency_and_is_compatible() -> None:
    table = {
        ("mwpm", 3): 120e-9,
        ("uf", 3): 260e-9,
        ("adaptive_g0.35", 3): 190e-9,
    }
    google = GoogleSuperconducting()
    ibm = IBMEagle()

    assert math.isclose(lookup_latency_sec("mwpm", 3, table), 120e-9, rel_tol=0.0, abs_tol=1e-18)
    assert google.is_compatible("mwpm", 3, table) is True
    assert google.is_compatible("uf", 3, table) is False
    assert google.is_compatible("adaptive_g0.35", 3, table) is True
    assert ibm.is_compatible("uf", 3, table) is True


def test_build_latency_table_and_matrix_shape() -> None:
    rows = [
        {"decoder_label": "mwpm", "distance": 3, "predicted_decode_time_sec": 100e-9},
        {"decoder_label": "mwpm", "distance": 5, "predicted_decode_time_sec": 210e-9},
        {"decoder_label": "uf", "distance": 3, "predicted_decode_time_sec": 180e-9},
        {"decoder_label": "uf", "distance": 5, "predicted_decode_time_sec": 290e-9},
    ]
    table = build_latency_table(rows)
    matrix = compatibility_matrix(
        architecture=IBMEagle(),
        decoder_labels=["mwpm", "uf"],
        distances=[3, 5],
        latency_table=table,
    )
    assert matrix == [
        [1, 1],
        [1, 1],
    ]

    photonic = PsiQuantumPhotonic()
    photonic_matrix = compatibility_matrix(
        architecture=photonic,
        decoder_labels=["mwpm", "uf"],
        distances=[3, 5],
        latency_table=table,
    )
    assert photonic_matrix == [
        [0, 0],
        [0, 0],
    ]


def test_architecture_classes_instantiation() -> None:
    assert GoogleSuperconducting().cycle_time_sec > 0
    assert IBMEagle().cycle_time_sec > 0
    assert IonQForte().cycle_time_sec > 0
    assert PsiQuantumPhotonic().cycle_time_sec > 0

