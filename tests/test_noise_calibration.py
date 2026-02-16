# tests/test_noise_calibration.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest
import stim

from src.noise.noise_calibration import (
    CalibrationCase,
    CalibrationConfig,
    SweepSpec,
    build_base_xzzx_circuit,
    estimate_ler_and_time,
    load_calibration_report,
    run_multi_model_calibration,
    run_noise_sweep,
    run_single_calibration_point,
    save_calibration_report,
)


# =============================================================================
# Validaciones de dataclasses
# =============================================================================
def test_calibration_case_valid() -> None:
    c = CalibrationCase(case_name="d3r2", distance=3, rounds=2, p=0.01, logical_basis="x")
    assert c.distance == 3
    assert c.rounds == 2
    assert c.logical_basis == "x"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"case_name": "bad_d_even", "distance": 4, "rounds": 2, "p": 0.01, "logical_basis": "x"},
        {"case_name": "bad_d_small", "distance": 1, "rounds": 2, "p": 0.01, "logical_basis": "x"},
        {"case_name": "bad_rounds", "distance": 3, "rounds": 0, "p": 0.01, "logical_basis": "x"},
        {"case_name": "bad_p_lo", "distance": 3, "rounds": 2, "p": -0.1, "logical_basis": "x"},
        {"case_name": "bad_p_hi", "distance": 3, "rounds": 2, "p": 1.2, "logical_basis": "x"},
        {"case_name": "bad_basis", "distance": 3, "rounds": 2, "p": 0.01, "logical_basis": "y"},
    ],
)
def test_calibration_case_invalid_raises(kwargs: Dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        CalibrationCase(**kwargs)


def test_sweep_spec_valid() -> None:
    s = SweepSpec(
        model_type="depolarizing",
        param_name="p",
        values=(0.001, 0.005, 0.01),
        objective="min_ler",
    )
    assert s.model_type == "depolarizing"
    assert len(s.values) == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "model_type": "",
            "param_name": "p",
            "values": (0.01,),
            "objective": "min_ler",
        },
        {
            "model_type": "depolarizing",
            "param_name": "",
            "values": (0.01,),
            "objective": "min_ler",
        },
        {
            "model_type": "depolarizing",
            "param_name": "p",
            "values": (),
            "objective": "min_ler",
        },
        {
            "model_type": "depolarizing",
            "param_name": "p",
            "values": (0.01,),
            "objective": "unknown_objective",
        },
    ],
)
def test_sweep_spec_invalid_raises(kwargs: Dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SweepSpec(**kwargs)


def test_calibration_config_valid() -> None:
    cfg = CalibrationConfig(shots=100, seed=123, keep_soft_info_samples=5)
    assert cfg.shots == 100
    assert cfg.keep_soft_info_samples == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shots": 0, "seed": 1, "keep_soft_info_samples": 0},
        {"shots": -1, "seed": 1, "keep_soft_info_samples": 0},
        {"shots": 10, "seed": 1, "keep_soft_info_samples": -2},
    ],
)
def test_calibration_config_invalid_raises(kwargs: Dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        CalibrationConfig(**kwargs)


# =============================================================================
# Smoke real del builder / simulación
# =============================================================================
def test_build_base_xzzx_circuit_smoke() -> None:
    circuit = build_base_xzzx_circuit(distance=3, rounds=2, logical_basis="x")
    assert isinstance(circuit, stim.Circuit)

    dem = circuit.detector_error_model(decompose_errors=True)
    assert dem.num_detectors > 0
    assert dem.num_observables >= 1


def test_estimate_ler_and_time_invalid_inputs_raise() -> None:
    dummy = stim.Circuit()
    with pytest.raises(ValueError):
        estimate_ler_and_time(dummy, shots=0, seed=1)

    with pytest.raises(ValueError):
        estimate_ler_and_time(dummy, shots=10, seed=1, keep_soft_info_samples=-1)


def test_run_single_calibration_point_real_smoke() -> None:
    case = CalibrationCase(
        case_name="d3_r2",
        distance=3,
        rounds=2,
        p=0.005,
        logical_basis="x",
    )
    cfg = CalibrationConfig(shots=40, seed=2026, keep_soft_info_samples=5)

    out = run_single_calibration_point(
        case=case,
        model_type="depolarizing",
        sweep_param_name="p",
        sweep_value=0.005,
        config=cfg,
        base_params=None,
        experiment_seed=12345,
    )

    assert out["status"] == "ok"
    assert out["case_name"] == "d3_r2"
    assert out["model_type"] == "depolarizing"
    assert 0.0 <= float(out["ler"]) <= 1.0
    assert out["num_detectors"] > 0
    assert out["num_observables"] >= 1
    assert out["avg_decode_time_sec"] >= 0.0
    assert "soft_info_samples" in out
    assert len(out["soft_info_samples"]) <= 5


def test_run_single_calibration_point_invalid_probability_raises() -> None:
    case = CalibrationCase(case_name="d3_r2", distance=3, rounds=2, p=0.005, logical_basis="x")
    cfg = CalibrationConfig(shots=20, seed=7, keep_soft_info_samples=0)

    with pytest.raises(ValueError):
        run_single_calibration_point(
            case=case,
            model_type="depolarizing",
            sweep_param_name="p",
            sweep_value=1.5,  # fuera de [0,1]
            config=cfg,
            base_params=None,
            experiment_seed=999,
        )


# =============================================================================
# run_noise_sweep (mockeado para ser determinista y rápido)
# =============================================================================
def test_run_noise_sweep_contract_min_ler(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.noise.noise_calibration as nc

    def fake_build_base_xzzx_circuit(*, distance: int, rounds: int, logical_basis: str) -> stim.Circuit:
        return stim.Circuit()

    def fake_run_single_calibration_point(
        *,
        case: CalibrationCase,
        model_type: str,
        sweep_param_name: str,
        sweep_value: Any,
        config: CalibrationConfig,
        base_params: Any = None,
        experiment_seed: int,
        base_circuit: Any = None,
    ) -> Dict[str, Any]:
        v = float(sweep_value)
        # mínimo de LER en v=0.20
        ler = abs(v - 0.20) + 0.001 * case.distance
        return {
            "case_name": case.case_name,
            "distance": case.distance,
            "rounds": case.rounds,
            "logical_basis": case.logical_basis,
            "model_type": model_type,
            "model_spec": {"type": model_type, sweep_param_name: v, "p": case.p},
            "sweep_param_name": sweep_param_name,
            "sweep_value": v,
            "shots": config.shots,
            "seed": int(experiment_seed),
            "ler": float(ler),
            "num_detectors": 10,
            "num_observables": 1,
            "decode_total_time_sec": 0.01,
            "avg_decode_time_sec": 0.001 + v,
            "status": "ok",
        }

    monkeypatch.setattr(nc, "build_base_xzzx_circuit", fake_build_base_xzzx_circuit)
    monkeypatch.setattr(nc, "run_single_calibration_point", fake_run_single_calibration_point)

    cases = [
        CalibrationCase(case_name="c1", distance=3, rounds=2, p=0.01),
        CalibrationCase(case_name="c2", distance=5, rounds=3, p=0.01),
    ]
    sweep = SweepSpec(
        model_type="depolarizing",
        param_name="p",
        values=(0.10, 0.20, 0.40),
        objective="min_ler",
    )
    cfg = CalibrationConfig(shots=50, seed=11, keep_soft_info_samples=0)

    report = run_noise_sweep(cases=cases, sweep=sweep, config=cfg)

    assert report["status"] == "ok"
    assert report["sweep"]["model_type"] == "depolarizing"
    assert report["sweep"]["param_name"] == "p"
    assert len(report["cases_summary"]) == 2
    assert len(report["aggregates"]["per_value_summary"]) == 3

    # Mejor valor global debe ser 0.20 (con nuestra función fake)
    assert report["aggregates"]["global_best"]["sweep_value"] == 0.20


def test_run_noise_sweep_contract_min_time(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.noise.noise_calibration as nc

    def fake_build_base_xzzx_circuit(*, distance: int, rounds: int, logical_basis: str) -> stim.Circuit:
        return stim.Circuit()

    def fake_run_single_calibration_point(
        *,
        case: CalibrationCase,
        model_type: str,
        sweep_param_name: str,
        sweep_value: Any,
        config: CalibrationConfig,
        base_params: Any = None,
        experiment_seed: int,
        base_circuit: Any = None,
    ) -> Dict[str, Any]:
        v = float(sweep_value)
        # tiempo mínimo en v=0.40
        avg_t = abs(v - 0.40) + 0.01
        return {
            "case_name": case.case_name,
            "distance": case.distance,
            "rounds": case.rounds,
            "logical_basis": case.logical_basis,
            "model_type": model_type,
            "model_spec": {"type": model_type, sweep_param_name: v, "p": case.p},
            "sweep_param_name": sweep_param_name,
            "sweep_value": v,
            "shots": config.shots,
            "seed": int(experiment_seed),
            "ler": 0.1 + v,
            "num_detectors": 10,
            "num_observables": 1,
            "decode_total_time_sec": avg_t * config.shots,
            "avg_decode_time_sec": avg_t,
            "status": "ok",
        }

    monkeypatch.setattr(nc, "build_base_xzzx_circuit", fake_build_base_xzzx_circuit)
    monkeypatch.setattr(nc, "run_single_calibration_point", fake_run_single_calibration_point)

    cases = [CalibrationCase(case_name="c1", distance=3, rounds=2, p=0.01)]
    sweep = SweepSpec(
        model_type="depolarizing",
        param_name="p",
        values=(0.10, 0.20, 0.40, 0.80),
        objective="min_time",
    )
    cfg = CalibrationConfig(shots=20, seed=22)

    report = run_noise_sweep(cases=cases, sweep=sweep, config=cfg)
    assert report["status"] == "ok"
    assert report["aggregates"]["global_best"]["sweep_value"] == 0.40


def test_run_noise_sweep_empty_cases_raises() -> None:
    sweep = SweepSpec(model_type="depolarizing", param_name="p", values=(0.01,))
    cfg = CalibrationConfig(shots=10, seed=1)
    with pytest.raises(ValueError):
        run_noise_sweep(cases=[], sweep=sweep, config=cfg)


# =============================================================================
# run_multi_model_calibration + IO
# =============================================================================
def test_run_multi_model_calibration_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.noise.noise_calibration as nc

    def fake_run_noise_sweep(*, cases: Any, sweep: SweepSpec, config: CalibrationConfig) -> Dict[str, Any]:
        return {
            "metadata": {"report_name": "noise_calibration_sweep", "timestamp_utc": "2026-02-16T00:00:00+00:00"},
            "sweep": {
                "model_type": sweep.model_type,
                "param_name": sweep.param_name,
                "values": list(sweep.values),
                "base_params": dict(sweep.base_params),
                "objective": sweep.objective,
            },
            "config": {"shots": config.shots, "seed": config.seed, "keep_soft_info_samples": config.keep_soft_info_samples},
            "cases_summary": [],
            "aggregates": {
                "num_cases": len(cases),
                "per_value_summary": [
                    {"sweep_value": float(sweep.values[0]), "num_points": len(cases), "mean_ler": 0.02, "mean_avg_decode_time_sec": 0.001}
                ],
                "global_best": {"sweep_value": float(sweep.values[0]), "num_points": len(cases), "mean_ler": 0.02, "mean_avg_decode_time_sec": 0.001},
                "mean_best_case_ler": 0.02,
                "mean_best_case_avg_decode_time_sec": 0.001,
            },
            "status": "ok",
        }

    monkeypatch.setattr(nc, "run_noise_sweep", fake_run_noise_sweep)

    cases = [CalibrationCase(case_name="c1", distance=3, rounds=2, p=0.01)]
    sweeps = [
        SweepSpec(model_type="depolarizing", param_name="p", values=(0.005, 0.01), objective="min_ler"),
        SweepSpec(model_type="biased", param_name="eta", values=(1.0, 2.0), objective="min_ler"),
    ]
    cfg = CalibrationConfig(shots=30, seed=99)

    out = run_multi_model_calibration(cases=cases, sweeps=sweeps, config=cfg)

    assert out["status"] == "ok"
    assert out["num_sweeps"] == 2
    assert len(out["sweeps_summary"]) == 2
    assert len(out["sweeps_reports"]) == 2
    assert out["sweeps_summary"][0]["model_type"] == "depolarizing"
    assert out["sweeps_summary"][1]["model_type"] == "biased"


def test_run_multi_model_calibration_empty_sweeps_raises() -> None:
    cases = [CalibrationCase(case_name="c1", distance=3, rounds=2, p=0.01)]
    cfg = CalibrationConfig(shots=10, seed=1)
    with pytest.raises(ValueError):
        run_multi_model_calibration(cases=cases, sweeps=[], config=cfg)


def test_save_and_load_calibration_report_roundtrip(tmp_path: Path) -> None:
    report = {
        "metadata": {"report_name": "noise_calibration_sweep"},
        "status": "ok",
        "value": 123,
    }
    out_path = tmp_path / "noise_calibration_report.json"

    saved = save_calibration_report(report, out_path)
    assert saved.exists()

    loaded = load_calibration_report(saved)
    assert loaded["status"] == "ok"
    assert loaded["value"] == 123
