# tests/test_adaptive_decoder.py
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pytest

# Dependencias requeridas by los decoders reales
pytest.importorskip("stim")
pytest.importorskip("pymatching")

from src.codes.xzzx_code import generate_xzzx_circuit
from src.switching.adaptive_decoder import AdaptiveConfig, AdaptiveDecoder


class _DummyDecoder:
    """
    Decoder mínimo for tests deterministas de lógica de switching.
    """

    def __init__(self, pred_bit: int, confidence: float, decode_time: float = 1e-6) -> None:
        self.pred_bit = int(pred_bit) & 1
        self.confidence = float(confidence)
        self.decode_time = float(decode_time)

    def decode_with_confidence(
        self, syndrome: Sequence[int] | np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float], float]:
        pred = np.array([self.pred_bit], dtype=np.uint8)
        soft = {
            "confidence_score": self.confidence,
            "dummy_metric": 1.0,
        }
        return pred, soft, self.decode_time


class _BadDecoder:
    """Not implementa decode_with_confidence -> must fail when constructing AdaptiveDecoder."""
    pass


def _build_circuit(distance: int = 3, rounds: int = 2, p: float = 0.01):
    return generate_xzzx_circuit(
        distance=distance,
        rounds=rounds,
        noise_model="depolarizing",
        p=p,
        logical_basis="x",
    )


def test_construction_with_defaults_ok() -> None:
    circuit = _build_circuit(distance=3, rounds=2, p=0.01)
    ad = AdaptiveDecoder(circuit)

    assert ad.num_detectors > 0
    assert ad.num_observables >= 1
    assert ad.fast_decoder is not None
    assert ad.accurate_decoder is not None
    assert 0.0 <= ad.config.g_threshold <= 1.0


def test_invalid_circuit_type_raises() -> None:
    with pytest.raises(TypeError):
        AdaptiveDecoder(circuit="not_a_stim_circuit")  # type: ignore[arg-type]


def test_invalid_threshold_in_config_raises() -> None:
    circuit = _build_circuit()

    bad_cfg = AdaptiveConfig(g_threshold=1.2)
    with pytest.raises(ValueError):
        AdaptiveDecoder(circuit, config=bad_cfg)


def test_invalid_decoder_interface_raises() -> None:
    circuit = _build_circuit()

    with pytest.raises(TypeError):
        AdaptiveDecoder(circuit, fast_decoder=_BadDecoder())  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        AdaptiveDecoder(circuit, accurate_decoder=_BadDecoder())  # type: ignore[arg-type]


def test_set_threshold_updates_value() -> None:
    circuit = _build_circuit()
    ad = AdaptiveDecoder(circuit, config=AdaptiveConfig(g_threshold=0.5))

    ad.set_threshold(0.9)
    assert ad.config.g_threshold == 0.9

    with pytest.raises(ValueError):
        ad.set_threshold(-0.1)


def test_decode_adaptive_output_contract_real_decoders() -> None:
    circuit = _build_circuit(distance=3, rounds=2, p=0.01)
    ad = AdaptiveDecoder(circuit, config=AdaptiveConfig(g_threshold=0.65))

    syndrome = np.zeros(ad.num_detectors, dtype=np.uint8)
    pred, info, dt = ad.decode_adaptive(syndrome)

    assert isinstance(pred, np.ndarray)
    assert pred.ndim == 1
    assert pred.size >= 1

    assert isinstance(info, dict)
    required = {
        "selected_decoder",
        "switched",
        "g_threshold",
        "fast_confidence_score",
        "fast_decode_time",
        "accurate_decode_time",
        "total_decode_time",
        "final_confidence_score",
        "final_soft_info",
        "fast_soft_info",
    }
    assert required.issubset(info.keys())

    assert info["selected_decoder"] in {"uf", "mwpm"}
    assert isinstance(info["switched"], bool)
    assert 0.0 <= float(info["g_threshold"]) <= 1.0
    assert 0.0 <= float(info["fast_confidence_score"]) <= 1.0
    assert dt >= 0.0
    assert float(info["total_decode_time"]) >= 0.0


def test_decode_adaptive_switches_when_confidence_below_threshold() -> None:
    circuit = _build_circuit(distance=3, rounds=2, p=0.01)

    fast = _DummyDecoder(pred_bit=0, confidence=0.2)
    accurate = _DummyDecoder(pred_bit=1, confidence=0.95)

    ad = AdaptiveDecoder(
        circuit,
        fast_decoder=fast,
        accurate_decoder=accurate,
        config=AdaptiveConfig(g_threshold=0.5),
    )

    syndrome = np.zeros(ad.num_detectors, dtype=np.uint8)
    pred, info, _ = ad.decode_adaptive(syndrome)

    assert info["switched"] is True
    assert info["selected_decoder"] == "mwpm"
    # Must use la predicción del decoder preciso (accurate)
    assert pred[0] == 1


def test_decode_adaptive_keeps_fast_when_confidence_above_threshold() -> None:
    circuit = _build_circuit(distance=3, rounds=2, p=0.01)

    fast = _DummyDecoder(pred_bit=0, confidence=0.8)
    accurate = _DummyDecoder(pred_bit=1, confidence=0.95)

    ad = AdaptiveDecoder(
        circuit,
        fast_decoder=fast,
        accurate_decoder=accurate,
        config=AdaptiveConfig(g_threshold=0.5),
    )

    syndrome = np.zeros(ad.num_detectors, dtype=np.uint8)
    pred, info, _ = ad.decode_adaptive(syndrome)

    assert info["switched"] is False
    assert info["selected_decoder"] == "uf"
    # Must use the fast decoder prediction
    assert pred[0] == 0


def test_benchmark_adaptive_output_contract_with_reference() -> None:
    circuit = _build_circuit(distance=3, rounds=2, p=0.01)
    ad = AdaptiveDecoder(
        circuit,
        config=AdaptiveConfig(g_threshold=0.65, compare_against_mwpm_in_benchmark=True),
    )

    res = ad.benchmark_adaptive(
        shots=30,
        g_threshold=0.65,
        keep_samples=10,
        compare_against_mwpm=True,
    )

    assert isinstance(res, dict)
    expected = {
        "shots",
        "g_threshold",
        "num_detectors",
        "num_observables",
        "error_rate_adaptive",
        "avg_decode_time_adaptive",
        "switch_rate",
        "samples",
        "status",
        "reference_mwpm",
        "speedup_vs_mwpm",
    }
    assert expected.issubset(res.keys())

    assert res["shots"] == 30
    assert res["status"] == "ok"

    er = float(res["error_rate_adaptive"])
    assert (0.0 <= er <= 1.0) or np.isnan(er)

    assert float(res["avg_decode_time_adaptive"]) >= 0.0

    sr = float(res["switch_rate"])
    assert 0.0 <= sr <= 1.0

    assert isinstance(res["samples"], list)
    assert len(res["samples"]) <= 10
    if len(res["samples"]) > 0:
        s0 = res["samples"][0]
        assert "selected_decoder" in s0
        assert s0["selected_decoder"] in {"uf", "mwpm"}

    ref = res["reference_mwpm"]
    assert "error_rate_mwpm" in ref
    assert "avg_decode_time_mwpm" in ref
    assert isinstance(ref["avg_decode_time_mwpm"], float)


def test_benchmark_adaptive_without_reference_key() -> None:
    circuit = _build_circuit(distance=3, rounds=2, p=0.01)
    ad = AdaptiveDecoder(
        circuit,
        config=AdaptiveConfig(g_threshold=0.65, compare_against_mwpm_in_benchmark=False),
    )

    res = ad.benchmark_adaptive(
        shots=20,
        compare_against_mwpm=False,
    )

    assert "reference_mwpm" not in res
    assert "speedup_vs_mwpm" not in res
    assert res["fast_mode"] is False
    assert 0.0 <= float(res["switch_rate"]) <= 1.0


def test_benchmark_threshold_controls_switch_rate_deterministically() -> None:
    """
    Deterministic switch_rate test using dummy decoders.
    """
    circuit = _build_circuit(distance=3, rounds=2, p=0.01)

    # Fast decoder always with confidence 0.4
    fast = _DummyDecoder(pred_bit=0, confidence=0.4, decode_time=1e-6)
    accurate = _DummyDecoder(pred_bit=1, confidence=0.9, decode_time=2e-6)

    ad = AdaptiveDecoder(
        circuit,
        fast_decoder=fast,
        accurate_decoder=accurate,
        config=AdaptiveConfig(g_threshold=0.5, compare_against_mwpm_in_benchmark=False),
    )

    # threshold menor que 0.4 -> nunca switch
    res_low = ad.benchmark_adaptive(shots=25, g_threshold=0.3, compare_against_mwpm=False)
    # threshold mayor que 0.4 -> siempre switch
    res_high = ad.benchmark_adaptive(shots=25, g_threshold=0.8, compare_against_mwpm=False)

    assert float(res_low["switch_rate"]) == 0.0
    assert float(res_high["switch_rate"]) == 1.0

    # Same behavior in fast mode.
    res_low_fast = ad.benchmark_adaptive(
        shots=25,
        g_threshold=0.3,
        compare_against_mwpm=False,
        fast_mode=True,
    )
    res_high_fast = ad.benchmark_adaptive(
        shots=25,
        g_threshold=0.8,
        compare_against_mwpm=False,
        fast_mode=True,
    )
    assert res_low_fast["fast_mode"] is True
    assert res_high_fast["fast_mode"] is True
    assert float(res_low_fast["switch_rate"]) == 0.0
    assert float(res_high_fast["switch_rate"]) == 1.0


def test_benchmark_adaptive_fast_mode_sample_contract() -> None:
    circuit = _build_circuit(distance=3, rounds=2, p=0.01)
    ad = AdaptiveDecoder(
        circuit,
        config=AdaptiveConfig(g_threshold=0.65, compare_against_mwpm_in_benchmark=False),
    )

    res = ad.benchmark_adaptive(
        shots=20,
        keep_samples=5,
        compare_against_mwpm=False,
        fast_mode=True,
    )

    assert res["fast_mode"] is True
    assert "reference_mwpm" not in res
    assert "speedup_vs_mwpm" not in res
    assert isinstance(res["samples"], list)
    assert len(res["samples"]) <= 5
    if res["samples"]:
        s0 = res["samples"][0]
        assert "selected_decoder" in s0
        assert "switched" in s0
        assert "fast_confidence_score" in s0
        assert "total_decode_time" in s0


def test_benchmark_invalid_shots_raises() -> None:
    circuit = _build_circuit(distance=3, rounds=2, p=0.01)
    ad = AdaptiveDecoder(circuit)

    with pytest.raises(ValueError):
        ad.benchmark_adaptive(shots=0)

    with pytest.raises(ValueError):
        ad.benchmark_adaptive(shots=-10)
