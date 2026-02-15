# tests/test_union_find_decoder.py
from __future__ import annotations

import numpy as np
import pytest

# Si faltan dependencias, se salta el archivo completo.
pytest.importorskip("stim")
pytest.importorskip("pymatching")

from src.codes.xzzx_code import generate_xzzx_circuit
from src.decoders.union_find_decoder import (
    UFConfidenceConfig,
    UnionFindDecoderWithSoftInfo,
)


def _build_decoder(
    distance: int = 3,
    rounds: int = 3,
    p: float = 0.01,
    noise_model: str | dict = "depolarizing",
    prefer_union_find: bool = True,
) -> UnionFindDecoderWithSoftInfo:
    circuit = generate_xzzx_circuit(
        distance=distance,
        rounds=rounds,
        noise_model=noise_model,
        p=p,
        logical_basis="x",
    )
    return UnionFindDecoderWithSoftInfo(
        circuit=circuit,
        prefer_union_find=prefer_union_find,
    )


def test_decoder_construction_ok() -> None:
    decoder = _build_decoder()
    assert decoder.num_detectors > 0
    assert decoder.num_observables >= 1
    assert decoder.matcher is not None
    assert decoder.sampler is not None
    assert isinstance(decoder.matcher_build_mode, str)
    assert isinstance(decoder.decode_mode, str)


def test_decode_with_confidence_output_structure() -> None:
    decoder = _build_decoder(distance=3, rounds=2, p=0.01)
    syndrome = np.zeros(decoder.num_detectors, dtype=np.uint8)

    prediction, soft_info, decode_time = decoder.decode_with_confidence(syndrome)

    assert isinstance(prediction, np.ndarray)
    assert prediction.ndim == 1
    assert prediction.size >= 1

    assert isinstance(soft_info, dict)
    required_keys = {
        "syndrome_weight",
        "total_weight",
        "normalized_weight",
        "clustering_quality",
        "merge_count",
        "n_clusters",
        "syndrome_density",
        "merge_ratio",
        "confidence_score",
        "decode_time",
        "is_union_find_backend",
    }
    assert required_keys.issubset(soft_info.keys())

    assert isinstance(decode_time, float)
    assert decode_time >= 0.0


@pytest.mark.parametrize("seed", [1, 7, 42, 1234])
def test_soft_metrics_ranges(seed: int) -> None:
    rng = np.random.default_rng(seed)
    decoder = _build_decoder(distance=3, rounds=2, p=0.02)

    syndrome = rng.integers(0, 2, size=decoder.num_detectors, dtype=np.uint8)
    _, soft_info, decode_time = decoder.decode_with_confidence(syndrome)

    conf = soft_info["confidence_score"]
    nw = soft_info["normalized_weight"]
    cq = soft_info["clustering_quality"]
    mr = soft_info["merge_ratio"]
    sd = soft_info["syndrome_density"]
    uf_flag = soft_info["is_union_find_backend"]

    assert 0.0 <= conf <= 1.0
    assert nw >= 0.0
    assert 0.0 <= cq <= 1.0
    assert 0.0 <= mr <= 1.0
    assert 0.0 <= sd <= 1.0
    assert uf_flag in (0.0, 1.0)
    assert soft_info["decode_time"] >= 0.0
    assert decode_time >= 0.0


def test_invalid_syndrome_shape_raises() -> None:
    decoder = _build_decoder()

    bad_2d = np.zeros((1, decoder.num_detectors), dtype=np.uint8)
    with pytest.raises(ValueError):
        decoder.decode_with_confidence(bad_2d)

    bad_len = np.zeros(decoder.num_detectors + 1, dtype=np.uint8)
    with pytest.raises(ValueError):
        decoder.decode_with_confidence(bad_len)


def test_invalid_circuit_type_raises() -> None:
    with pytest.raises(TypeError):
        UnionFindDecoderWithSoftInfo(circuit="not_a_stim_circuit")  # type: ignore[arg-type]


def test_custom_confidence_config_is_accepted() -> None:
    circuit = generate_xzzx_circuit(
        distance=3,
        rounds=2,
        noise_model="depolarizing",
        p=0.01,
        logical_basis="x",
    )
    cfg = UFConfidenceConfig(
        a_cluster=2.0,
        b_norm_weight=1.0,
        c_density=0.7,
        d_merge=0.6,
        e_backend=0.3,
        bias=0.1,
    )
    decoder = UnionFindDecoderWithSoftInfo(circuit, confidence_config=cfg)

    syndrome = np.zeros(decoder.num_detectors, dtype=np.uint8)
    _, soft_info, _ = decoder.decode_with_confidence(syndrome)

    assert 0.0 <= soft_info["confidence_score"] <= 1.0


def test_benchmark_output_contract() -> None:
    decoder = _build_decoder(distance=3, rounds=2, p=0.02)

    res = decoder.benchmark(shots=40, keep_soft_info_samples=10)

    assert isinstance(res, dict)
    expected_top = {
        "shots",
        "num_detectors",
        "num_observables",
        "error_rate",
        "avg_decode_time",
        "soft_info_samples",
        "backend_info",
    }
    assert expected_top.issubset(res.keys())

    assert res["shots"] == 40
    assert isinstance(res["num_detectors"], int) and res["num_detectors"] > 0
    assert isinstance(res["num_observables"], int) and res["num_observables"] >= 1

    assert isinstance(res["error_rate"], float)
    assert (0.0 <= res["error_rate"] <= 1.0) or np.isnan(res["error_rate"])

    assert isinstance(res["avg_decode_time"], float)
    assert res["avg_decode_time"] >= 0.0

    assert isinstance(res["soft_info_samples"], list)
    assert len(res["soft_info_samples"]) <= 10
    if len(res["soft_info_samples"]) > 0:
        s0 = res["soft_info_samples"][0]
        assert "confidence_score" in s0
        assert 0.0 <= s0["confidence_score"] <= 1.0

    backend = res["backend_info"]
    assert isinstance(backend, dict)
    assert "matcher_build_mode" in backend
    assert "decode_mode" in backend
    assert "is_union_find_backend" in backend
    assert isinstance(backend["is_union_find_backend"], bool)


def test_benchmark_invalid_shots_raises() -> None:
    decoder = _build_decoder()

    with pytest.raises(ValueError):
        decoder.benchmark(shots=0)

    with pytest.raises(ValueError):
        decoder.benchmark(shots=-5)


def test_prefer_union_find_false_still_runs() -> None:
    decoder = _build_decoder(prefer_union_find=False)
    syndrome = np.zeros(decoder.num_detectors, dtype=np.uint8)

    pred, soft_info, dt = decoder.decode_with_confidence(syndrome)
    assert isinstance(pred, np.ndarray)
    assert pred.size >= 1
    assert 0.0 <= soft_info["confidence_score"] <= 1.0
    assert dt >= 0.0
