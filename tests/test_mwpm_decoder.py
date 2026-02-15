# tests/test_mwpm_decoder.py
from __future__ import annotations

import numpy as np
import pytest

# Si faltan dependencias, se salta el archivo completo.
pytest.importorskip("stim")
pytest.importorskip("pymatching")

from src.codes.xzzx_code import generate_xzzx_circuit
from src.decoders.mwpm_decoder import ConfidenceConfig, MWPMDecoderWithSoftInfo


def _build_decoder(distance: int = 3, rounds: int = 3, p: float = 0.01) -> MWPMDecoderWithSoftInfo:
    """
    Helper para construir un decoder listo para tests.
    """
    circuit = generate_xzzx_circuit(
        distance=distance,
        rounds=rounds,
        noise_model="depolarizing",
        p=p,
        logical_basis="x",
    )
    return MWPMDecoderWithSoftInfo(circuit)


def test_decoder_construction_ok() -> None:
    """
    Verifica que el decoder se construye correctamente y expone metadatos básicos.
    """
    decoder = _build_decoder()
    assert decoder.num_detectors > 0
    assert decoder.num_observables >= 1
    assert decoder.matcher is not None
    assert decoder.sampler is not None


def test_decode_with_confidence_output_structure() -> None:
    """
    decode_with_confidence debe devolver:
    - prediction
    - soft_info con claves requeridas
    - decode_time
    """
    decoder = _build_decoder()
    syndrome = np.zeros(decoder.num_detectors, dtype=np.uint8)

    prediction, soft_info, decode_time = decoder.decode_with_confidence(syndrome)

    assert isinstance(prediction, np.ndarray)
    assert prediction.ndim == 1
    assert prediction.size >= 1  # al menos un observable lógico esperado

    assert isinstance(soft_info, dict)
    required_keys = {
        "syndrome_weight",
        "total_weight",
        "weight_gap",
        "normalized_weight",
        "confidence_score",
        "decode_time",
    }
    assert required_keys.issubset(set(soft_info.keys()))

    assert isinstance(decode_time, float)
    assert decode_time >= 0.0


@pytest.mark.parametrize("seed", [1, 7, 42, 1234])
def test_confidence_and_metrics_ranges(seed: int) -> None:
    """
    Rango de métricas soft-info:
    - confidence_score en [0,1]
    - normalized_weight >= 0
    - weight_gap en [0,1]
    - decode_time >= 0
    """
    rng = np.random.default_rng(seed)
    decoder = _build_decoder(distance=3, rounds=2, p=0.02)

    syndrome = rng.integers(0, 2, size=decoder.num_detectors, dtype=np.uint8)
    _, soft_info, decode_time = decoder.decode_with_confidence(syndrome)

    conf = soft_info["confidence_score"]
    nw = soft_info["normalized_weight"]
    wg = soft_info["weight_gap"]

    assert 0.0 <= conf <= 1.0
    assert nw >= 0.0
    assert 0.0 <= wg <= 1.0
    assert soft_info["decode_time"] >= 0.0
    assert decode_time >= 0.0


def test_invalid_syndrome_shape_raises() -> None:
    """
    Síndrome con shape incorrecto debe fallar.
    """
    decoder = _build_decoder()

    # 2D en lugar de 1D
    bad = np.zeros((1, decoder.num_detectors), dtype=np.uint8)
    with pytest.raises(ValueError):
        decoder.decode_with_confidence(bad)

    # longitud incorrecta
    bad_len = np.zeros(decoder.num_detectors + 1, dtype=np.uint8)
    with pytest.raises(ValueError):
        decoder.decode_with_confidence(bad_len)


def test_invalid_circuit_type_raises() -> None:
    """
    Constructor debe validar tipo de circuito.
    """
    with pytest.raises(TypeError):
        MWPMDecoderWithSoftInfo(circuit="not_a_stim_circuit")  # type: ignore[arg-type]


def test_custom_confidence_config_is_accepted() -> None:
    """
    Debe aceptar una config custom de confianza.
    """
    circuit = generate_xzzx_circuit(
        distance=3,
        rounds=2,
        noise_model="depolarizing",
        p=0.01,
        logical_basis="x",
    )
    cfg = ConfidenceConfig(a=1.8, b=0.9, c=0.4, bias=0.1)
    decoder = MWPMDecoderWithSoftInfo(circuit, confidence_config=cfg)

    syndrome = np.zeros(decoder.num_detectors, dtype=np.uint8)
    _, soft_info, _ = decoder.decode_with_confidence(syndrome)

    assert 0.0 <= soft_info["confidence_score"] <= 1.0


def test_benchmark_output_contract() -> None:
    """
    benchmark debe devolver las claves principales y tipos esperados.
    """
    decoder = _build_decoder(distance=3, rounds=2, p=0.02)

    res = decoder.benchmark(shots=40, keep_soft_info_samples=10)

    assert isinstance(res, dict)
    expected = {
        "shots",
        "num_detectors",
        "num_observables",
        "error_rate",
        "avg_decode_time",
        "soft_info_samples",
    }
    assert expected.issubset(res.keys())

    assert res["shots"] == 40
    assert isinstance(res["num_detectors"], int) and res["num_detectors"] > 0
    assert isinstance(res["num_observables"], int) and res["num_observables"] >= 1

    # error_rate suele estar en [0,1] (nan solo en edge-cases sin observables)
    assert isinstance(res["error_rate"], float)
    assert (0.0 <= res["error_rate"] <= 1.0) or np.isnan(res["error_rate"])

    assert isinstance(res["avg_decode_time"], float)
    assert res["avg_decode_time"] >= 0.0

    assert isinstance(res["soft_info_samples"], list)
    assert len(res["soft_info_samples"]) <= 10
    if len(res["soft_info_samples"]) > 0:
        sample0 = res["soft_info_samples"][0]
        assert "confidence_score" in sample0
        assert 0.0 <= sample0["confidence_score"] <= 1.0


def test_benchmark_invalid_shots_raises() -> None:
    """
    shots inválido debe fallar.
    """
    decoder = _build_decoder()

    with pytest.raises(ValueError):
        decoder.benchmark(shots=0)

    with pytest.raises(ValueError):
        decoder.benchmark(shots=-5)
