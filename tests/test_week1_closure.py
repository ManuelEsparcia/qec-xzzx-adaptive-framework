from __future__ import annotations

import numpy as np
import pytest

stim = pytest.importorskip("stim")
pytest.importorskip("pymatching")

from src.codes.xzzx_code import generate_xzzx_circuit
from src.decoders.mwpm_decoder import MWPMDecoderWithSoftInfo
from src.noise.noise_models import apply_noise_model, build_noise_model


def test_week1_closure_smoke() -> None:
    """
    Binary Week 1 closure smoke test:
    - XZZX generation for d=3,5,7,9
    - 5 noise models build/apply
    - MWPM decode+benchmark with soft-info contract
    """
    distances = [3, 5, 7, 9]
    for d in distances:
        rounds = 2 if d <= 5 else 1
        circuit = generate_xzzx_circuit(
            distance=d,
            rounds=rounds,
            noise_model="none",
            p=0.0,
            logical_basis="x",
        )
        assert isinstance(circuit, stim.Circuit)
        assert int(circuit.num_detectors) > 0
        assert int(circuit.num_observables) >= 1

    base = generate_xzzx_circuit(
        distance=3,
        rounds=2,
        noise_model="none",
        p=0.0,
        logical_basis="x",
    )

    noise_specs = [
        {"type": "depolarizing", "p": 0.01},
        {"type": "biased", "p": 0.01, "eta": 100.0},
        {"type": "circuit_level", "p": 0.01},
        {"type": "phenomenological", "p": 0.01},
        {
            "type": "correlated",
            "p": 0.01,
            "correlation_length": 2,
            "correlation_strength": 0.5,
            "topology": "line",
            "pauli": "Z",
        },
    ]

    for spec in noise_specs:
        model = build_noise_model(spec)
        noisy = model.apply_to_circuit(base)
        assert isinstance(noisy, stim.Circuit)

    noisy_for_decode = apply_noise_model(base, model={"type": "depolarizing", "p": 0.005})
    decoder = MWPMDecoderWithSoftInfo(noisy_for_decode)

    syndrome = np.zeros(decoder.num_detectors, dtype=np.uint8)
    prediction, soft_info, decode_time = decoder.decode_with_confidence(syndrome)
    assert prediction.ndim == 1
    assert isinstance(decode_time, float) and decode_time >= 0.0

    expected_soft_keys = {
        "syndrome_weight",
        "total_weight",
        "weight_gap",
        "normalized_weight",
        "confidence_score",
        "decode_time",
    }
    assert expected_soft_keys.issubset(soft_info.keys())
    assert 0.0 <= float(soft_info["confidence_score"]) <= 1.0

    bench = decoder.benchmark(shots=20, keep_soft_info_samples=4)
    assert bench["shots"] == 20
    assert int(bench["num_detectors"]) > 0
    assert int(bench["num_observables"]) >= 1
    assert isinstance(bench["soft_info_samples"], list)
    assert len(bench["soft_info_samples"]) <= 4
