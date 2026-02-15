# src/pipelines/week1_person1_pipeline.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.codes.xzzx_code import (
    circuit_summary,
    generate_xzzx_circuit,
    logical_error_rate_mwpm,
)
from src.decoders.mwpm_decoder import MWPMDecoderWithSoftInfo


@dataclass(frozen=True)
class Week1Person1PipelineConfig:
    """
    Configuración del pipeline de integración (Semana 1, Persona 1).
    """
    distance: int = 3
    rounds: int = 3
    noise_model: Any = "depolarizing"  # str | dict | callable | objeto con apply_to_circuit
    p: float = 0.01
    logical_basis: str = "x"
    shots: int = 300
    keep_soft_info_samples: Optional[int] = 100
    reference_ler_shots: int = 300


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_week1_person1_pipeline(
    config: Week1Person1PipelineConfig,
) -> Dict[str, Any]:
    """
    Ejecuta pipeline completo de integración:
      1) Genera circuito XZZX
      2) Construye decoder MWPM con soft-info
      3) Ejecuta benchmark
      4) Calcula LER de referencia con helper de xzzx_code

    Returns
    -------
    dict con:
      - metadata
      - config
      - circuit_summary
      - benchmark
      - reference
      - status
    """
    # 1) Circuito
    circuit = generate_xzzx_circuit(
        distance=config.distance,
        rounds=config.rounds,
        noise_model=config.noise_model,
        p=config.p,
        logical_basis=config.logical_basis,
    )

    # 2) Summary rápido del circuito
    c_summary = circuit_summary(circuit)

    # 3) Decoder + benchmark integrado
    decoder = MWPMDecoderWithSoftInfo(circuit)
    bench = decoder.benchmark(
        shots=config.shots,
        keep_soft_info_samples=config.keep_soft_info_samples,
    )

    # 4) LER referencia por helper directo (consistencia cruzada)
    ref_shots = int(max(1, config.reference_ler_shots))
    ler_ref = float(logical_error_rate_mwpm(circuit, shots=ref_shots))

    result: Dict[str, Any] = {
        "metadata": {
            "pipeline": "week1_person1_pipeline",
            "timestamp_utc": _utc_now_iso(),
        },
        "config": asdict(config),
        "circuit_summary": c_summary,
        "benchmark": bench,
        "reference": {
            "ler_mwpm_helper": ler_ref,
            "reference_ler_shots": ref_shots,
        },
        "status": "ok",
    }
    return result


def save_pipeline_result(result: Dict[str, Any], output_path: str | Path) -> Path:
    """
    Guarda resultado del pipeline en JSON.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return path


def run_and_save_week1_person1_pipeline(
    config: Week1Person1PipelineConfig,
    output_path: str | Path,
) -> Dict[str, Any]:
    """
    Conveniencia: ejecuta pipeline y guarda JSON.
    """
    result = run_week1_person1_pipeline(config)
    save_pipeline_result(result, output_path)
    return result


if __name__ == "__main__":
    # Ejecución rápida manual:
    # py -3.10 src/pipelines/week1_person1_pipeline.py
    cfg = Week1Person1PipelineConfig(
        distance=3,
        rounds=3,
        noise_model="depolarizing",
        p=0.01,
        logical_basis="x",
        shots=200,
        keep_soft_info_samples=50,
        reference_ler_shots=200,
    )
    out = run_week1_person1_pipeline(cfg)
    print(json.dumps(out, ensure_ascii=False, indent=2))
