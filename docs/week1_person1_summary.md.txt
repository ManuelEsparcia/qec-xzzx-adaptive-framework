# Week 1 — Person 1 Summary (Q1_QEC)

## Scope completed

Durante la Semana 1 (Persona 1) se completaron los siguientes bloques:

1. **XZZX code base**
   - Implementación de `src/codes/xzzx_code.py`
   - Generación de circuitos (distancias impares, rondas configurables)
   - Validaciones de entrada
   - Soporte de ruido base y helpers MWPM

2. **MWPM + soft information**
   - Implementación de `src/decoders/mwpm_decoder.py`
   - Clase `MWPMDecoderWithSoftInfo`
   - Método `decode_with_confidence(...)`
   - Métricas soft-info:
     - `syndrome_weight`
     - `total_weight`
     - `weight_gap`
     - `normalized_weight`
     - `confidence_score`
     - `decode_time`
   - Método `benchmark(...)`

3. **Integración E2E**
   - Pipeline completo en `src/pipelines/week1_person1_pipeline.py`
   - Test de integración en `tests/test_week1_person1_integration.py`
   - Guardado de resultados JSON

4. **Benchmark base (evidencia Semana 1)**
   - Script: `scripts/run_week1_person1_benchmark.py`
   - Salida: `results/week1_person1_baseline.json`

---

## Validation status

### Unit tests previos
- `tests/test_xzzx_code.py`: **6 passed**
- `tests/test_mwpm_decoder.py`: **11 passed**
- Suite conjunta: **17 passed**

### Integración E2E
- `tests/test_week1_person1_integration.py`: **6 passed**

> Total validaciones ejecutadas con éxito en esta fase: **23 tests passed**.

---

## Baseline benchmark results (300 shots/case)

| Case | d | rounds | p | benchmark_error_rate | reference_ler_helper | abs_delta | avg_decode_time_sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| case_d3_r2_p0.005 | 3 | 2 | 0.005 | 0.003333 | 0.000000 | 0.003333 | 0.000015876 |
| case_d3_r3_p0.01  | 3 | 3 | 0.010 | 0.020000 | 0.016667 | 0.003333 | 0.000013727 |
| case_d5_r3_p0.01  | 5 | 3 | 0.010 | 0.006667 | 0.010000 | 0.003333 | 0.000020297 |

### Aggregates
- mean_benchmark_error_rate: **0.010000**
- mean_reference_ler_helper: **0.008889**
- mean_abs_delta_benchmark_vs_reference: **0.003333**
- mean_avg_decode_time_sec: **0.000016633**

---

## Reproducibility commands

```bash
py -3.10 -m pytest tests/test_xzzx_code.py tests/test_mwpm_decoder.py tests/test_week1_person1_integration.py -v
py -3.10 -m src.pipelines.week1_person1_pipeline
py -3.10 scripts/run_week1_person1_benchmark.py
