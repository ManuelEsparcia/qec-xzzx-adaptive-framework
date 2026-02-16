# Week 2 - Person 2 Final Report (Noise Calibration)

## 1. Objetivo
Calibrar parámetros de modelos de ruido para XZZX y seleccionar valores óptimos bajo criterio `min_ler`.

## 2. Entregables implementados
- `src/noise/noise_calibration.py`
- `scripts/run_week2_person2_noise_calibration.py`
- `tests/test_noise_calibration.py`
- `tests/test_week2_person2_noise_calibration_smoke.py`

## 3. Validación técnica
- `py -3.10 -m pytest tests/test_noise_calibration.py -v` ✅
- `py -3.10 -m pytest tests/test_week2_person2_noise_calibration_smoke.py -v` ✅
- Suite completa: `py -3.10 -m pytest tests -v` ✅

## 4. Experimentos ejecutados
### Quick run
- Output: `results/week2_person2_noise_calibration_quick.json`
- Config: shots=80, seed=12345, models=depolarizing,biased, fast=true

### Full run
- Output: `results/week2_person2_noise_calibration.json`
- Config: shots=300, seed=2026, models=depolarizing,biased,circuit_level,phenomenological,correlated

## 5. Mejores parámetros (full run)
- depolarizing: p=0.0025
- biased: eta=1.5
- circuit_level: p=0.0025
- phenomenological: p=0.001
- correlated: p=0.001

## 6. Observaciones
- Flujo de calibración multi-modelo estable y reproducible con seed fija.
- Contratos JSON correctos para integración futura en pipeline adaptativo.

## 7. Estado
Semana 2 Persona 2 completada y lista para integración con siguientes fases.
