# qec-xzzx-adaptive-framework
Framework de investigacion para evaluar decoders de QEC sobre codigos tipo XZZX usando `stim` + `pymatching`.

## Objetivo
Construir y comparar una pipeline reproducible para:
- Generacion de circuitos tipo memory (rotated surface templates de Stim)
- Inyeccion de modelos de ruido
- Decodificacion con MWPM, Union-Find y switching adaptativo
- Benchmarking y reporte de resultados

## Estado actual
- Generacion de circuito XZZX: implementada
- Modelos de ruido (5): implementados
- Decoders:
  - MWPM con soft-info: implementado
  - Union-Find con soft-info: implementado
  - Adaptive (switch UF->MWPM por umbral): implementado
- Suite de tests: pasando (`74 passed` en entorno local conda `qec-xzzx`)

## Estructura del repositorio
```text
src/
  codes/        # generacion de circuitos y helpers base
  noise/        # modelos de ruido
  decoders/     # MWPM / UF
  switching/    # decoder adaptativo
  pipelines/    # pipeline end-to-end
scripts/        # benchmarks y ejecuciones reproducibles
tests/          # unit + integracion + smoke tests
results/        # salidas JSON de benchmarks
docs/           # reportes de avance
```

## Requisitos
- Python 3.11 recomendado
- Dependencias en `requirements.txt`

### Dependencias principales
- `numpy`
- `stim`
- `pymatching`
- `pytest`

## Instalacion rapida (Conda)
```bash
conda create -n qec-xzzx python=3.11 -y
conda activate qec-xzzx
python -m pip install -r requirements.txt
```

## Ejecutar tests
```bash
python -m pytest -q
```

## Ejecucion de scripts principales
### Semana 1 - baseline
```bash
python -m scripts.run_week1_person1_benchmark
```

### Semana 1 - benchmark de ruido
```bash
python -m scripts.run_week1_person2_noise_benchmark --shots 400
```

### Semana 2 - comparacion de decoders
```bash
python -m scripts.run_week2_person1_decoder_comparison --shots 300 --g-threshold 0.65
```

### Semana 2 - barrido de threshold (paired)
```bash
python -m scripts.run_week2_person1_paired_threshold_sweep --shots 2000 --thresholds 0.20,0.35,0.40,0.60,0.80
```

## Resultados
Los JSON de salida se guardan en `results/`.

## Notas
- El proyecto usa imports tipo `from src...`; ejecutar desde la raiz del repositorio.
- Para VS Code se puede fijar el interprete en `.vscode/settings.json`.
