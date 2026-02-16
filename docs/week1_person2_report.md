# Week 1 — Person 2 Final Report (Noise Modeling)

## 1) Objetivo

Implementar y validar el bloque de modelado de ruido para el framework QEC-XZZX:
- `src/noise/noise_models.py`
- `tests/test_noise_models.py`
- integración E2E con circuito + decoder
- benchmark base para evidencia cuantitativa de Semana 1

---

## 2) Estado de implementación

### 2.1 Modelos de ruido implementados
- Depolarizing
- Biased (sesgado)
- Circuit-level
- Phenomenological
- Correlated

### 2.2 Test unitarios
Ejecución:
```bash
py -3.10 -m pytest tests/test_noise_models.py -v
