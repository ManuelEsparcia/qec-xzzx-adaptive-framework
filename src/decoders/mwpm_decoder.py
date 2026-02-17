# src/decoders/mwpm_decoder.py
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import stim
except Exception as exc:  # pragma: no cover
    stim = None  # type: ignore[assignment]
    _STIM_IMPORT_ERROR = exc
else:
    _STIM_IMPORT_ERROR = None

try:
    import pymatching
except Exception as exc:  # pragma: no cover
    pymatching = None  # type: ignore[assignment]
    _PYMATCHING_IMPORT_ERROR = exc
else:
    _PYMATCHING_IMPORT_ERROR = None


@dataclass(frozen=True)
class ConfidenceConfig:
    """
    Confidence function configuration (heuristic, Week 1).
    confidence = sigmoid(a*weight_gap - b*normalized_weight - c*syndrome_weight_norm + bias)
    """

    a: float = 2.0
    b: float = 1.0
    c: float = 0.5
    bias: float = 0.0
    eps: float = 1e-12


class MWPMDecoderWithSoftInfo:
    """
    MWPM decoder with soft information for the adaptive framework.

    Notes:
    - `weight_gap` is heuristic in this version (Week 1).
    - Designed to be stable and quick to integrate.
    """

    def __init__(
        self,
        circuit: "stim.Circuit",
        confidence_config: Optional[ConfidenceConfig] = None,
    ) -> None:
        self._require_stim()
        self._require_pymatching()

        if not isinstance(circuit, stim.Circuit):
            raise TypeError(
                f"circuit must be stim.Circuit, received: {type(circuit).__name__}"
            )

        self.circuit = circuit
        self.conf_cfg = confidence_config or ConfidenceConfig()

        self.dem = self.circuit.detector_error_model(decompose_errors=True)
        self.matcher = pymatching.Matching.from_detector_error_model(self.dem)
        self.sampler = self.circuit.compile_detector_sampler()

        self.num_detectors = int(getattr(self.circuit, "num_detectors", 0))
        self.num_observables = int(getattr(self.circuit, "num_observables", 0))

        if self.num_detectors <= 0:
            raise ValueError("Circuit has no detectors (num_detectors <= 0).")

    @staticmethod
    def _require_stim() -> None:
        if stim is None:
            raise ImportError(
                "stim is not installed or failed to import. "
                "Install with: pip install stim"
            ) from _STIM_IMPORT_ERROR

    @staticmethod
    def _require_pymatching() -> None:
        if pymatching is None:
            raise ImportError(
                "pymatching is not installed or failed to import. "
                "Install with: pip install pymatching"
            ) from _PYMATCHING_IMPORT_ERROR

    @staticmethod
    def _sigmoid(x: float) -> float:
        # Numerically stable implementation.
        if x >= 0:
            z = np.exp(-x)
            return float(1.0 / (1.0 + z))
        z = np.exp(x)
        return float(z / (1.0 + z))

    def _ensure_syndrome_vector(self, syndrome: Sequence[int] | np.ndarray) -> np.ndarray:
        arr = np.asarray(syndrome, dtype=np.uint8)

        if arr.ndim != 1:
            raise ValueError(
                f"syndrome must be a 1D vector of size num_detectors={self.num_detectors}. "
                f"Received shape: {arr.shape}"
            )

        if arr.size != self.num_detectors:
            raise ValueError(
                f"Invalid syndrome size. Expected: {self.num_detectors}, received: {arr.size}"
            )

        return (arr & 1).astype(np.uint8)

    @staticmethod
    def _normalize_prediction(prediction: Any) -> np.ndarray:
        pred = np.asarray(prediction, dtype=np.uint8)
        if pred.ndim == 0:
            pred = pred.reshape(1)
        elif pred.ndim > 1:
            pred = pred.ravel()
        return (pred & 1).astype(np.uint8)

    def _estimate_total_weight(self, syndrome: np.ndarray, prediction: np.ndarray) -> float:
        """
        Heuristic estimate of the total matching cost.
        """
        syndrome_weight = float(np.sum(syndrome))
        pred_weight = float(np.sum(prediction))
        total_weight = 0.7 * syndrome_weight + 0.3 * pred_weight
        return float(max(0.0, total_weight))

    def _estimate_weight_gap(self, normalized_weight: float, syndrome_weight_norm: float) -> float:
        """
        Heuristic gap between best and second-best solutions.
        Not exact at this stage.
        """
        base_gap = 1.0 / (1.0 + max(0.0, normalized_weight))
        adjusted = base_gap + 0.15 * (1.0 - max(0.0, min(1.0, syndrome_weight_norm)))
        return float(max(0.0, min(1.0, adjusted)))

    def _compute_confidence(
        self,
        weight_gap: float,
        normalized_weight: float,
        syndrome_weight_norm: float,
    ) -> float:
        c = self.conf_cfg
        raw = (
            c.a * float(weight_gap)
            - c.b * float(normalized_weight)
            - c.c * float(syndrome_weight_norm)
            + c.bias
        )
        conf = self._sigmoid(raw)
        return float(max(0.0, min(1.0, conf)))

    def decode_with_confidence(
        self, syndrome: Sequence[int] | np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float], float]:
        """
        Decode one syndrome and return:
        - prediction (binary np.ndarray)
        - soft_info (metrics dict)
        - decode_time (seconds)
        """
        s = self._ensure_syndrome_vector(syndrome)

        t0 = perf_counter()
        pred_raw = self.matcher.decode(s)
        decode_time = float(perf_counter() - t0)

        prediction = self._normalize_prediction(pred_raw)

        syndrome_weight = float(np.sum(s))
        total_weight = self._estimate_total_weight(s, prediction)
        normalized_weight = float(total_weight / max(1.0, syndrome_weight))
        syndrome_weight_norm = float(syndrome_weight / max(1.0, float(self.num_detectors)))

        weight_gap = self._estimate_weight_gap(
            normalized_weight=normalized_weight,
            syndrome_weight_norm=syndrome_weight_norm,
        )

        confidence_score = self._compute_confidence(
            weight_gap=weight_gap,
            normalized_weight=normalized_weight,
            syndrome_weight_norm=syndrome_weight_norm,
        )

        soft_info: Dict[str, float] = {
            "syndrome_weight": syndrome_weight,
            "total_weight": float(total_weight),
            "weight_gap": float(weight_gap),
            "normalized_weight": float(normalized_weight),
            "confidence_score": float(confidence_score),
            "decode_time": float(decode_time),
        }

        return prediction, soft_info, decode_time

    def benchmark(
        self,
        shots: int = 1000,
        keep_soft_info_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Basic decoder benchmark:
        - error_rate
        - avg_decode_time
        - soft_info_samples
        """
        if not isinstance(shots, int) or shots <= 0:
            raise ValueError(f"shots must be int > 0. Received: {shots}")

        try:
            sampled = self.sampler.sample(shots=shots, separate_observables=True)
        except TypeError as exc:
            raise RuntimeError(
                "Your stim version does not support sample(..., separate_observables=True). "
                "Update stim to use benchmark with error_rate."
            ) from exc

        if not isinstance(sampled, tuple) or len(sampled) != 2:
            raise RuntimeError(
                "Expected Stim to return (detector_samples, observable_flips)."
            )

        dets, obs = sampled
        dets = np.asarray(dets, dtype=np.uint8)
        obs = np.asarray(obs, dtype=np.uint8)

        if dets.ndim != 2:
            raise RuntimeError(f"Invalid detector sample shape: {dets.shape}")
        if obs.ndim == 1:
            obs = obs[:, np.newaxis]
        if obs.ndim != 2:
            raise RuntimeError(f"Invalid observable sample shape: {obs.shape}")
        if dets.shape[0] != obs.shape[0]:
            raise RuntimeError(
                f"Shot count mismatch: dets={dets.shape[0]}, obs={obs.shape[0]}"
            )

        failures: List[bool] = []
        decode_times: List[float] = []
        soft_infos: List[Dict[str, float]] = []

        for i in range(shots):
            pred, s_info, dt = self.decode_with_confidence(dets[i])
            decode_times.append(float(dt))

            if keep_soft_info_samples is None or len(soft_infos) < keep_soft_info_samples:
                soft_infos.append(s_info)

            obs_i = (obs[i] & 1).astype(np.uint8)
            pred_i = (pred & 1).astype(np.uint8)
            n = min(obs_i.size, pred_i.size)
            if n == 0:
                failures.append(False)
            else:
                failures.append(bool(np.any(pred_i[:n] != obs_i[:n])))

        error_rate = float(np.mean(failures)) if failures else float("nan")
        avg_decode_time = float(np.mean(decode_times)) if decode_times else 0.0

        return {
            "shots": int(shots),
            "num_detectors": int(self.num_detectors),
            "num_observables": int(self.num_observables),
            "error_rate": error_rate,
            "avg_decode_time": avg_decode_time,
            "soft_info_samples": soft_infos,
        }


__all__ = ["ConfidenceConfig", "MWPMDecoderWithSoftInfo"]
