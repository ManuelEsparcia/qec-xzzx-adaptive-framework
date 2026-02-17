from __future__ import annotations

import math
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

try:
    import beliefmatching  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover
    beliefmatching = None  # type: ignore[assignment]
    _BELIEFMATCHING_IMPORT_ERROR = exc
else:
    _BELIEFMATCHING_IMPORT_ERROR = None


@dataclass(frozen=True)
class BMConfidenceConfig:
    """
    Confidence heuristic for the BP/BM decoder.
    """

    a_agreement: float = 2.0
    a_sparse: float = 1.2
    b_norm_weight: float = 1.1
    c_entropy: float = 0.8
    d_backend: float = 0.3
    bias: float = 0.0
    eps: float = 1e-12


class BeliefMatchingDecoderWithSoftInfo:
    """
    Belief-Propagation / Belief-Matching decoder with robust fallback.

    Backend priority:
    1) `beliefmatching` package (if available with compatible API),
    2) `pymatching` with BP kwargs if the installed version supports it,
    3) default `pymatching`.
    """

    def __init__(
        self,
        circuit: "stim.Circuit",
        confidence_config: Optional[BMConfidenceConfig] = None,
        prefer_belief_propagation: bool = True,
    ) -> None:
        self._require_stim()
        self._require_pymatching()

        if not isinstance(circuit, stim.Circuit):
            raise TypeError(
                f"circuit must be stim.Circuit, received: {type(circuit).__name__}"
            )

        self.circuit = circuit
        self.conf_cfg = confidence_config or BMConfidenceConfig()
        self.dem = self.circuit.detector_error_model(decompose_errors=True)
        self.num_detectors = int(getattr(self.circuit, "num_detectors", 0))
        self.num_observables = int(getattr(self.circuit, "num_observables", 0))

        if self.num_detectors <= 0:
            raise ValueError("Circuit has no detectors (num_detectors <= 0).")

        self.matcher, self.matcher_build_mode = self._build_matcher(
            self.dem, prefer_belief_propagation=prefer_belief_propagation
        )
        self.decode_kwargs, self.decode_mode = self._select_decode_mode(
            self.matcher,
            self.num_detectors,
            prefer_belief_propagation=prefer_belief_propagation,
        )
        self.sampler = self.circuit.compile_detector_sampler()

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
    def _build_matcher(
        dem: "stim.DetectorErrorModel",
        prefer_belief_propagation: bool = True,
    ) -> Tuple[Any, str]:
        if not prefer_belief_propagation:
            return pymatching.Matching.from_detector_error_model(dem), "pymatching_default"

        # Try dedicated beliefmatching backend first.
        if beliefmatching is not None:
            matching_cls = getattr(beliefmatching, "Matching", None)
            if matching_cls is not None and hasattr(matching_cls, "from_detector_error_model"):
                try:
                    m = matching_cls.from_detector_error_model(dem)
                    return m, "beliefmatching_from_dem"
                except Exception:
                    pass

        candidate_kwargs = [
            {"decoder": "belief_propagation"},
            {"decoder": "bp"},
            {"decoder": "belief_matching"},
            {"decoder": "bm"},
            {"algorithm": "belief_propagation"},
            {"algorithm": "bp"},
            {"method": "belief_propagation"},
            {"decode_method": "belief_propagation"},
        ]

        for kw in candidate_kwargs:
            try:
                m = pymatching.Matching.from_detector_error_model(dem, **kw)
                return m, f"bp_via_from_dem:{kw}"
            except (TypeError, ValueError):
                continue

        return pymatching.Matching.from_detector_error_model(dem), "pymatching_default"

    @staticmethod
    def _select_decode_mode(
        matcher: Any,
        num_detectors: int,
        prefer_belief_propagation: bool = True,
    ) -> Tuple[Dict[str, Any], str]:
        zero = np.zeros(num_detectors, dtype=np.uint8)

        if not prefer_belief_propagation:
            _ = matcher.decode(zero)
            return {}, "default_decode"

        candidate_kwargs = [
            {"decoder": "belief_propagation"},
            {"decoder": "bp"},
            {"decoder": "belief_matching"},
            {"decoder": "bm"},
            {"algorithm": "belief_propagation"},
            {"algorithm": "bp"},
            {"method": "belief_propagation"},
            {"decode_method": "belief_propagation"},
            {},
        ]

        for kw in candidate_kwargs:
            try:
                _ = matcher.decode(zero, **kw)
                if kw:
                    return kw, f"bp_via_decode:{kw}"
                return {}, "default_decode"
            except (TypeError, ValueError):
                continue

        _ = matcher.decode(zero)
        return {}, "default_decode"

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = np.exp(-x)
            return float(1.0 / (1.0 + z))
        z = np.exp(x)
        return float(z / (1.0 + z))

    def _ensure_syndrome_vector(self, syndrome: Sequence[int] | np.ndarray) -> np.ndarray:
        arr = np.asarray(syndrome, dtype=np.uint8)
        if arr.ndim != 1:
            raise ValueError(
                f"syndrome must be vector 1D de tamano num_detectors={self.num_detectors}. "
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

    @staticmethod
    def _binary_entropy(x: float, eps: float) -> float:
        x = float(max(0.0, min(1.0, x)))
        if x <= eps or x >= 1.0 - eps:
            return 0.0
        return float(-(x * math.log(x + eps) + (1.0 - x) * math.log(1.0 - x + eps)))

    def _estimate_total_weight(self, s: np.ndarray, pred: np.ndarray) -> float:
        syndrome_weight = float(np.sum(s))
        pred_weight = float(np.sum(pred))
        return float(max(0.0, 0.6 * syndrome_weight + 0.4 * pred_weight))

    def _is_bp_backend(self) -> bool:
        text = f"{self.matcher_build_mode} | {self.decode_mode}".lower()
        return ("belief" in text) or (" bp" in text) or ("'bp'" in text)

    def _decode_backend(self, syndrome: np.ndarray) -> np.ndarray:
        try:
            pred_raw = self.matcher.decode(syndrome, **self.decode_kwargs)
        except TypeError:
            pred_raw = self.matcher.decode(syndrome)
        return self._normalize_prediction(pred_raw)

    def _compute_confidence(
        self,
        agreement_score: float,
        syndrome_density: float,
        normalized_weight: float,
        entropy_proxy: float,
        is_bp_backend: bool,
    ) -> float:
        c = self.conf_cfg
        backend_bonus = 1.0 if is_bp_backend else -0.2
        raw = (
            c.a_agreement * float(agreement_score)
            + c.a_sparse * float(1.0 - syndrome_density)
            - c.b_norm_weight * float(normalized_weight)
            - c.c_entropy * float(entropy_proxy)
            + c.d_backend * float(backend_bonus)
            + c.bias
        )
        conf = self._sigmoid(raw)
        return float(max(0.0, min(1.0, conf)))

    def decode_with_confidence(
        self, syndrome: Sequence[int] | np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float], float]:
        s = self._ensure_syndrome_vector(syndrome)

        t0 = perf_counter()
        pred = self._decode_backend(s)
        decode_time = float(perf_counter() - t0)

        syndrome_weight = float(np.sum(s))
        pred_weight = float(np.sum(pred))
        total_weight = self._estimate_total_weight(s, pred)
        normalized_weight = float(total_weight / max(1.0, syndrome_weight))
        syndrome_density = float(syndrome_weight / max(1.0, float(self.num_detectors)))

        prediction_density = float(pred_weight / max(1.0, float(max(1, pred.size))))
        agreement_score = float(max(0.0, 1.0 - abs(syndrome_density - prediction_density)))
        entropy_proxy = self._binary_entropy(syndrome_density, self.conf_cfg.eps)
        is_bp = self._is_bp_backend()

        confidence_score = self._compute_confidence(
            agreement_score=agreement_score,
            syndrome_density=syndrome_density,
            normalized_weight=normalized_weight,
            entropy_proxy=entropy_proxy,
            is_bp_backend=is_bp,
        )

        soft_info: Dict[str, float] = {
            "syndrome_weight": float(syndrome_weight),
            "prediction_weight": float(pred_weight),
            "total_weight": float(total_weight),
            "normalized_weight": float(normalized_weight),
            "syndrome_density": float(syndrome_density),
            "prediction_density": float(prediction_density),
            "agreement_score": float(agreement_score),
            "entropy_proxy": float(entropy_proxy),
            "confidence_score": float(confidence_score),
            "decode_time": float(decode_time),
            "is_bp_backend": float(1.0 if is_bp else 0.0),
        }
        return pred, soft_info, decode_time

    def benchmark(
        self,
        shots: int = 1000,
        keep_soft_info_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
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
            raise RuntimeError("Expected Stim to return (detector_samples, observable_flips).")

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
            "error_rate": float(error_rate),
            "avg_decode_time": float(avg_decode_time),
            "soft_info_samples": soft_infos,
            "backend_info": {
                "matcher_build_mode": self.matcher_build_mode,
                "decode_mode": self.decode_mode,
                "is_bp_backend": bool(self._is_bp_backend()),
                "beliefmatching_available": bool(beliefmatching is not None),
            },
        }


__all__ = ["BMConfidenceConfig", "BeliefMatchingDecoderWithSoftInfo"]
