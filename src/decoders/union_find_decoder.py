# src/decoders/union_find_decoder.py
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
class UFConfidenceConfig:
    """
    Confidence heuristic configuration for Union-Find.

    confidence = sigmoid(
        a_cluster * clustering_quality
        - b_norm_weight * normalized_weight
        - c_density * syndrome_density
        - d_merge * merge_ratio
        + e_backend * backend_bonus
        + bias
    )
    """
    a_cluster: float = 2.2
    b_norm_weight: float = 1.1
    c_density: float = 0.9
    d_merge: float = 0.8
    e_backend: float = 0.2
    bias: float = 0.0
    eps: float = 1e-12


class UnionFindDecoderWithSoftInfo:
    """
    Fast decoder with a Union-Find-style interface and soft information.

    Important note:
    - It tries to use a UF backend when the environment/API supports it.
    - If unavailable, it falls back to the default matcher decode and sets
      `is_union_find_backend=False` in soft_info for transparency.
    """

    def __init__(
        self,
        circuit: "stim.Circuit",
        confidence_config: Optional[UFConfidenceConfig] = None,
        prefer_union_find: bool = True,
    ) -> None:
        self._require_stim()
        self._require_pymatching()

        if not isinstance(circuit, stim.Circuit):
            raise TypeError(
                f"circuit must be stim.Circuit, received: {type(circuit).__name__}"
            )

        self.circuit = circuit
        self.conf_cfg = confidence_config or UFConfidenceConfig()

        self.dem = self.circuit.detector_error_model(decompose_errors=True)
        self.num_detectors = int(getattr(self.circuit, "num_detectors", 0))
        self.num_observables = int(getattr(self.circuit, "num_observables", 0))

        if self.num_detectors <= 0:
            raise ValueError("Circuit has no detectors (num_detectors <= 0).")

        self.matcher, self.matcher_build_mode = self._build_matcher(
            self.dem, prefer_union_find=prefer_union_find
        )
        self.decode_kwargs, self.decode_mode = self._select_decode_mode(
            self.matcher, self.num_detectors, prefer_union_find=prefer_union_find
        )
        self.sampler = self.circuit.compile_detector_sampler()

    # ------------------------------------------------------------------
    # Import checks
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Matcher/decode mode selection
    # ------------------------------------------------------------------
    @staticmethod
    def _build_matcher(
        dem: "stim.DetectorErrorModel",
        prefer_union_find: bool = True,
    ) -> Tuple["pymatching.Matching", str]:
        """
        Try to construct a matcher in UF mode if the API supports it.
        Otherwise, fall back to the default constructor.
        """
        if not prefer_union_find:
            return pymatching.Matching.from_detector_error_model(dem), "default"

        candidate_kwargs = [
            {"decoder": "union_find"},
            {"decoder": "uf"},
            {"algorithm": "union_find"},
            {"algorithm": "uf"},
            {"method": "union_find"},
            {"decode_method": "union_find"},
        ]

        for kw in candidate_kwargs:
            try:
                m = pymatching.Matching.from_detector_error_model(dem, **kw)
                return m, f"union_find_via_from_dem:{kw}"
            except (TypeError, ValueError):
                continue

        # Safe fallback
        return pymatching.Matching.from_detector_error_model(dem), "default"

    @staticmethod
    def _select_decode_mode(
        matcher: "pymatching.Matching",
        num_detectors: int,
        prefer_union_find: bool = True,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Try to discover decode(...) kwargs that enable UF at runtime.
        """
        zero = np.zeros(num_detectors, dtype=np.uint8)

        if not prefer_union_find:
            # Default decode
            _ = matcher.decode(zero)
            return {}, "default_decode"

        candidate_kwargs = [
            {"decoder": "union_find"},
            {"decoder": "uf"},
            {"algorithm": "union_find"},
            {"algorithm": "uf"},
            {"method": "union_find"},
            {"decode_method": "union_find"},
            {},  # fallback
        ]

        for kw in candidate_kwargs:
            try:
                _ = matcher.decode(zero, **kw)
                if kw:
                    return kw, f"union_find_via_decode:{kw}"
                return {}, "default_decode"
            except (TypeError, ValueError):
                continue

        # In a very rare case, try plain decode without kwargs.
        _ = matcher.decode(zero)
        return {}, "default_decode"

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
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
                f"syndrome must be vector 1D de tamaño num_detectors={self.num_detectors}. "
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
    def _cluster_stats_from_syndrome(s: np.ndarray) -> Tuple[int, int, float, float]:
        """
        UF clustering heuristic using the 1D structure of active detector indices:
        - n_clusters: contiguous segments of active detectors
        - merge_count: active - n_clusters
        - clustering_quality in [0,1]: n_clusters / active
        - merge_ratio in [0,1]: merge_count / active
        """
        active = np.flatnonzero(s)
        active_count = int(active.size)

        if active_count == 0:
            return 0, 0, 1.0, 0.0

        diffs = np.diff(active)
        # New cluster when indices are not contiguous (gap > 1)
        n_clusters = int(1 + np.sum(diffs > 1))
        merge_count = int(max(0, active_count - n_clusters))

        clustering_quality = float(n_clusters / max(1, active_count))
        merge_ratio = float(merge_count / max(1, active_count))
        return n_clusters, merge_count, clustering_quality, merge_ratio

    def _estimate_total_weight(self, s: np.ndarray, pred: np.ndarray) -> float:
        syndrome_weight = float(np.sum(s))
        pred_weight = float(np.sum(pred))
        # Stable blend, similar in spirit to this project's MWPM decoder.
        total_weight = 0.65 * syndrome_weight + 0.35 * pred_weight
        return float(max(0.0, total_weight))

    def _compute_confidence(
        self,
        clustering_quality: float,
        normalized_weight: float,
        syndrome_density: float,
        merge_ratio: float,
        is_union_backend: bool,
    ) -> float:
        c = self.conf_cfg
        backend_bonus = 1.0 if is_union_backend else -0.2

        raw = (
            c.a_cluster * float(clustering_quality)
            - c.b_norm_weight * float(normalized_weight)
            - c.c_density * float(syndrome_density)
            - c.d_merge * float(merge_ratio)
            + c.e_backend * float(backend_bonus)
            + c.bias
        )
        conf = self._sigmoid(raw)
        return float(max(0.0, min(1.0, conf)))

    def _decode_backend(self, syndrome: np.ndarray) -> np.ndarray:
        pred_raw = self.matcher.decode(syndrome, **self.decode_kwargs)
        return self._normalize_prediction(pred_raw)

    def _is_union_find_backend(self) -> bool:
        text = f"{self.matcher_build_mode} | {self.decode_mode}".lower()
        return "union_find" in text or "'uf'" in text or " uf" in text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def decode_with_confidence(
        self, syndrome: Sequence[int] | np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float], float]:
        """
        Decode one syndrome and return:
        - prediction (binary np.ndarray)
        - soft_info (dict)
        - decode_time (float, seconds)
        """
        s = self._ensure_syndrome_vector(syndrome)

        t0 = perf_counter()
        pred = self._decode_backend(s)
        decode_time = float(perf_counter() - t0)

        syndrome_weight = float(np.sum(s))
        total_weight = self._estimate_total_weight(s, pred)
        normalized_weight = float(total_weight / max(1.0, syndrome_weight))
        syndrome_density = float(syndrome_weight / max(1.0, float(self.num_detectors)))

        n_clusters, merge_count, clustering_quality, merge_ratio = self._cluster_stats_from_syndrome(s)
        is_uf = self._is_union_find_backend()

        confidence_score = self._compute_confidence(
            clustering_quality=clustering_quality,
            normalized_weight=normalized_weight,
            syndrome_density=syndrome_density,
            merge_ratio=merge_ratio,
            is_union_backend=is_uf,
        )

        soft_info: Dict[str, float] = {
            "syndrome_weight": float(syndrome_weight),
            "total_weight": float(total_weight),
            "normalized_weight": float(normalized_weight),
            "clustering_quality": float(clustering_quality),
            "merge_count": float(merge_count),
            "n_clusters": float(n_clusters),
            "syndrome_density": float(syndrome_density),
            "merge_ratio": float(merge_ratio),
            "confidence_score": float(confidence_score),
            "decode_time": float(decode_time),
            # Traceability flags (numeric for easier serialization/analysis).
            "is_union_find_backend": float(1.0 if is_uf else 0.0),
        }

        return pred, soft_info, decode_time

    def benchmark(
        self,
        shots: int = 1000,
        keep_soft_info_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Benchmark for the UF-like decoder:
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
                "Your Stim version does not support sample(..., separate_observables=True). "
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
                fail_i = bool(np.any(pred_i[:n] != obs_i[:n]))
                failures.append(fail_i)

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
                "is_union_find_backend": bool(self._is_union_find_backend()),
            },
        }


__all__ = ["UFConfidenceConfig", "UnionFindDecoderWithSoftInfo"]
