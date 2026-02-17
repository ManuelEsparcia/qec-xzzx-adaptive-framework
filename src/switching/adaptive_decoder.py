# src/switching/adaptive_decoder.py
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

from src.decoders.mwpm_decoder import MWPMDecoderWithSoftInfo
from src.decoders.union_find_decoder import UnionFindDecoderWithSoftInfo


@dataclass(frozen=True)
class AdaptiveConfig:
    """
    Adaptive decoder configuration.
    """
    g_threshold: float = 0.65
    # If True, benchmark also measures pure MWPM reference to compute speedup.
    compare_against_mwpm_in_benchmark: bool = True


class AdaptiveDecoder:
    """
    Adaptive decoder:
      1) Decode with a fast decoder (UF by default).
      2) If confidence_score < g_threshold, falls back to the accurate decoder (MWPM by default).
    """

    def __init__(
        self,
        circuit: "stim.Circuit",
        fast_decoder: Optional[Any] = None,
        accurate_decoder: Optional[Any] = None,
        config: Optional[AdaptiveConfig] = None,
    ) -> None:
        self._require_stim()

        if not isinstance(circuit, stim.Circuit):
            raise TypeError(
                f"circuit must be stim.Circuit, received: {type(circuit).__name__}"
            )

        self.circuit = circuit
        self.config = config or AdaptiveConfig()

        self._validate_threshold(self.config.g_threshold)

        # Default decoders if none are injected
        self.fast_decoder = fast_decoder or UnionFindDecoderWithSoftInfo(circuit)
        self.accurate_decoder = accurate_decoder or MWPMDecoderWithSoftInfo(circuit)

        # Minimal interface validation
        self._validate_decoder_interface(self.fast_decoder, "fast_decoder")
        self._validate_decoder_interface(self.accurate_decoder, "accurate_decoder")

        self.sampler = self.circuit.compile_detector_sampler()
        self.num_detectors = int(getattr(self.circuit, "num_detectors", 0))
        self.num_observables = int(getattr(self.circuit, "num_observables", 0))

        if self.num_detectors <= 0:
            raise ValueError("Circuit has no detectors (num_detectors <= 0).")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _require_stim() -> None:
        if stim is None:
            raise ImportError(
                "stim is not installed or failed to import. "
                "Install with: pip install stim"
            ) from _STIM_IMPORT_ERROR

    @staticmethod
    def _validate_threshold(g_threshold: float) -> None:
        if not isinstance(g_threshold, (float, int)):
            raise TypeError(f"g_threshold must be numeric. Received: {type(g_threshold)}")
        if not (0.0 <= float(g_threshold) <= 1.0):
            raise ValueError(f"g_threshold must be in [0,1]. Received: {g_threshold}")

    @staticmethod
    def _validate_decoder_interface(decoder: Any, name: str) -> None:
        if not hasattr(decoder, "decode_with_confidence"):
            raise TypeError(f"{name} does not implement decode_with_confidence(...)")

    @staticmethod
    def _normalize_prediction(prediction: Any) -> np.ndarray:
        pred = np.asarray(prediction, dtype=np.uint8)
        if pred.ndim == 0:
            pred = pred.reshape(1)
        elif pred.ndim > 1:
            pred = pred.ravel()
        return (pred & 1).astype(np.uint8)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_threshold(self, g_threshold: float) -> None:
        self._validate_threshold(g_threshold)
        self.config = AdaptiveConfig(
            g_threshold=float(g_threshold),
            compare_against_mwpm_in_benchmark=self.config.compare_against_mwpm_in_benchmark,
        )

    def decode_adaptive(
        self,
        syndrome: Sequence[int] | np.ndarray,
        g_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any], float]:
        """
        Two-stage adaptive decoding:
          - UF (fast)
          - fallback a MWPM if insufficient confidence
        """
        threshold = float(self.config.g_threshold if g_threshold is None else g_threshold)
        self._validate_threshold(threshold)

        t0 = perf_counter()

        pred_fast, soft_fast, t_fast = self.fast_decoder.decode_with_confidence(syndrome)
        pred_fast = self._normalize_prediction(pred_fast)

        fast_conf = float(soft_fast.get("confidence_score", 0.0))
        switched = bool(fast_conf < threshold)

        if switched:
            pred_final, soft_final, t_acc = self.accurate_decoder.decode_with_confidence(syndrome)
            pred_final = self._normalize_prediction(pred_final)
            selected_decoder = "mwpm"
        else:
            pred_final = pred_fast
            soft_final = soft_fast
            t_acc = 0.0
            selected_decoder = "uf"

        total_decode_time = float(perf_counter() - t0)

        adaptive_info: Dict[str, Any] = {
            "selected_decoder": selected_decoder,
            "switched": switched,
            "g_threshold": float(threshold),
            "fast_confidence_score": fast_conf,
            "fast_decode_time": float(t_fast),
            "accurate_decode_time": float(t_acc),
            "total_decode_time": float(total_decode_time),
            # Useful soft info from the finally selected decoder.
            "final_confidence_score": float(soft_final.get("confidence_score", fast_conf)),
            "final_soft_info": soft_final,
            # For traceability (useful for later analysis).
            "fast_soft_info": soft_fast,
        }

        return pred_final, adaptive_info, total_decode_time

    def benchmark_adaptive(
        self,
        shots: int = 1000,
        g_threshold: Optional[float] = None,
        keep_samples: Optional[int] = 100,
        compare_against_mwpm: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Benchmark for the adaptive decoder.
        Returns:
          - error_rate_adaptive
          - avg_decode_time_adaptive
          - switch_rate
          - (optional) MWPM reference and speedup
        """
        if not isinstance(shots, int) or shots <= 0:
            raise ValueError(f"shots must be int > 0. Received: {shots}")

        threshold = float(self.config.g_threshold if g_threshold is None else g_threshold)
        self._validate_threshold(threshold)

        do_compare = (
            self.config.compare_against_mwpm_in_benchmark
            if compare_against_mwpm is None
            else bool(compare_against_mwpm)
        )

        try:
            sampled = self.sampler.sample(shots=shots, separate_observables=True)
        except TypeError as exc:
            raise RuntimeError(
                "Tu versión de stim does not support sample(..., separate_observables=True). "
                "Update stim to use benchmark with error_rate."
            ) from exc

        if not isinstance(sampled, tuple) or len(sampled) != 2:
            raise RuntimeError(
                "Expected que Stim devolviera (detector_samples, observable_flips)."
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

        failures_adapt: List[bool] = []
        decode_times_adapt: List[float] = []
        switch_count = 0
        samples: List[Dict[str, Any]] = []

        # Optional MWPM reference
        failures_mwpm: List[bool] = []
        decode_times_mwpm: List[float] = []

        for i in range(shots):
            syndrome_i = dets[i]
            obs_i = (obs[i] & 1).astype(np.uint8)

            pred_a, info_a, dt_a = self.decode_adaptive(syndrome_i, g_threshold=threshold)
            pred_a = (pred_a & 1).astype(np.uint8)

            decode_times_adapt.append(float(dt_a))
            if bool(info_a.get("switched", False)):
                switch_count += 1

            n_a = min(obs_i.size, pred_a.size)
            if n_a == 0:
                fail_a = False
            else:
                fail_a = bool(np.any(pred_a[:n_a] != obs_i[:n_a]))
            failures_adapt.append(fail_a)

            if keep_samples is None or len(samples) < keep_samples:
                samples.append(
                    {
                        "selected_decoder": info_a.get("selected_decoder", "unknown"),
                        "switched": bool(info_a.get("switched", False)),
                        "fast_confidence_score": float(info_a.get("fast_confidence_score", 0.0)),
                        "final_confidence_score": float(info_a.get("final_confidence_score", 0.0)),
                        "total_decode_time": float(info_a.get("total_decode_time", dt_a)),
                    }
                )

            if do_compare:
                pred_m, _, dt_m = self.accurate_decoder.decode_with_confidence(syndrome_i)
                pred_m = self._normalize_prediction(pred_m)
                decode_times_mwpm.append(float(dt_m))

                n_m = min(obs_i.size, pred_m.size)
                if n_m == 0:
                    fail_m = False
                else:
                    fail_m = bool(np.any(pred_m[:n_m] != obs_i[:n_m]))
                failures_mwpm.append(fail_m)

        error_rate_adapt = float(np.mean(failures_adapt)) if failures_adapt else float("nan")
        avg_time_adapt = float(np.mean(decode_times_adapt)) if decode_times_adapt else 0.0
        switch_rate = float(switch_count / max(1, shots))

        result: Dict[str, Any] = {
            "shots": int(shots),
            "g_threshold": float(threshold),
            "num_detectors": int(self.num_detectors),
            "num_observables": int(self.num_observables),
            "error_rate_adaptive": error_rate_adapt,
            "avg_decode_time_adaptive": avg_time_adapt,
            "switch_rate": switch_rate,
            "samples": samples,
            "status": "ok",
        }

        if do_compare:
            error_rate_mwpm = float(np.mean(failures_mwpm)) if failures_mwpm else float("nan")
            avg_time_mwpm = float(np.mean(decode_times_mwpm)) if decode_times_mwpm else float("nan")

            if np.isfinite(avg_time_mwpm) and avg_time_adapt > 0:
                speedup_vs_mwpm = float(avg_time_mwpm / avg_time_adapt)
            else:
                speedup_vs_mwpm = float("nan")

            result["reference_mwpm"] = {
                "error_rate_mwpm": error_rate_mwpm,
                "avg_decode_time_mwpm": avg_time_mwpm,
            }
            result["speedup_vs_mwpm"] = speedup_vs_mwpm

        return result


__all__ = ["AdaptiveConfig", "AdaptiveDecoder"]
