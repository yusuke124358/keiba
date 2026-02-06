"""
確率校正（Calibration）

Isotonic Regression または Platt Scaling で確率を校正
"""
import logging
from typing import Optional
import pickle
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

def _clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip_prob(p)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _fit_temperature_scaling(y_prob: np.ndarray, y_true: np.ndarray) -> float:
    """Fit temperature T>0 for p' = sigmoid(logit(p)/T) by NLL minimization."""
    p = np.asarray(y_prob, dtype=float).reshape(-1)
    y = np.asarray(y_true, dtype=float).reshape(-1)
    if p.size == 0:
        return 1.0
    z = _logit(p)

    def nll(log_t: float) -> float:
        t = float(np.exp(log_t))
        if not np.isfinite(t) or t <= 0:
            t = 1.0
        zt = z / t
        # Stable binary log-loss from logits: softplus(z) - y*z
        return float(np.mean(np.logaddexp(0.0, zt) - (y * zt)))

    lo, hi = float(np.log(0.05)), float(np.log(10.0))
    best = 0.0
    best_loss = float("inf")
    for _ in range(3):
        grid = np.linspace(lo, hi, num=60)
        losses = [nll(v) for v in grid]
        i = int(np.argmin(losses))
        best = float(grid[i])
        best_loss = float(losses[i])
        # Narrow the window around the best point.
        span = (hi - lo) * 0.25
        lo, hi = best - span, best + span

    t_final = float(np.exp(best))
    if not np.isfinite(t_final) or t_final <= 0:
        t_final = 1.0
    logger.info(f"Temperature scaling fit: T={t_final:.6f}, nll={best_loss:.6f}")
    return t_final


class ProbabilityCalibrator:
    """確率校正器"""
    
    def __init__(self, method: str = "isotonic"):
        """
        Args:
            method: "isotonic" or "platt" or "temperature"
        """
        self.method = method
        self.calibrator = None
    
    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        """
        校正器を学習
        
        Args:
            y_prob: 予測確率
            y_true: 実績（0/1）
        """
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(
                y_min=0.0, 
                y_max=1.0, 
                out_of_bounds="clip"
            )
            self.calibrator.fit(y_prob, y_true)
        elif self.method == "platt":
            self.calibrator = LogisticRegression()
            self.calibrator.fit(y_prob.reshape(-1, 1), y_true)
        elif self.method == "temperature":
            # Store temperature (float) as the calibrator payload.
            self.calibrator = float(_fit_temperature_scaling(y_prob, y_true))
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self
    
    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """確率を校正"""
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted")
        
        if self.method == "isotonic":
            return self.calibrator.transform(y_prob)
        elif self.method == "platt":
            return self.calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        elif self.method == "temperature":
            t = float(self.calibrator)
            if not np.isfinite(t) or t <= 0:
                t = 1.0
            z = _logit(np.asarray(y_prob, dtype=float)) / t
            return _sigmoid(z)
    
    def save(self, path: Path) -> None:
        """保存"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "method": self.method,
                "calibrator": self.calibrator,
            }, f)
    
    @classmethod
    def load(cls, path: Path) -> "ProbabilityCalibrator":
        """読み込み"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        obj = cls(method=data["method"])
        obj.calibrator = data["calibrator"]
        return obj


def calibrate_model(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    method: str = "isotonic",
    calibrator_path: Optional[Path] = None,
) -> tuple[ProbabilityCalibrator, np.ndarray]:
    """
    確率を校正
    
    Args:
        y_prob: 予測確率
        y_true: 実績
        method: "isotonic" or "platt" or "temperature"
        calibrator_path: 保存先
    
    Returns:
        (calibrator, calibrated_probs)
    """
    calibrator = ProbabilityCalibrator(method=method)
    calibrator.fit(y_prob, y_true)
    
    calibrated = calibrator.transform(y_prob)
    
    if calibrator_path:
        calibrator.save(calibrator_path)
    
    logger.info(f"Calibration complete: method={method}")
    return calibrator, calibrated



