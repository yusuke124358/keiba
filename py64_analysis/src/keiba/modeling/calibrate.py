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


class ProbabilityCalibrator:
    """確率校正器"""
    
    def __init__(self, method: str = "isotonic"):
        """
        Args:
            method: "isotonic" or "platt"
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
        method: "isotonic" or "platt"
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



