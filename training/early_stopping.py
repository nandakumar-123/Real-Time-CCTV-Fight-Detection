"""
Early Stopping Handler.
Monitors validation metric (e.g., F1-score or loss) with configurable patience and min_delta.
"""

from typing import Optional
from utils.logger import train_logger

class EarlyStopping:
    """Early stops training when monitored metric ceases to improve."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-3,
        mode: str = "max"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

        assert self.mode in ["min", "max"], "EarlyStopping mode must be 'min' or 'max'"

    def step(self, current_score: float) -> bool:
        """
        Updates early stopping state.
        Returns:
            is_best (bool): True if current score is the best observed so far.
        """
        if self.best_score is None:
            self.best_score = current_score
            return True

        if self.mode == "max":
            improved = current_score > (self.best_score + self.min_delta)
        else:
            improved = current_score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            train_logger.info(f"EarlyStopping counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                train_logger.info("Early stopping triggered: no performance improvement.")
            return False

