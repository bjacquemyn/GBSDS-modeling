"""
Proportional Odds Model for GBS Disability Outcome Shifts.

This module implements the proportional odds model used to simulate
treatment effects on ordinal disability outcomes in GBS clinical trials.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ShiftResult:
    """Container for shift calculation results.
    
    Attributes:
        new_proportions: Array of new category proportions (percentages).
        odds_ratio: The odds ratio used for the shift.
        control_walking: Control arm walking independence rate (%).
        new_walking: New walking independence rate (%).
        risk_difference: Absolute difference in walking rates (%).
    """
    new_proportions: np.ndarray
    odds_ratio: float
    control_walking: float
    new_walking: float
    risk_difference: float


class ProportionalOddsModel:
    """Proportional odds model for GBS disability outcome shifts.
    
    The proportional odds model assumes a common odds ratio applies
    across all cumulative probability thresholds of an ordinal outcome.
    
    Attributes:
        control_proportions: Control arm category proportions (percentages).
        walking_cutoff: Maximum score considered "walking independent" (default: 2).
    """
    
    def __init__(
        self, 
        control_proportions: np.ndarray,
        walking_cutoff: int = 2
    ):
        """Initialize the model with control arm proportions.
        
        Args:
            control_proportions: Array of percentages for each disability category.
                                  Must sum to 100.
            walking_cutoff: Maximum GBS score considered as walking independent.
        
        Raises:
            ValueError: If proportions don't sum to ~100 or contain invalid values.
        """
        self.control_proportions = np.array(control_proportions, dtype=float)
        self.walking_cutoff = walking_cutoff
        self._validate()
    
    def _validate(self) -> None:
        """Validate control arm proportions."""
        total = np.sum(self.control_proportions)
        if not (99.9 <= total <= 100.1):
            raise ValueError(f"Proportions must sum to 100, got {total:.1f}")
        if np.any(self.control_proportions < 0):
            raise ValueError("Proportions cannot be negative")
        if np.any(self.control_proportions > 100):
            raise ValueError("Individual proportions cannot exceed 100")
    
    @property
    def control_probs(self) -> np.ndarray:
        """Control arm proportions as probabilities (0-1)."""
        return self.control_proportions / 100.0
    
    @property
    def control_walking_rate(self) -> float:
        """Control arm walking independence rate (percentage)."""
        return float(np.sum(self.control_proportions[:self.walking_cutoff + 1]))
    
    def calculate_shift(self, odds_ratio: float) -> ShiftResult:
        """Calculate the shifted distribution given an odds ratio.
        
        Applies the proportional odds assumption: the same OR applies
        to all cumulative probability thresholds.
        
        Args:
            odds_ratio: Common odds ratio to apply. OR > 1 shifts toward 
                        better outcomes, OR < 1 shifts toward worse.
        
        Returns:
            ShiftResult containing new proportions and summary statistics.
        """
        # Calculate cumulative probabilities P(Y <= k)
        cum_probs = np.cumsum(self.control_probs)
        cum_probs[-1] = 1.0  # Handle precision
        
        new_cum_probs = []
        
        # Apply OR to each cumulative threshold (except last which stays 1.0)
        for i in range(len(self.control_probs) - 1):
            cp = cum_probs[i]
            
            if cp >= 1.0 or cp <= 0.0:
                new_cp = cp
            else:
                odds = cp / (1.0 - cp)
                shifted_odds = odds * odds_ratio
                new_cp = shifted_odds / (1.0 + shifted_odds)
            
            new_cum_probs.append(new_cp)
        
        new_cum_probs.append(1.0)
        
        # Convert cumulative probabilities back to category percentages
        new_props = np.diff(np.insert(new_cum_probs, 0, 0)) * 100
        
        # Ensure non-negative (can happen with extreme ORs)
        new_props = np.maximum(new_props, 0)
        
        # Calculate walking rates
        new_walking = float(np.sum(new_props[:self.walking_cutoff + 1]))
        
        return ShiftResult(
            new_proportions=new_props,
            odds_ratio=odds_ratio,
            control_walking=self.control_walking_rate,
            new_walking=new_walking,
            risk_difference=new_walking - self.control_walking_rate
        )
    
    def find_or_for_target(
        self, 
        target_risk_difference: float,
        cutoff_idx: Optional[int] = None
    ) -> float:
        """Back-calculate the OR required to achieve a target risk difference.
        
        Args:
            target_risk_difference: Desired absolute increase in walking rate 
                                    (as a decimal, e.g., 0.20 for 20%).
            cutoff_idx: Index of the cutoff category. Defaults to walking_cutoff.
        
        Returns:
            The odds ratio required to achieve the target improvement.
        """
        if cutoff_idx is None:
            cutoff_idx = self.walking_cutoff
        
        control_cum_prob = np.sum(self.control_probs[:cutoff_idx + 1])
        target_cum_prob = control_cum_prob + target_risk_difference
        
        # Clamp to valid probability range
        target_cum_prob = np.clip(target_cum_prob, 0.001, 0.999)
        
        # Calculate odds
        control_odds = control_cum_prob / (1.0 - control_cum_prob)
        target_odds = target_cum_prob / (1.0 - target_cum_prob)
        
        if control_odds == 0:
            return 1.0
        
        return target_odds / control_odds
