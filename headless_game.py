"""
Headless Snake game implementation for AI training.
No pygame dependencies - uses the core SnakeEngine.
"""
import numpy as np
from typing import Tuple

from snake_engine import SnakeEngine


class HeadlessGame:
    """Headless version of Snake game for AI training - wrapper around SnakeEngine"""
    
    def __init__(self) -> None:
        self.engine = SnakeEngine()
    
    def reset_game(self) -> None:
        """Reset game to initial state"""
        self.engine.reset()
    
    def step(self, action: np.ndarray) -> Tuple[float, bool, int]:
        """Execute one game step - main interface for AI training"""
        return self.engine.step_with_action(action)
    
    def get_state(self) -> np.ndarray:
        """Get current game state as numpy array for AI"""
        return self.engine.get_state_array()
    
    def get_observation(self) -> dict:
        """Get detailed observation for AI training"""
        return self.engine.get_observation()
    
    # Convenience properties for backward compatibility
    @property
    def snake_positions(self):
        return self.engine.snake_positions
    
    @property
    def apple_position(self):
        return self.engine.apple_position
    
    @property
    def score(self):
        return self.engine.score
    
    @property
    def game_over(self):
        return self.engine.game_over
    
    @property
    def reward(self):
        return self.engine.reward 