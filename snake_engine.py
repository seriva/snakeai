"""
Core Snake game engine - contains all game logic without any rendering dependencies.
This is used by both the pygame GUI version and the headless AI training version.
"""
import random
from typing import List, Tuple, Optional

import numpy as np

from base import Base


class SnakeEngine(Base):
    """Core Snake game engine - pure game logic without rendering"""
    
    def __init__(self) -> None:
        super().__init__()
        self.reset()
    
    def reset(self) -> None:
        """Reset game to initial state"""
        self.snake_positions: List[Tuple[int, int]] = [(0, self.BLOCK_WIDTH)]
        self.snake_direction = 'right'
        self.apple_position = (self.BLOCK_WIDTH * 4, self.BLOCK_WIDTH * 5)
        self.score = 0
        self.game_over = False
        self.reward = 0
    
    # Snake properties
    @property
    def snake_head(self) -> Tuple[int, int]:
        """Get snake head position"""
        return self.snake_positions[0]
    
    @property
    def snake_length(self) -> int:
        """Get snake length"""
        return len(self.snake_positions)
    
    # Direction handling
    def set_direction(self, direction: str) -> None:
        """Set snake direction with validation to prevent reverse movement"""
        opposite = {'right': 'left', 'left': 'right', 'up': 'down', 'down': 'up'}
        if direction != opposite.get(self.snake_direction, ''):
            self.snake_direction = direction
    
    # Core game mechanics
    def _get_next_head_position(self) -> Tuple[int, int]:
        """Get the next head position based on current direction"""
        head_x, head_y = self.snake_positions[0]
        dx, dy = self.DIRECTIONS[self.snake_direction]
        return (head_x + dx, head_y + dy)
    
    def _move_snake(self) -> None:
        """Move snake in current direction"""
        next_head = self._get_next_head_position()
        # Add new head and always remove tail (growth happens separately)
        self.snake_positions.insert(0, next_head)
        self.snake_positions.pop()
    
    def _grow_snake(self) -> None:
        """Grow snake by adding segment at tail position"""
        tail = self.snake_positions[-1]
        self.snake_positions.append(tail)
    
    def _check_wall_collision(self, position: Tuple[int, int]) -> bool:
        """Check if position hits a wall"""
        x, y = position
        return (x < 0 or x >= self.SCREEN_SIZE or y < 0 or y >= self.SCREEN_SIZE)
    
    def _check_self_collision(self, position: Tuple[int, int]) -> bool:
        """Check if position hits snake body"""
        return position in self.snake_positions[1:]
    
    def _check_collision(self, position: Tuple[int, int]) -> bool:
        """Check if position collides with walls or snake body"""
        return self._check_wall_collision(position) or self._check_self_collision(position)
    
    def _check_food_collision(self, position: Tuple[int, int]) -> bool:
        """Check if position collides with apple"""
        return position == self.apple_position
    
    def _randomize_apple_position(self) -> None:
        """Generate new apple position avoiding snake body"""
        snake_positions_set = set(self.snake_positions)
        
        while True:
            x = random.randint(0, self.MAX_FOOD_INDEX) * self.BLOCK_WIDTH
            y = random.randint(0, self.MAX_FOOD_INDEX) * self.BLOCK_WIDTH
            new_pos = (x, y)
            
            if new_pos not in snake_positions_set:
                self.apple_position = new_pos
                break
    
    # Main game step
    def step(self, direction: Optional[str] = None) -> Tuple[float, bool, int]:
        """Execute one game step"""
        # Set direction if provided
        if direction:
            self.set_direction(direction)
        
        # Move snake first (like original game)
        self._move_snake()
        head_pos = self.snake_positions[0]
        
        # Set default step reward
        self.reward = self.REWARDS['step']
        
        # Check collisions FIRST (before growing)
        if self._check_collision(head_pos):
            self.game_over = True
            self.reward = self.REWARDS['collision']
            return self.reward, self.game_over, self.score
        
        # Check food collision (only if not game over)
        if self._check_food_collision(head_pos):
            self.score += 1
            self._grow_snake()
            self._randomize_apple_position()
            self.reward = self.REWARDS['food']
        
        return self.reward, self.game_over, self.score
    
    # AI Training utilities
    def step_with_action(self, action: np.ndarray) -> Tuple[float, bool, int]:
        """Execute one game step with AI action array"""
        action_map = ['right', 'down', 'left', 'up']
        direction_idx = np.argmax(action)
        direction = action_map[direction_idx]
        return self.step(direction)
    
    def get_state_array(self) -> np.ndarray:
        """Get current game state as numpy array for AI"""
        grid_size = self.SCREEN_SIZE // self.BLOCK_WIDTH
        state = np.zeros((grid_size, grid_size, 3))  # 3 channels: snake head, snake body, apple
        
        # Mark snake positions
        for i, pos in enumerate(self.snake_positions):
            x, y = pos[0] // self.BLOCK_WIDTH, pos[1] // self.BLOCK_WIDTH
            if 0 <= x < grid_size and 0 <= y < grid_size:
                if i == 0:  # Head
                    state[y, x, 0] = 1
                else:  # Body
                    state[y, x, 1] = 1
        
        # Mark apple position
        apple_x, apple_y = self.apple_position[0] // self.BLOCK_WIDTH, self.apple_position[1] // self.BLOCK_WIDTH
        if 0 <= apple_x < grid_size and 0 <= apple_y < grid_size:
            state[apple_y, apple_x, 2] = 1
        
        return state
    
    def get_observation(self) -> dict:
        """Get detailed observation for AI training"""
        head_x, head_y = self.snake_positions[0]
        apple_x, apple_y = self.apple_position
        
        return {
            'snake_head': (head_x // self.BLOCK_WIDTH, head_y // self.BLOCK_WIDTH),
            'snake_body': [(x // self.BLOCK_WIDTH, y // self.BLOCK_WIDTH) for x, y in self.snake_positions[1:]],
            'apple': (apple_x // self.BLOCK_WIDTH, apple_y // self.BLOCK_WIDTH),
            'direction': self.snake_direction,
            'score': self.score,
            'snake_length': len(self.snake_positions),
            'game_over': self.game_over
        } 