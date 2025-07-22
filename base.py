"""
Base Configuration Module

This module provides the foundational configuration for the Snake game,
including screen dimensions, colors, game mechanics, and directional mappings.
All game components inherit from this base class to ensure consistent configuration.
"""

from typing import Dict, Tuple, Union


# Type aliases for better code clarity
ColorTuple = Tuple[int, int, int]
Position = Tuple[int, int]
RewardValue = Union[int, float]


class Base:
    """
    Base configuration class for the Snake game.
    
    This class centralizes all game configuration including screen dimensions,
    colors, game mechanics parameters, and directional mappings. It ensures
    consistent configuration across all game components.
    
    All configuration values are set as instance attributes to maintain
    backward compatibility with existing code.
    """
    
    def __init__(self) -> None:
        """
        Initialize the base configuration with all game constants.
        
        Sets up screen dimensions, colors, game mechanics, and directional
        mappings that will be used throughout the game system.
        """
        # Screen and display configuration
        self._setup_screen_config()
        
        # Color palette definition
        self._setup_colors()
        
        # Game mechanics configuration
        self._setup_game_config()
        
        # Movement and direction mappings
        self._setup_direction_mappings()
    
    def _setup_screen_config(self) -> None:
        """Configure screen dimensions and related calculations."""
        self.SCREEN_SIZE: int = 600
        self.BLOCK_WIDTH: int = 20
        
        # Calculate maximum valid food position index
        # This ensures food spawns within screen boundaries
        self.MAX_FOOD_INDEX: int = (self.SCREEN_SIZE - self.BLOCK_WIDTH) // self.BLOCK_WIDTH
    
    def _setup_colors(self) -> None:
        """Define the color palette used throughout the game."""
        # Primary game colors (RGB tuples)
        self.BLACK: ColorTuple = (0, 0, 0)        # Background
        self.WHITE: ColorTuple = (255, 255, 255)  # Text/UI elements
        self.GREEN: ColorTuple = (0, 255, 0)      # Snake body
        self.RED: ColorTuple = (255, 0, 0)        # Apple/food
        self.GRAY: ColorTuple = (200, 200, 200)   # Score text
    
    def _setup_game_config(self) -> None:
        """Configure game mechanics and timing parameters."""
        # Game timing configuration
        self.GAME_SPEED: int = 10  # Frames per second
        
        # Enhanced reward system for AI training
        self.REWARDS: Dict[str, RewardValue] = {
            'step': -0.02,        # Increased urgency (was -0.01)
            'food': 10,           # Keep food reward  
            'collision': -15,     # More penalty (was -10)
            'closer_to_food': 0.3,   # Stronger guidance (was 0.1)
            'farther_from_food': -0.3,  # Stronger penalty (was -0.1)
            'survival_bonus': 0.01,      # Doubled (was 0.005)
            'length_bonus': 0.5,          # Keep same
            'inefficient_movement': -0.05  # Keep same
        }
    
    def _setup_direction_mappings(self) -> None:
        """
        Configure directional movement mappings.
        
        Maps direction names to (x, y) coordinate changes.
        These values represent pixel offsets based on BLOCK_WIDTH.
        """
        self.DIRECTIONS: Dict[str, Position] = {
            'right': (self.BLOCK_WIDTH, 0),   # Move right by one block
            'left': (-self.BLOCK_WIDTH, 0),   # Move left by one block
            'up': (0, -self.BLOCK_WIDTH),     # Move up by one block (negative Y)
            'down': (0, self.BLOCK_WIDTH)     # Move down by one block
        }
