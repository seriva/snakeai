"""
Snake Game - Pygame GUI Implementation

This module provides a graphical interface for the Snake game using pygame.
It uses the SnakeEngine for all game logic and focuses solely on rendering and user input.
"""

import random
import sys
from typing import List, Tuple, Optional

import numpy as np
import pygame

from base import Base
from snake_engine import SnakeEngine


class SnakeRenderer:
    """
    Handles rendering of the snake using pygame.
    
    This class is responsible for drawing the snake segments on the screen
    using efficient pygame rectangle operations.
    """
    
    def __init__(self, parent_screen: pygame.Surface, base_config: Base) -> None:
        """
        Initialize the snake renderer.
        
        Args:
            parent_screen: The pygame surface to draw on
            base_config: Base configuration object containing colors and dimensions
        """
        self.parent_screen = parent_screen
        self.color = base_config.GREEN
        self.block_width = base_config.BLOCK_WIDTH
        
    def draw(self, snake_positions: List[Tuple[int, int]]) -> None:
        """
        Draw snake using pygame rectangles for optimal performance.
        
        Args:
            snake_positions: List of (x, y) coordinates for each snake segment
        """
        for pos in snake_positions:
            pygame.draw.rect(
                self.parent_screen, 
                self.color, 
                (pos[0], pos[1], self.block_width, self.block_width)
            )


class AppleRenderer:
    """
    Handles rendering of the apple using pygame.
    
    This class is responsible for drawing the apple/food item on the screen
    using efficient pygame rectangle operations.
    """
    
    def __init__(self, parent_screen: pygame.Surface, base_config: Base) -> None:
        """
        Initialize the apple renderer.
        
        Args:
            parent_screen: The pygame surface to draw on
            base_config: Base configuration object containing colors and dimensions
        """
        self.parent_screen = parent_screen
        self.color = base_config.RED
        self.block_width = base_config.BLOCK_WIDTH
    
    def draw(self, apple_position: Tuple[int, int]) -> None:
        """
        Draw apple using pygame rectangle for optimal performance.
        
        Args:
            apple_position: (x, y) coordinates of the apple
        """
        pygame.draw.rect(
            self.parent_screen, 
            self.color,
            (apple_position[0], apple_position[1], self.block_width, self.block_width)
        )


class Game(Base):
    """
    Main Game class that handles the pygame GUI for Snake.
    
    This class manages the game window, user input, rendering, and coordinates
    with the SnakeEngine for all game logic. It provides both human-playable
    interface and AI training interface.
    """
    
    # UI Constants
    SCORE_FONT_SIZE = 20
    GAME_OVER_FONT_SIZE = 30
    SCORE_POSITION_OFFSET = 120
    SCORE_MARGIN = 10
    GAME_OVER_TEXT_OFFSET = 40
    
    def __init__(self) -> None:
        """Initialize the game with pygame setup and create necessary components."""
        super().__init__()
        
        # Initialize pygame with error handling
        try:
            pygame.init()
            pygame.display.set_caption("Snake")
            
            self.surface = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE))
            self.font = pygame.font.SysFont('arial', self.SCORE_FONT_SIZE)
            self.game_over_font = pygame.font.SysFont('arial', self.GAME_OVER_FONT_SIZE)
        except pygame.error as e:
            print(f"Failed to initialize pygame: {e}")
            sys.exit(1)
        
        # Use the core engine for all game logic
        self.engine = SnakeEngine()
        
        # Create renderers for drawing components
        self.snake_renderer = SnakeRenderer(self.surface, self)
        self.apple_renderer = AppleRenderer(self.surface, self)
        
        # Game state flags
        self.should_exit = False
        
        # Direction mapping for keyboard input
        self._direction_map = {
            pygame.K_UP: 'up',
            pygame.K_DOWN: 'down',
            pygame.K_LEFT: 'left',
            pygame.K_RIGHT: 'right'
        }
    
    def reset_game(self) -> None:
        """
        Reset game to initial state.
        
        This delegates to the engine's reset method to maintain separation of concerns.
        """
        self.engine.reset()
    
    def update_game_state(self) -> None:
        """
        Update game state for one frame.
        
        All game logic is handled by the engine to maintain clean architecture.
        """
        self.engine.step()
    
    def draw(self) -> None:
        """
        Draw the entire game state to the screen.
        
        This method handles both the active game state and game over screen.
        """
        self.surface.fill(self.BLACK)
        
        if not self.engine.game_over:
            self._draw_active_game()
        else:
            self._draw_game_over_screen()
    
    def _draw_active_game(self) -> None:
        """Draw the active game state including snake, apple, and score."""
        self.snake_renderer.draw(self.engine.snake_positions)
        self.apple_renderer.draw(self.engine.apple_position)
        self._draw_score()
    
    def _draw_score(self) -> None:
        """Draw the current score in the top-right corner."""
        score_text = self.font.render(f"Score: {self.engine.score}", True, self.GRAY)
        score_x = self.SCREEN_SIZE - self.SCORE_POSITION_OFFSET
        self.surface.blit(score_text, (score_x, self.SCORE_MARGIN))
    
    def _draw_game_over_screen(self) -> None:
        """
        Draw the game over screen with final score and restart instruction.
        
        Centers the text on the screen for better visual presentation.
        """
        center_x = self.SCREEN_SIZE // 2
        center_y = self.SCREEN_SIZE // 2
        
        # Game over message
        game_over_text = self.game_over_font.render(
            'Game Over! Press SPACE to restart', True, self.WHITE
        )
        game_over_rect = game_over_text.get_rect(center=(center_x, center_y))
        self.surface.blit(game_over_text, game_over_rect)
        
        # Final score
        score_text = self.game_over_font.render(
            f"Final Score: {self.engine.score}", True, self.WHITE
        )
        score_rect = score_text.get_rect(
            center=(center_x, center_y + self.GAME_OVER_TEXT_OFFSET)
        )
        self.surface.blit(score_text, score_rect)
    
    def handle_input(self, key: int) -> None:
        """
        Handle keyboard input for game control.
        
        Supported keys:
        - Arrow keys: Control snake direction
        - SPACE: Restart game when game over
        - ESCAPE: Exit the game
        
        Args:
            key: pygame key constant representing the pressed key
        """
        if key in self._direction_map:
            self.engine.set_direction(self._direction_map[key])
        elif key == pygame.K_SPACE and self.engine.game_over:
            self.reset_game()
        elif key == pygame.K_ESCAPE:
            self.should_exit = True
    
    def run_ai_step(self, move: np.ndarray) -> Tuple[float, bool, int]:
        """
        Execute one game step for AI training.
        
        This method provides backward compatibility with existing AI training code
        by delegating to the engine's step_with_action method.
        
        Args:
            move: numpy array representing the AI's chosen action
            
        Returns:
            Tuple containing (reward, game_over_status, current_score)
        """
        return self.engine.step_with_action(move)
    
    # Convenience properties for backward compatibility with existing code
    @property
    def score(self) -> int:
        """Get current game score."""
        return self.engine.score
    
    @property
    def game_over(self) -> bool:
        """Get current game over status."""
        return self.engine.game_over
    
    @property
    def reward(self) -> float:
        """Get current reward value."""
        return self.engine.reward


def main() -> None:
    """
    Main game loop with optimized event handling and rendering.
    
    This function creates the game instance and runs the main event loop,
    handling user input, game updates, and rendering at the configured frame rate.
    """
    try:
        game = Game()
        clock = pygame.time.Clock()
        running = True
        
        while running:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    game.handle_input(event.key)
            
            # Check if user wants to exit via escape key
            if game.should_exit:
                running = False
            
            # Update game state only if game is active
            if not game.engine.game_over:
                game.update_game_state()
            
            # Render the current frame
            game.draw()
            pygame.display.flip()  # More efficient than pygame.display.update()
            
            # Maintain consistent frame rate
            clock.tick(game.GAME_SPEED)
        
    except Exception as e:
        print(f"An error occurred during game execution: {e}")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()