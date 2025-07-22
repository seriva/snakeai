"""
AI Demo - Watch your trained DQN V1 and V2 agents play Snake!
Run this after training to see how well your models perform.
"""
import pygame
import numpy as np
from typing import Dict, Tuple, List, Optional

from dqn_agent_v1 import DQNAgentV1
from dqn_agent_v2 import DQNAgentV2
from headless_game import HeadlessGame


class AIDemo:
    """Demo class to watch AI agents play Snake with visual feedback"""
    
    # Constants
    DEFAULT_SPEED_MULTIPLIER = 1.0
    DEFAULT_GAMES = 3
    GAME_OVER_DISPLAY_FRAMES = 15
    TARGET_FPS = 30
    
    def __init__(self):        
        # Agent management
        self.agents: Dict[str, Optional[object]] = {
            'v1': None,
            'v2': None
        }
        
    def load_agent(self, agent_type: str) -> bool:
        """Load a specific agent type"""
        if agent_type == 'v1':
            if self.agents['v1'] is None:
                agent = DQNAgentV1()
                if agent.load_model("model/dqn_agent_v1.pth", "model/dqn_agent_v1_data.json"):
                    agent.epsilon = 0.0  # Pure exploitation
                    self.agents['v1'] = agent
                    print("🤖 DQN V1 Agent:")
                    print(f"   Model: model/dqn_agent_v1.pth")
                    print(f"   Status: Ready for inference")
                    return True
                else:
                    return False
            return True
            
        elif agent_type == 'v2':
            if self.agents['v2'] is None:
                agent = DQNAgentV2()
                if agent.load_model("model/dqn_agent_v2.pth", "model/dqn_agent_v2_data.json"):
                    agent.epsilon = 0.0  # Pure exploitation  
                    self.agents['v2'] = agent
                    print("🚀 DQN V2 Agent:")
                    print(f"   Model: model/dqn_agent_v2.pth")
                    print(f"   Status: Ready for inference")
                    return True
                else:
                    return False
            return True
            
        return False
    

    
    def _handle_events(self) -> Tuple[bool, bool]:
        """Handle pygame events. Returns (continue_running, restart_game)"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False, False
                elif event.key == pygame.K_SPACE:
                    return True, True
        return True, False
    

    

    
    def get_available_agents(self) -> Dict[str, bool]:
        """Check which agents are available"""
        available = {}
        
        # Check DQN V1
        try:
            available['v1'] = self.load_agent('v1')
        except:
            available['v1'] = False
            
        # Check DQN V2
        try:
            available['v2'] = self.load_agent('v2')
        except:
            available['v2'] = False
            
        return available
    
    def _init_side_by_side_games(self):
        """Initialize two games for side-by-side comparison."""
        from base import Base  # Import for screen size constants
        pygame.init()
        
        # Create wider window for two games side by side
        base_config = Base()
        screen_width = base_config.SCREEN_SIZE * 2 + 40  # 20px gap between games
        screen_height = base_config.SCREEN_SIZE + 80  # Extra space for labels
        self.comparison_surface = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Snake AI Comparison - DQN V1 vs V2")
        
        # Create separate surfaces for each game
        self.game_size = base_config.SCREEN_SIZE
        self.block_width = base_config.BLOCK_WIDTH
        self.left_surface = pygame.Surface((self.game_size, self.game_size))
        self.right_surface = pygame.Surface((self.game_size, self.game_size))
        
        # Store colors for rendering
        self.colors = {
            'black': base_config.BLACK,
            'green': base_config.GREEN,
            'red': base_config.RED,
            'white': base_config.WHITE
        }
        
        # Create headless games for game logic
        self.headless_left = HeadlessGame()
        self.headless_right = HeadlessGame()
        
        # Create font for labels
        self.label_font = pygame.font.SysFont('arial', 24)
    
    def run_side_by_side_demo(self, games_to_play: int = DEFAULT_GAMES, speed_multiplier: float = DEFAULT_SPEED_MULTIPLIER):
        """Run both agents side by side for comparison."""
        # Load both agents
        if not (self.load_agent('v1') and self.load_agent('v2')):
            print("❌ Both agents must be available for side-by-side comparison!")
            return
        
        print(f"✅ Both agents loaded successfully!")
        print(f"   V1: {type(self.agents['v1']).__name__}")
        print(f"   V2: {type(self.agents['v2']).__name__}")
        
        self._init_side_by_side_games()
        
        try:
            print(f"\n🎮 Starting Side-by-Side Comparison")
            print(f"Left: DQN V1  |  Right: DQN V2")
            print(f"Playing {games_to_play} games...")
            print("Press ESCAPE to quit, SPACE to restart current games")
            print("=" * 50)
            
            clock = pygame.time.Clock()
            # Use slower default speed for side-by-side comparison
            from base import Base
            game_speed = max(3, int(Base().GAME_SPEED * speed_multiplier))
            print(f"🎮 Game speed: {game_speed} FPS")
            
            v1_scores = []
            v2_scores = []
            current_game = 1
            
            while current_game <= games_to_play:
                # Reset both games
                self.headless_left.reset_game()
                self.headless_right.reset_game()
                
                print(f"\n🎯 Game {current_game}/{games_to_play}")
                step_count = 0
                
                # Main comparison loop
                while True:
                    continue_running, restart_games = self._handle_events()
                    
                    if not continue_running:
                        print("\n👋 Demo ended by user")
                        return
                    
                    if restart_games:
                        print("🔄 Games restarted by user")
                        break
                    
                    # Check if both games are over
                    if self.headless_left.game_over and self.headless_right.game_over:
                        break
                    
                    # Run DQN V1 agent (left side)
                    if not self.headless_left.game_over:
                        state_left = self.agents['v1'].get_state(self.headless_left)
                        action_left = self.agents['v1'].get_action(state_left, epsilon=0.0)
                        move_left = np.zeros(4, dtype=np.float32)
                        move_left[action_left] = 1
                        self.headless_left.step(move_left)
                    
                    # Run DQN V2 agent (right side)
                    if not self.headless_right.game_over:
                        state_right = self.agents['v2'].get_state(self.headless_right)
                        action_right = self.agents['v2'].get_action(state_right, epsilon=0.0)
                        move_right = np.zeros(4, dtype=np.float32)
                        move_right[action_right] = 1
                        self.headless_right.step(move_right)
                    
                    # Render both games
                    self._draw_side_by_side(current_game, games_to_play)
                    
                    pygame.display.flip()
                    clock.tick(game_speed)
                    
                    step_count += 1
                    # Show progress every 50 steps
                    if step_count % 50 == 0:
                        print(f"   Step {step_count}: V1={self.headless_left.engine.score}, V2={self.headless_right.engine.score}")
                
                # Handle game over
                if self.headless_left.game_over and self.headless_right.game_over:
                    v1_score = self.headless_left.engine.score
                    v2_score = self.headless_right.engine.score
                    
                    v1_scores.append(v1_score)
                    v2_scores.append(v2_score)
                    
                    print(f"💀 Game Over! V1: {v1_score} | V2: {v2_score}")
                    
                    # Show final scores for a moment
                    for _ in range(self.GAME_OVER_DISPLAY_FRAMES):
                        continue_running, _ = self._handle_events()
                        if not continue_running:
                            return
                        self._draw_side_by_side(current_game, games_to_play)
                        pygame.display.flip()
                        clock.tick(self.TARGET_FPS)
                    
                    current_game += 1
            
            # Show final comparison
            self._show_comparison_summary(v1_scores, v2_scores)
            
        except KeyboardInterrupt:
            print("\n⏹️ Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during demo: {e}")
        finally:
            # Clean up pygame
            pygame.quit()
    
    def _draw_side_by_side(self, current_game: int, total_games: int):
        """Draw both games side by side."""
        # Clear main surface
        self.comparison_surface.fill((20, 20, 20))  # Dark background
        
        # Draw game counter at top center
        game_counter_text = self.label_font.render(f"Game {current_game}/{total_games}", True, (255, 255, 255))
        total_width = self.game_size * 2 + 40  # Total window width
        counter_x = (total_width - game_counter_text.get_width()) // 2
        self.comparison_surface.blit(game_counter_text, (counter_x, 5))
        
        # Draw left game (DQN V1)
        self._draw_single_game(self.left_surface, self.headless_left, "DQN V1")
        self.comparison_surface.blit(self.left_surface, (10, 50))
        
        # Draw right game (DQN V2)
        self._draw_single_game(self.right_surface, self.headless_right, "DQN V2")
        self.comparison_surface.blit(self.right_surface, (self.game_size + 30, 50))
        
        # Draw labels
        v1_label = self.label_font.render("DQN V1", True, (255, 255, 255))
        v2_label = self.label_font.render("DQN V2", True, (255, 255, 255))
        
        # Center labels above each game
        v1_x = 10 + (self.game_size - v1_label.get_width()) // 2
        v2_x = self.game_size + 30 + (self.game_size - v2_label.get_width()) // 2
        
        self.comparison_surface.blit(v1_label, (v1_x, 20))
        self.comparison_surface.blit(v2_label, (v2_x, 20))
        
        # Draw scores
        v1_score_text = self.label_font.render(f"Score: {self.headless_left.engine.score}", True, (255, 255, 255))
        v2_score_text = self.label_font.render(f"Score: {self.headless_right.engine.score}", True, (255, 255, 255))
        
        v1_score_x = 10 + (self.game_size - v1_score_text.get_width()) // 2
        v2_score_x = self.game_size + 30 + (self.game_size - v2_score_text.get_width()) // 2
        
        self.comparison_surface.blit(v1_score_text, (v1_score_x, self.game_size + 55))
        self.comparison_surface.blit(v2_score_text, (v2_score_x, self.game_size + 55))
    
    def _draw_single_game(self, surface, headless_game, agent_name):
        """Draw a single game on the given surface."""
        # Fill background
        surface.fill(self.colors['black'])
        
        # Draw snake
        for pos in headless_game.engine.snake_positions:
            pygame.draw.rect(surface, self.colors['green'], 
                           (pos[0], pos[1], self.block_width, self.block_width))
        
        # Draw apple
        apple_pos = headless_game.engine.apple_position
        pygame.draw.rect(surface, self.colors['red'], 
                        (apple_pos[0], apple_pos[1], self.block_width, self.block_width))
    
    def _show_comparison_summary(self, v1_scores: List[int], v2_scores: List[int]):
        """Display comparison summary statistics."""
        print("\n" + "=" * 60)
        print("📊 SIDE-BY-SIDE COMPARISON RESULTS")
        print("=" * 60)
        
        print(f"\nDQN V1 Agent (Simple):")
        print(f"  Average Score: {np.mean(v1_scores):.2f}")
        print(f"  Best Score:    {max(v1_scores)}")
        print(f"  All Scores:    {v1_scores}")
        
        print(f"\nDQN V2 Agent (Enhanced):")
        print(f"  Average Score: {np.mean(v2_scores):.2f}")
        print(f"  Best Score:    {max(v2_scores)}")
        print(f"  All Scores:    {v2_scores}")
        
        # Winner determination
        v1_avg = np.mean(v1_scores)
        v2_avg = np.mean(v2_scores)
        
        print(f"\n🏆 WINNER: ", end="")
        if v2_avg > v1_avg:
            improvement = ((v2_avg - v1_avg) / v1_avg) * 100
            print(f"DQN V2 (+{improvement:.1f}% better)")
        elif v1_avg > v2_avg:
            improvement = ((v1_avg - v2_avg) / v2_avg) * 100
            print(f"DQN V1 (+{improvement:.1f}% better)")
        else:
            print("TIE!")
        
        print("=" * 60)


def get_user_input(prompt: str, default: str, input_type=str):
    """Get user input with default value and type conversion"""
    try:
        user_input = input(prompt).strip()
        if not user_input:
            return input_type(default)
        return input_type(user_input)
    except ValueError:
        print(f"Invalid input, using default: {default}")
        return input_type(default)


def main():
    """Main demo function - runs side-by-side comparison"""
    print("🐍 Snake AI Demo - Side-by-Side Comparison")
    print("=" * 45)
    
    demo = AIDemo()
    
    # Check available agents
    available_agents = demo.get_available_agents()
    
    if not (available_agents['v1'] and available_agents['v2']):
        print("❌ Both DQN V1 and V2 models are required!")
        print("Available agents:")
        print(f"  - DQN V1: {'✅' if available_agents['v1'] else '❌'}")
        print(f"  - DQN V2: {'✅' if available_agents['v2'] else '❌'}")
        print("\nPlease train both agents first:")
        print("  uv run dqn_agent_v1.py")
        print("  uv run dqn_agent_v2.py")
        return
    
    print("\n✅ Both agents loaded successfully!")
    print("  - DQN V1 (Standard Deep Q-Network)")
    print("  - DQN V2 (Enhanced with reward shaping & better hyperparameters)")
    
    print("\n🎮 Controls during demo:")
    print("  - ESC: Quit demo")
    print("  - SPACE: Restart current games")
    print("  - Close window: Quit demo\n")
    
    # Get demo parameters
    games = get_user_input("How many games to play? (default 5): ", "5", int)
    speed = get_user_input("Speed multiplier (default 10.0): ", "10.0", float)

    
    # Run the side-by-side demo
    demo.run_side_by_side_demo(games, speed)
    
    print("👋 Demo completed!")


if __name__ == "__main__":
    main() 