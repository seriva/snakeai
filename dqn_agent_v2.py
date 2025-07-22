"""
Enhanced DQN (Deep Q-Network) Snake AI Agent - Version 2.

Improved implementation of Deep Q-Learning with multiple enhancements over V1
for better performance and training stability.

Key Improvements over V1:
- **Enhanced Neural Network**: Larger architecture (256 hidden units, 2 layers) 
  with dropout regularization vs V1's simple 64-unit single layer
- **Optimized Hyperparameters**: Better learning rate (0.003), higher discount 
  factor (0.99), slower epsilon decay, and more frequent learning updates
- **Learning Rate Scheduling**: Adaptive learning rate that decreases over time
  for improved convergence stability  
- **Reward Shaping**: Distance-based rewards for moving toward food, survival
  bonuses, and length-based bonuses to accelerate learning
- **Enhanced Training**: Dual step methods (basic + enhanced with reward shaping)
  and improved performance tracking with learning rate monitoring

Features:
- Enhanced Deep Q-Network with larger MLP architecture and regularization
- Reward-shaped experience replay with survival and distance bonuses
- Adaptive learning rate scheduling for training stability
- 16-feature state representation with immediate danger detection
- Epsilon-greedy exploration with optimized decay schedule
- Soft target network updates with improved update frequency

This version significantly outperforms V1 through architectural improvements,
better hyperparameter tuning, and reward engineering techniques.
"""

import json
import os
import random
from collections import deque
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from headless_game import HeadlessGame


class SimpleDQN(nn.Module):
    """
    Enhanced Deep Q-Network with bigger multi-layer perceptron architecture.
    
    Architecture: Input → Hidden1 (256) → Hidden2 (256) → Output (4 actions)
    Includes dropout for regularization.
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(SimpleDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # NEW: Additional layer
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)  # NEW: Dropout for regularization

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  # NEW: Additional layer
        x = self.dropout(x)
        return self.fc3(x)


class ReplayBuffer:
    """
    Basic experience replay buffer for storing and sampling past experiences.
    
    Stores transitions (state, action, reward, next_state, done) and samples
    uniformly at random for training stability.
    """
    
    def __init__(self, capacity: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.memory: List = []

    def push(self, experience: Tuple) -> None:
        """Store a new experience, removing oldest if at capacity."""
        self.memory.append(experience)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)  # Remove oldest experience

    def sample(self, batch_size: int) -> Tuple:
        """Sample a random batch of experiences for training."""
        experiences = random.sample(self.memory, k=batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(np.vstack([e[0] for e in experiences])).to(self.device)
        actions = torch.LongTensor(np.vstack([e[1] for e in experiences])).to(self.device)
        rewards = torch.FloatTensor(np.vstack([e[2] for e in experiences])).to(self.device)
        next_states = torch.FloatTensor(np.vstack([e[3] for e in experiences])).to(self.device)
        dones = torch.FloatTensor(np.vstack([e[4] for e in experiences])).to(self.device)
        
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.memory)


class DQNAgentV2:
    """
    Enhanced DQN Snake AI agent implementing Deep Q-Learning (Version 2).
    
    Enhanced version with bigger network, reward shaping, learning rate scheduling,
    and improved hyperparameters for better performance.
    """
    
    # Training hyperparameters
    LEARNING_RATE = 0.003      # Reduced from 0.01
    GAMMA = 0.99              # Increased from 0.95
    EPSILON_START = 1.0
    EPSILON_DECAY = 0.9995    # Slower decay (was 0.995)
    EPSILON_END = 0.02        # Higher minimum (was 0.01)
    BATCH_SIZE = 64           # Reduced from 128 for more frequent updates
    BUFFER_SIZE = 100_000     # Reduced from 200_000 
    LEARNING_FREQ = 2         # More frequent learning (was 4)
    SOFT_UPDATE_TAU = 5e-3    # Slower target updates (was 1e-2)
    
    def __init__(self, state_size: int = 16, action_size: int = 4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        
        # Neural networks
        self.q_local = SimpleDQN(state_size, action_size).to(self.device)
        self.q_target = SimpleDQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_local.parameters(), lr=self.LEARNING_RATE)
        
        # NEW: Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=1000,  # Reduce LR every 1000 learning steps
            gamma=0.95       # Multiply LR by 0.95 each time
        )
        
        # Experience replay
        self.memory = ReplayBuffer(self.BUFFER_SIZE)
        
        # Training state
        self.step_count = 0
        
        # Performance tracking
        self.scores = deque(maxlen=100)
        self.best_score = 0
        self.epsilon = self.EPSILON_START
        
        # NEW: Learning rate tracking
        self.learning_rates = deque(maxlen=100)
        
        # NEW: Track previous state for reward shaping
        self.previous_distance_to_food = None
        self.previous_snake_length = 1

    # ==================== STATE REPRESENTATION ====================
    
    def get_state(self, game: HeadlessGame) -> np.ndarray:
        """
        Extract 16-feature state representation from game.
        
        Features (16 total):
        - Danger detection (8): Collision risk in 8 directions around head
        - Current direction (4): One-hot encoded movement direction
        - Food location (4): Binary indicators for food position relative to head
        """
        head_x, head_y = game.engine.snake_head
        food_x, food_y = game.engine.apple_position
        block_width = game.engine.BLOCK_WIDTH

        # Define 8 points around the snake head for danger detection
        danger_points = [
            (head_x - block_width, head_y),      # left
            (head_x + block_width, head_y),      # right
            (head_x, head_y - block_width),      # up
            (head_x, head_y + block_width),      # down
            (head_x - block_width, head_y - block_width),  # left-up
            (head_x - block_width, head_y + block_width),  # left-down
            (head_x + block_width, head_y - block_width),  # right-up
            (head_x + block_width, head_y + block_width),  # right-down
        ]

        # Check for danger in each direction
        danger_features = [self._is_collision(game, point) for point in danger_points]
        
        # Current movement direction (one-hot encoded)
        direction_features = [
            game.engine.snake_direction == direction
            for direction in ["left", "right", "up", "down"]
        ]
        
        # Food location relative to snake head
        food_features = [
            food_x < head_x,  # food left
            food_x > head_x,  # food right
            food_y < head_y,  # food up
            food_y > head_y,  # food down (fixed from original bug)
        ]

        # Combine all features
        state = danger_features + direction_features + food_features
        return np.array(state, dtype=int)

    def _is_collision(self, game: HeadlessGame, position: Tuple[int, int]) -> bool:
        """Check if a position would result in collision (wall or snake body)."""
        x, y = position
        
        # Wall collision
        if (x < 0 or x >= game.engine.SCREEN_SIZE or 
            y < 0 or y >= game.engine.SCREEN_SIZE):
            return True
        
        # Snake body collision
        return position in game.engine.snake_positions

    # ==================== ACTION SELECTION ====================
    
    def get_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """Select action using epsilon-greedy policy."""
        eps = epsilon if epsilon is not None else self.epsilon
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get Q-values from network
        self.q_local.eval()
        with torch.no_grad():
            action_values = self.q_local(state_tensor)
        self.q_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return action_values.argmax().item()  # Greedy action
        else:
            return random.randint(0, 3)  # Random action

    # ==================== REWARD ENHANCEMENT ====================
    
    def _calculate_enhanced_reward(self, game: HeadlessGame, base_reward: float, 
                                 prev_state: np.ndarray, current_state: np.ndarray) -> float:
        """Calculate enhanced reward with distance-based shaping and survival bonuses."""
        enhanced_reward = base_reward
        
        # Get current game state info
        head_x, head_y = game.engine.snake_head
        food_x, food_y = game.engine.apple_position
        current_length = len(game.engine.snake_positions)
        
        # Handle different reward scenarios
        if abs(base_reward - (-10)) < 0.001:  # Game over (collision reward)
            return enhanced_reward
            
        if abs(base_reward - 10) < 0.001:  # Got food
            # Bonus scales with current score/length
            length_bonus = 0.5 * current_length  # Using direct value since REWARDS might not be accessible
            enhanced_reward = base_reward + length_bonus
            # Reset distance tracking after eating food
            self.previous_distance_to_food = None
            self.previous_snake_length = current_length
            return enhanced_reward
        
        # Distance-based reward shaping for normal steps
        current_distance = abs(head_x - food_x) + abs(head_y - food_y)  # Manhattan distance
        
        # Add survival bonus
        enhanced_reward += 0.005
        
        # Distance-based reward (only if we have previous distance)
        if self.previous_distance_to_food is not None:
            if current_distance < self.previous_distance_to_food:
                # Moving closer to food
                enhanced_reward += 0.1
            elif current_distance > self.previous_distance_to_food:
                # Moving farther from food
                enhanced_reward += -0.1
        
        # Update distance tracking
        self.previous_distance_to_food = current_distance
        
        return enhanced_reward

    # ==================== LEARNING ====================
    
    def step(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """Enhanced step with reward shaping."""
        # Store experience in replay buffer with original reward first
        self.memory.push((state, action, reward, next_state, done))
        
        self.step_count += 1
        
        # Learn every few steps if we have enough experiences
        if (self.step_count % self.LEARNING_FREQ == 0 and 
            len(self.memory) > self.BATCH_SIZE):
            experiences = self.memory.sample(self.BATCH_SIZE)
            self._learn(experiences)

    def enhanced_step(self, game: HeadlessGame, state: np.ndarray, action: int, 
                     reward: float, next_state: np.ndarray, done: bool) -> None:
        """Enhanced step with reward shaping - use this for training."""
        # Calculate enhanced reward
        enhanced_reward = self._calculate_enhanced_reward(game, reward, state, next_state)
        
        # Store experience with enhanced reward
        self.memory.push((state, action, enhanced_reward, next_state, done))
        
        self.step_count += 1
        
        # Learn every few steps if we have enough experiences
        if (self.step_count % self.LEARNING_FREQ == 0 and 
            len(self.memory) > self.BATCH_SIZE):
            experiences = self.memory.sample(self.BATCH_SIZE)
            self._learn(experiences)

    def _learn(self, experiences: Tuple) -> None:
        """Enhanced learning with learning rate scheduling."""
        states, actions, rewards, next_states, dones = experiences
        
        # Get expected Q values from local model
        q_expected = self.q_local(states).gather(1, actions)
        
        # Get max predicted Q values for next states from target model
        q_targets_next = self.q_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute target Q values for current states
        q_targets = rewards + (self.GAMMA * q_targets_next * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # NEW: Step the learning rate scheduler
        self.scheduler.step()
        
        # NEW: Track current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)
        
        # Update target network
        self._soft_update_target_network()

    def _soft_update_target_network(self) -> None:
        """Soft update target network parameters using τ (tau)."""
        for target_param, local_param in zip(self.q_target.parameters(), self.q_local.parameters()):
            target_param.data.copy_(
                self.SOFT_UPDATE_TAU * local_param.data + 
                (1.0 - self.SOFT_UPDATE_TAU) * target_param.data
            )

    # ==================== MODEL PERSISTENCE ====================
    
    def save_model(self, model_path: str = "model/dqn_agent_v2.pth", 
                   data_path: str = "model/dqn_agent_v2_data.json") -> None:
        """Save the trained model and training data."""
        # Create model directory
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model state
        torch.save(self.q_local.state_dict(), model_path)
        
        # Save training data
        data = {
            'best_score': self.best_score,
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }
        with open(data_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_model(self, model_path: str = "model/dqn_agent_v2.pth",
                   data_path: str = "model/dqn_agent_v2_data.json") -> bool:
        """Load a trained model and training data."""
        # Check if model file exists
        if not os.path.exists(model_path):
            return False
        
        # Load model state
        self.q_local.load_state_dict(torch.load(model_path))
        self.q_target.load_state_dict(torch.load(model_path))  # Sync target network
        
        # Load training data if available
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                data = json.load(f)
                self.best_score = data.get('best_score', 0)
                self.epsilon = data.get('epsilon', self.EPSILON_START)
                self.step_count = data.get('step_count', 0)
        
        print(f"✅ Model loaded! Best score: {self.best_score}, Epsilon: {self.epsilon:.3f}")
        return True


def train_dqn_agent(episodes: int = 10_000) -> None:
    """
    Train the DQN Snake agent V2.
    
    Args:
        episodes: Number of training episodes to run
    """
    print("🐍 Starting DQN Snake AI Training (V2)")
    print("="*50)
    
    # Initialize game and agent
    game = HeadlessGame()
    agent = DQNAgentV2()
    
    # Try to load existing model
    agent.load_model()
    
    print(f"📱 Device: {agent.device}")
    print(f"🎯 Episodes: {episodes:,}")
    print(f"🧠 Network: Simple MLP ({sum(p.numel() for p in agent.q_local.parameters()):,} params)")
    print(f"💾 Experience Replay: Basic ({agent.BUFFER_SIZE:,} capacity)")
    print(f"📊 State features: {agent.state_size}")
    print("="*50)
    
    # Training loop
    scores_window = deque(maxlen=100)
    
    for episode in range(episodes):
        game.reset_game()
        state = agent.get_state(game)
        total_reward = 0
        steps = 0
        
        # Episode loop
        while not game.game_over and steps < 200_000:  # Max steps per episode
            # Select and execute action
            action = agent.get_action(state)
            move = np.zeros(4)
            move[action] = 1
            
            reward, done, score = game.step(move)
            next_state = agent.get_state(game)
            
            # Learn from experience
            agent.enhanced_step(game, state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Update tracking
        scores_window.append(score)
        agent.scores.append(score)
        agent.best_score = max(agent.best_score, score)
        agent.epsilon = max(agent.EPSILON_END, agent.EPSILON_DECAY * agent.epsilon)
        
        # Periodic saving and reporting
        if episode % 100 == 0:
            agent.save_model()
        
        if episode % 50 == 0:
            avg_score = np.mean(scores_window)
            current_lr = agent.optimizer.param_groups[0]['lr']  # NEW: Get current LR
            print(f"Episode {episode:5d} | "
                  f"Score: {score:3d} | "
                  f"Best: {agent.best_score:3d} | "
                  f"Avg: {avg_score:5.1f} | "
                  f"Steps: {steps:4d} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"LR: {current_lr:.2e}")  # NEW: Show learning rate
    
    # Final save
    agent.save_model()
    print("\n🎉 Training completed!")
    print(f"🏆 Best score achieved: {agent.best_score}")


if __name__ == "__main__":
    train_dqn_agent()
