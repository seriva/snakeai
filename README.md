## About

A Snake game AI project featuring two implementations of Deep Q-Network (DQN) algorithms. The project compares the original DQN (V1) with an enhanced version (V2) that includes improved hyperparameters, reward shaping, and architectural enhancements. Built with PyTorch.

**References:**
- [Original Implementation for the DQN V1 approach](https://github.com/SumitJainUTD/pytorch-ann-snake-game-ai)
- DQN Paper: "Human-level control through deep reinforcement learning" (Mnih et al., 2015)

**Performance**: Current V2 model performs 72.5% better than the V1 model as compared in 1000 games played.

## Features

- **Play Snake**: Manual gameplay with `game.py`
- **Two AI Agents**: DQN V1 (original) and DQN V2 (enhanced) implementations
- **Side-by-Side Demo**: Watch both agents play simultaneously for direct comparison
- **Training**: Train both AI agents with different approaches
- **Model Saving**: Save and load trained models

## Tech Stack

- **Language**: Python
- **ML Framework**: PyTorch
- **Package Manager**: uv
- **Dependencies**: PyTorch, NumPy, Pygame
- **Architecture**: Deep Q-Network (DQN) algorithms
- **Training**: Reinforcement learning with neural networks

## Project Structure

```
snake-ai/
├── ai_demo.py              # Watch your AI agents play (V1 vs V2)
├── dqn_agent_v1.py         # Original DQN implementation
├── dqn_agent_v2.py         # Enhanced DQN implementation  
├── game.py                 # Play the game yourself
├── headless_game.py        # Fast training without graphics
├── snake_engine.py         # Core game logic
├── base.py                 # Game settings
├── model/                  # Trained AI models
│   ├── dqn_agent_v1.pth                # DQN V1 model
│   ├── dqn_agent_v1_data.json          # DQN V1 training data
│   ├── dqn_agent_v2.pth                # DQN V2 model
│   └── dqn_agent_v2_data.json          # DQN V2 training data
└── pyproject.toml          # Dependencies
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/seriva/snakeai.git
cd snakeai

# Install dependencies
uv sync
```

### 1. Play the Game Yourself
```bash
uv run game.py
```

### 2. Train the AI Agents

**Train DQN V1 (Original):**
```bash
uv run dqn_agent_v1.py
```

**Train DQN V2 (Enhanced):**
```bash
uv run dqn_agent_v2.py
```

### 3. Watch the AI Play
```bash
uv run ai_demo.py
```
Watch both agents play side-by-side for direct comparison!

## AI Agents

### DQN V1 (Original)
The classic Deep Q-Network approach that learns which actions are good in each game situation.

- Simple neural network (16 inputs → 64 hidden → 4 outputs)
- Basic hyperparameters and reward system
- Standard epsilon-greedy exploration strategy

### DQN V2 (Enhanced) 
An improved version of the standard DQN with several enhancements:

- Deeper neural network (16 inputs → 256 hidden → 256 hidden → 4 outputs)
- Advanced reward shaping (distance-based rewards, survival bonuses)
- Improved hyperparameters and learning rate scheduling
- Better training stability and performance

## What the AI Sees

The AI gets 16 pieces of information about the game:

1. **Danger Detection (8 features)**: Is there danger in each of 8 directions around the snake head?
2. **Current Direction (4 features)**: Which way is the snake currently moving?
3. **Food Location (4 features)**: Which direction is the food relative to the snake head?

Both agents use this same information to make decisions.

## Training

### How the AI Learns
- Plays thousands of games and remembers what happened
- Gets rewards for eating food (+10) and penalties for dying (-10)
- Starts by exploring randomly, gradually gets smarter
- Updates its strategy based on past experiences

### Training Time
- Each agent trains for about 10,000 games
- Takes about 2-3 hours on a modern computer
- Progress is saved automatically so you can stop and resume
