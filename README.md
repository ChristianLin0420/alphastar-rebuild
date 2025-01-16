# AlphaStar Implementation in PyTorch

A PyTorch implementation of DeepMind's AlphaStar, an AI system for playing StarCraft II. This project focuses on reimplementing the core neural network architecture and training pipeline, with an emphasis on modularity, testability, and research extensibility.

## 🎯 Introduction

AlphaStar is a groundbreaking AI system that achieved Grandmaster level in StarCraft II, a complex real-time strategy game. This implementation focuses on:

- **Neural Architecture**: A faithful reproduction of the core AlphaStar architecture, including:
  - Multi-modal feature processing (scalar, entity, spatial)
  - Transformer-based entity processing
  - LSTM-based temporal reasoning
  - Auto-regressive action generation

- **Training Pipeline**: Support for both supervised and reinforcement learning:
  - Supervised learning from demonstration data
  - Reinforcement learning with PPO
  - Prioritized experience replay
  - League training infrastructure

- **Research Focus**: Built for AI research with:
  - Modular component design
  - Comprehensive test coverage
  - Configurable architecture
  - Detailed logging and monitoring

- **Quality Assurance**: Emphasis on code quality through:
  - Extensive unit tests
  - Integration tests
  - Type hints
  - Documentation

## 🌟 Overview

This implementation includes:

- **Neural Architecture**: Core AlphaStar neural network components
- **Training Pipeline**: Both supervised and reinforcement learning capabilities
- **Modular Design**: Easily extensible architecture for research
- **Testing Framework**: Comprehensive test suite for all components

## 🏗️ Project Structure

```
alphastar-rebuild/
├── configs/
│   ├── default.yaml         # Default configuration
│   ├── config.py           # Configuration loading utilities
│   └── training_config.py  # Training parameters
├── models/
│   ├── modules/
│   │   ├── action_heads.py    # Action selection components
│   │   ├── core.py           # LSTM-based temporal processing
│   │   ├── entity_encoder.py # Entity feature processing
│   │   ├── scalar_encoder.py # Scalar feature processing
│   │   └── spatial_encoder.py # Spatial feature processing
│   └── alphastar.py        # Main model architecture
├── train/
│   └── train.py           # Training loop implementation
├── utils/
│   ├── replay_buffer.py   # Experience replay implementations
│   └── checkpointing.py   # Model checkpointing
├── tests/                 # Comprehensive test suite
│   ├── test_action_heads.py
│   ├── test_alphastar.py
│   ├── test_core.py
│   ├── test_entity_encoder.py
│   ├── test_preprocessor.py
│   ├── test_scalar_encoder.py
│   └── test_spatial_encoder.py
└── scripts/
    ├── train_alphastar.py # Training script
    └── run_tests.py       # Test runner
```

## 🧠 Model Architecture

### Core Components

1. **Feature Encoders**
   - **Scalar Encoder**: Processes global game state features
   - **Entity Encoder**: Transformer-based unit processing
   - **Spatial Encoder**: Processes map information using ResNet blocks

2. **Core Processing**
   - LSTM-based temporal processing
   - State tracking and memory management
   - Integration of different feature streams

3. **Action Generation**
   - **Action Type Head**: Selects action categories
   - **Pointer Network**: Unit selection mechanism
   - **Spatial Action Head**: Location-based actions
   - **Value Head**: State value estimation

### Key Features

- Multi-head attention for entity processing
- Residual connections in spatial processing
- Auto-regressive action selection
- Prioritized experience replay
- Both supervised and reinforcement learning support

## 🛠️ Requirements

```
# Core Dependencies
torch>=2.0.0
numpy>=1.24.0
pytest>=7.3.0
pytest-cov>=4.1.0
wandb>=0.15.0
pyyaml>=6.0
protobuf<=3.20.0
pysc2>=3.0.0
```

## 🚀 Installation

1. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/alphastar-rebuild.git
   cd alphastar-rebuild
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🧪 Testing Framework

The project includes a comprehensive testing suite covering all components:

### Test Structure

```
tests/
├── conftest.py              # Shared test fixtures
├── test_action_heads.py     # Action selection tests
├── test_alphastar.py       # Full model integration tests
├── test_core.py            # LSTM processing tests
├── test_entity_encoder.py  # Entity processing tests
├── test_preprocessor.py    # Input preprocessing tests
├── test_scalar_encoder.py  # Scalar feature tests
└── test_spatial_encoder.py # Spatial feature tests
```

### Test Categories

1. **Unit Tests**
   - Individual component functionality
   - Input/output shapes
   - Value ranges and constraints
   - Edge cases

2. **Integration Tests**
   - Component interactions
   - Full model forward/backward passes
   - Training loop functionality
   - Configuration loading

3. **Property Tests**
   - Gradient flow
   - Memory usage
   - Numerical stability
   - Batch size handling

### Running Tests

```bash
# Basic test execution
python scripts/run_tests.py

# Test with options
python scripts/run_tests.py [options]

Options:
  --verbose, -v         Enable verbose output
  --coverage, -c        Generate coverage report
  --html               Generate HTML coverage report
  --test-path          Path to test directory (default: tests)
  --pattern            Pattern to match test files (default: test_*)
  --workers, -n        Number of workers for parallel execution
```

Example Usage:
```bash
# Run specific test file
python scripts/run_tests.py --pattern "test_action_heads.py"

# Run with coverage and HTML report
python scripts/run_tests.py --coverage --html

# Run in parallel with 4 workers
python scripts/run_tests.py --workers 4
```

## 🎮 Training Arguments

The training script (`scripts/train_alphastar.py`) supports various arguments for customization:

```bash
python scripts/train_alphastar.py [options]

Required Arguments:
  --config              Path to YAML configuration file

Optional Arguments:
  --checkpoint         Path to checkpoint to resume from
  --mode              Training mode: 'supervised' or 'reinforcement'
  --wandb-project     Weights & Biases project name
  --wandb-entity      Weights & Biases username/team
  --wandb-run-name    Weights & Biases run name
  --device            Device to train on (cuda/cpu)
  --debug             Enable debug logging
```

Example Usage:
```bash
# Basic supervised training
python scripts/train_alphastar.py \
  --config configs/default.yaml \
  --mode supervised

# Resume training from checkpoint
python scripts/train_alphastar.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/model_1000.pt \
  --mode reinforcement

# Training with W&B logging
python scripts/train_alphastar.py \
  --config configs/default.yaml \
  --wandb-project alphastar \
  --wandb-entity your-username \
  --wandb-run-name experiment-1
```

## 📊 Monitoring and Logging

The training process can be monitored through various means:

### Weights & Biases Integration
```bash
# Enable W&B logging
python scripts/train_alphastar.py \
  --config configs/default.yaml \
  --wandb-project alphastar
```

Tracked Metrics:
- Loss curves (policy, value, auxiliary)
- Action distributions
- Network gradients
- Resource usage
- Model architecture
- Hyperparameters

### Console Logging
The training script provides detailed console output:
```
[INFO] Epoch 1/100
[INFO] Training loss: 2.345
[INFO] Validation win rate: 0.456
[INFO] Saving checkpoint: model_1000.pt
```

## 🔧 Configuration

The project uses a hierarchical configuration system:

### Training Configuration (training_config.py)
```python
# Model dimensions
SCALAR_INPUT_DIM = 32
ENTITY_INPUT_DIM = 64
SPATIAL_INPUT_DIM = 16
MAX_ENTITIES = 100
SPATIAL_SIZE = (64, 64)

# Network parameters
TRANSFORMER_DIM = 64
TRANSFORMER_HEADS = 2
TRANSFORMER_LAYERS = 2
LSTM_DIM = 128
LSTM_LAYERS = 2

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
```

### Default Configuration (default.yaml)
Includes comprehensive settings for:
- Input dimensions
- Network architecture
- Action space configuration
- Training parameters
- RL parameters
- League training settings
- Environment configuration

## 🧪 Testing

The project includes extensive tests for all components:

- **Action Heads**: Tests for action selection mechanisms
- **Core Processing**: LSTM and temporal processing tests
- **Encoders**: Tests for all feature encoders
- **Full Model**: Integration tests for the complete architecture

## 📈 Future Work

- [ ] Complete SC2 environment integration
- [ ] Implement league training
- [ ] Add model visualization tools
- [ ] Expand test coverage
- [ ] Add performance benchmarks
- [ ] Implement distributed training

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📚 References

- [AlphaStar: Mastering the Real-Time Strategy Game StarCraft II](https://deepmind.com/blog/article/alphastar-mastering-real-time-strategy-game-starcraft-ii)
- [PySC2: StarCraft II Learning Environment](https://github.com/deepmind/pysc2)

