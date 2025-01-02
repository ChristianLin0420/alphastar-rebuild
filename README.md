# AlphaStar Implementation in PyTorch

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/0*nGKjNfQ3KScc5uia.jpg" alt="AlphaStar Logo"/>
</p>

A comprehensive PyTorch implementation of DeepMind's AlphaStar, an AI system that achieved Grandmaster level in StarCraft II. This project reimagines the groundbreaking architecture with modern PyTorch practices, providing researchers and enthusiasts with a robust foundation for StarCraft II AI development.

## 🌟 Overview

This implementation brings together several cutting-edge components:

- **Neural Architecture**: Complete reproduction of the AlphaStar neural network architecture
- **Learning Pipeline**: Both supervised learning from human replays and reinforcement learning
- **Battle-tested Components**: Thoroughly validated implementation of key mechanisms
- **Research Ready**: Modular design enabling easy experimentation and extension
- **Production Quality**: Comprehensive testing, logging, and visualization tools

## 🏗️ Project Structure

Our codebase is organized for clarity and maintainability:

```
alphastar/
├── configs/
│   └── training_config.py    # Centralized configuration management
├── data/
│   ├── datamodule.py        # Efficient data pipeline handling
│   └── preprocessor.py      # StarCraft II state processing
├── models/
│   ├── action_heads.py      # Strategic decision making
│   ├── alphastar.py        # Core architecture integration
│   ├── core.py             # Temporal reasoning module
│   ├── entity_encoder.py   # Unit and building processing
│   ├── scalar_encoder.py   # Game state analysis
│   └── spatial_encoder.py  # Tactical map understanding
├── utils/
│   ├── checkpointing.py    # Training state management
│   ├── logger.py          # Comprehensive metric tracking
│   └── validation.py      # Input safeguarding
├── tests/                 # Quality assurance
└── scripts/
    └── train_alphastar.py # Training orchestration
```

## 🛠️ Requirements

### Core Dependencies
Carefully selected versions ensuring compatibility and performance:
```
torch>=1.9.0      # Deep learning framework
pysc2>=3.0.0      # StarCraft II interface
numpy>=1.19.0     # Numerical computations
wandb>=0.15.0     # Experiment tracking
tqdm>=4.60.0      # Progress tracking
pytest>=6.2.0     # Testing framework
```

### System Requirements
Recommended specifications for optimal training performance:
- 🖥️ CUDA-capable GPU (RTX 3080 or better recommended)
- 💾 32GB RAM (minimum for large-scale training)
- 💽 100GB SSD space (for game installation and replay data)

## 🚀 Installation

Follow these steps to set up your development environment:

1. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/alphastar-pytorch.git
   cd alphastar-pytorch
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install StarCraft II**
   ```bash
   bash scripts/install_sc2.sh
   python -m pysc2.bin.download_maps
   ```

## 🧠 Model Architecture

Our implementation features three main processing stages:

### 1. Input Processing
- **Scalar Encoder**: Processes global game state features
  - Resource management
  - Population metrics
  - Game progression indicators
- **Entity Encoder**: Transformer-based unit analysis
  - Unit type embedding
  - Positional encoding
  - Multi-head attention
- **Spatial Encoder**: ResNet-based map understanding
  - Terrain analysis
  - Resource location processing
  - Unit positioning

### 2. Core Processing
- **LSTM-based Integration**
  - Temporal state tracking
  - Strategic planning
  - Memory management
- **Advanced Features**
  - Layer normalization for stable training
  - Residual connections for deep network training
  - Dropout for regularization

### 3. Action Generation
- **Strategic Decision Making**
  - Action type selection
  - Unit selection via pointer network
  - Target location prediction
  - Value estimation for policy improvement

## 🎮 Usage

### Training Pipeline

1. **Basic Training**
   ```bash
   python -m scripts.train_alphastar
   ```

2. **Resume Training**
   ```bash
   python -m scripts.train_alphastar --resume checkpoints/alphastar_step_1000.pt
   ```

3. **Custom Configuration**
   ```bash
   python -m scripts.train_alphastar --config configs/custom_config.yaml
   ```

### Model Evaluation
```bash
# Evaluate the best model
python -m scripts.train_alphastar --evaluate checkpoints/alphastar_best.pt

# Evaluate with custom settings
python -m scripts.train_alphastar --evaluate checkpoints/model.pt --num-games 100
```

### Monitoring Training

Training progress is tracked using Weights & Biases (wandb):
```bash
# Basic training with wandb logging
python -m scripts.train_alphastar --wandb-project alphastar

# Specify wandb entity and run name
python -m scripts.train_alphastar --wandb-project alphastar --wandb-entity your-username --wandb-run-name experiment-1
```

Key metrics tracked:
- Loss curves (policy, value, auxiliary tasks)
- Action distributions
- Win rates
- Resource management statistics
- Unit production patterns
- Model architecture visualization
- Training hyperparameters
- System metrics (GPU usage, memory)

### Configuration

Fine-tune your training with these key parameters:

```python
# Architecture Configuration
SCALAR_INPUT_DIM = 9        # Game state features
ENTITY_INPUT_DIM = 10       # Unit properties
SPATIAL_INPUT_CHANNELS = 32 # Map information
NUM_ACTIONS = 1000          # Action space size

# Training Hyperparameters
BATCH_SIZE = 512           # Training batch size
LEARNING_RATE = 3e-4       # Adam optimizer LR
NUM_EPOCHS = 100          # Training duration
GRADIENT_CLIP = 1.0       # Gradient stability

# PPO Hyperparameters
PPO_CLIP = 0.2           # Policy update constraint
VALUE_LOSS_COEF = 0.5    # Value function weight
ENTROPY_COEF = 0.01      # Exploration promotion
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Document all public functions and classes

### Monitoring

Launch TensorBoard:
```bash
tensorboard --logdir logs/
```

Available metrics:
- Training loss curves
- Action distribution
- Value predictions
- Learning rate
- Gradient norms

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add/update tests
5. Submit a pull request

Please ensure:
- All tests pass
- Code is well-documented
- Changes are backward compatible

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{alphastar-pytorch,
    author = {Your Name},
    title = {AlphaStar PyTorch Implementation},
    year = {2024},
    publisher = {GitHub},
    url = {https://github.com/yourusername/alphastar-pytorch}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- DeepMind's AlphaStar team for their groundbreaking research
- PyTorch team for the excellent deep learning framework
- StarCraft II and PySC2 communities

## Contact

- Issues: Use GitHub Issues
- Email: your.email@example.com
- Twitter: @yourusername

## Roadmap

- [ ] Implement multi-agent training
- [ ] Add more architectures
- [ ] Improve documentation
- [ ] Add visualization tools
- [ ] Support for custom reward functions
