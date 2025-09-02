# TrajGNN: Trajectory Graph Neural Network

A deep learning model for trajectory representation learning and prediction using Graph Neural Networks.

## Features

- Graph-based trajectory representation
- Spatio-temporal embedding
- Support for both supervised and self-supervised learning
- Flexible architecture for various trajectory analysis tasks

## Model Architecture

The model consists of three main components:

1. **Temporal Embedding**: Processes time transition vectors between node pairs
2. **Location Embedding**: Enhanced GAT (Graph Attention Network) for spatial features
3. **Spatio-Temporal Embedding**: Fusion of spatial and temporal features

## Installation

```bash
# Clone the repository
git clone https://github.com/TXFR/GraphRL.git
cd GraphRL

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```python
# Supervised learning
python train.py --supervised

# Self-supervised learning (default)
python train.py

# Mixed training
python train.py --self-supervised
```

### Testing

```python
python test_model.py
```

## Model Configuration

Key parameters can be configured in `config/model_config.py`:

- Time slots: 48 (30-minute intervals)
- Output hours: 24 (predictions per day)
- Location categories: 8 (types of places)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request