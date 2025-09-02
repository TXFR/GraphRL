# Jointly spatial-temporal representation learning for individual trajectories

## Model Architecture

The model consists of three main components:

1. **Temporal Embedding**: Processes time transition vectors between node pairs
2. **Location Embedding**: Enhanced GAT for spatial features with auxiliary information integration
3. **Spatio-Temporal Embedding**: Fusion of spatial and temporal features for joint representation learning

## Self-Supervised Learning Modes

GraphRL supports supervised learning and self-supervised learning modes for trajectory representation learning:

### NONE Mode (Supervised Learning Only)
- **Description**: Pure supervised learning using labeled trajectory data
- **Outputs**: `t_cls` (time proability), `l_cls` (location proability), `t_l_cls` (spatio-temporal proability)
- **Use Case**: When you have sufficient labeled trajectory data

### Contrastive Mode
- **Description**: Pure self-supervised contrastive learning that creates positive pairs through data augmentation to learn unique representations for each user's trajectory graph (completely label-free)
- **Supervision Outputs**: `t_cls`, `l_cls`, `t_l_cls`
- **SSL Outputs**: `contrastive` (128-dimensional graph-level representation)
- **Use Case**: Learning trajectory representations without any similarity labels

### Link Prediction Mode
- **Description**: Self-supervised learning by predicting edge relationships within trajectory graphs to understand spatio-temporal patterns
- **Supervision Outputs**: `t_cls`, `l_cls`, `t_l_cls`
- **SSL Outputs**:
  - `edge_pred` (edge existence probability predictions)
  - `edge_labels` (ground truth edge link)
- **Use Case**: Understanding structural patterns in trajectory graphs

### Combined Mode
- **Description**: Joint contrastive learning and link prediction for comprehensive representation learning
- **Supervision Outputs**: `t_cls`, `l_cls`, `t_l_cls`
- **SSL Outputs**:
  - `contrastive` (128-dimensional contrastive representation)
  - `edge_pred` (edge existence probability predictions)
  - `edge_labels` (ground truth edge link)
- **Use Case**: Best representation learning combining both contrastive and structural objectives

## Installation

```bash
# Clone the repository
git clone https://github.com/TXFR/GraphRL.git
cd GraphRL

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start with Test Mode

If you don't have a prepared dataset yet, you can start by running the test mode to understand the model structure and functionality. The test script automatically generates synthetic trajectory data and demonstrates all model components:

```bash
# Quick test to understand model structure
python test_model.py --all-modes        # Test all SSL modes with synthetic data
```

This helps you:
- ✅ Understand the model architecture and data flow
- ✅ Verify installation and dependencies
- ✅ Learn about different self-supervised learning modes
- ✅ Get familiar with output formats and loss components




### Training

```bash

# Self-supervised learning (recommended)
python train.py --ssl                    # Combined contrastive + link prediction
python train.py --ssl --ssl-weight 0.8   # Adjust SSL loss weight

# Legacy SSL modes (for compatibility)
python train.py --ssl-mode contrastive   # Contrastive learning only
python train.py --ssl-mode reconstruction # Link prediction only
python train.py --ssl-mode combined      # Combined mode (same as --ssl)

# Customize loss weights
python train.py --ssl --ssl-weight 0.7 --supervised-weight 0.2
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