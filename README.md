# Enhanced FedProx

The repo is a enhanced version of the [FedProx](https://github.com/litian96/FedProx) project, with several improvements to boost performance and adaptability in federated learning scenarios.

## Overview

Enhanced FedProx builds upon the original FedProx framework with the following key improvements:

- **Dynamic Aggregation Mechanism**: Adjusts aggregation weights based on client model similarity
- **Enhanced Dataset Support**: Added support for enhanced MNIST dataset
- **Improved Client Selection Strategy**: Intelligent client selection based on historical performance
- **Adaptive Learning Rate**: Automatically adjusts learning rates according to training progress

## Installation

### Key Requirements
- Python 3.12.9
- TensorFlow 2.19
- NumPy 2.1.3
- Matplotlib 3.10.1

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/FedProx_enhanced.git
cd FedProx_enhanced

# Create a virtual environment (optional)
conda create -n fedprox python=3.12

# Install dependencies
pip install -r requirements.txt

# Install TensorFlow with CUDA support
python3 -m pip install 'tensorflow[and-cuda]'
```

## Usage
### Running Enhanced Experiments
To run enhanced experiments, use the following command:
```bash
bash run_fedprox.sh [dataset] [drop_percent] [mu]
```
- `dataset`: The name of the dataset to use.
- `drop_percent`: The percentage of clients to drop.
- `mu`: The regularization strength.

Example:
```bash
bash run_fedprox.sh mnist 0.9 1
```
### Running Original FedProx Experiments
To run original FedProx experiments, use the following command:
```bash
bash run_fedprox_ori.sh [dataset] [drop_percent]
```

## Project Structure
FedProx_enhanced/
├── data/                   # Dataset directory
├── flearn/                 # Core federated learning implementation
│   ├── models/             # Model definitions
│   ├── trainers/           # Trainer implementations
│   ├── optimizer/          # Optimizer implementations
│   └── utils/              # Utility functions
├── results/                # Directory for experiment results
├── utils/                  # Helper scripts
├── main.py                 # Main program entry which can run both original and enhanced experiments
├── run_fedprox_ori.sh      # Script for running original FedProx experiments
└── run_fedprox.sh          # Script for running enhanced FedProx experiments

