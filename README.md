# Hands-on TensorFlow & NumPy

A practical guide to learning and using TensorFlow and NumPy for data manipulation and machine learning.

## Overview
This repository contains examples, exercises, and tutorials that demonstrate how to use TensorFlow and NumPy together effectively for data science and deep learning tasks.

## Contents
- Basic tensor operations
- NumPy integration with TensorFlow
- Data preprocessing techniques
- Simple neural networks
- Performance comparison between TensorFlow and NumPy

## Getting Started

### Option 1: With TensorFlow (Python 3.10)
```bash
# Create Python 3.10 virtual environment
python3.10 -m venv venv_tf

# Activate virtual environment
source venv_tf/bin/activate  # On Unix/macOS
# or
.\venv_tf\Scripts\activate   # On Windows

# Install requirements
pip install tensorflow numpy matplotlib pandas scikit-learn jupyter

# Run example
python examples/basic_operations.py
```

### Option 2: Without TensorFlow (any Python version)
This repository includes a TensorFlow substitute that provides basic functionality using only NumPy.

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate   # On Windows

# Install requirements (excluding TensorFlow)
pip install numpy matplotlib pandas scikit-learn jupyter

# Run example with TensorFlow substitute
python examples/basic_operations_substitute.py
```

## Running Jupyter Notebooks
```bash
# Activate your chosen virtual environment
source venv_tf/bin/activate  # With TensorFlow
# or
source venv/bin/activate     # Without TensorFlow

# Start Jupyter
jupyter notebook
```

## License
MIT