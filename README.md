# Sigmoid Colon

<p align="center">
    <a href="https://github.com/poopingface/sigmoidcolon/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/poopingface/sigmoidcolon.svg?color=blue">
    </a>
    <a href="https://poopingface.github.io/sigmoidcolon">
        <img alt="Website" src="https://img.shields.io/website/http/poopingface.github.io/sigmoidcolon/index.html?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/poopingface/sigmoidcolon/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/poopingface/sigmoidcolon.svg">
    </a>
</p>

<img src="docs/fit.png" height=256><img src="extras/mnist_accuracy.png" height=256>

The biologically inspired activation function. [Read our (toilet) paper](https://poopingface.github.io/sigmoidcolon).

## Installation

Install with pip:

```bash
pip install sigmoidcolon
```

## Usage

Class version

```python
from torch import nn
from sigmoidcolon import SigmoidColon

model = nn.Sequential(
    nn.Linear(784, 64),
    SigmoidColon()
    nn.Linear(64, 10)
)
```

Functional version

```python
from torch import nn
from sigmoidcolon.functional import sigmoidcolon

x = torch.randn(100)
y = sigmoidcolon(x)
```

## Development

Not satisfied with the function? Redo it with:

```bash
# install dev requirements
pip install Pillow numpy matplotlib

# run code generation tool
python scripts/codegen.py docs/sigmoidcolon.png
```

