# Sigmoid Colon

The biologically inspired activation function.

## Usage

Class version

```python
from torch import nn
from sigmoidcolon import SigmoidColon

model = nn.Sequential(
    nn.Linear(784, 64),
    Sigmoid()
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
python scripts/codegen.py sigmoidcolon.png
```

