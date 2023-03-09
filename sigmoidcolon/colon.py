import torch
import sigmoidcolon.functional as F

class SigmoidColon(torch.nn.Module):
    def forward(self, x):
        return F.sigmoidcolon(x)
