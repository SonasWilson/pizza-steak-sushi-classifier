from pathlib import Path
import torch
from torch import nn

def save_model(model: nn.Module,
               path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

