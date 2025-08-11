import torch
import torch.nn as nn
from torch.nn import functional as F

from cs336_systems.benchmark_model import get_best_device


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print("fc1 output:", x.dtype)
        x = self.ln(x)
        print("ln output:", x.dtype)
        x = self.fc2(x)
        print("logits:", x.dtype)
        return x


# dtype = torch.float16
dtype = torch.bfloat16
device = get_best_device()

model = ToyModel(10, 10).to(device)
x = torch.randn(10, 10, device=device)

print(f"{device=}")

with torch.autocast(device_type=device, dtype=dtype):
    param_dtypes = {param.dtype for param in model.parameters()}
    print("param_dtypes:", param_dtypes)

    y = model(x)

    loss = F.cross_entropy(y, torch.randint(10, (10,), device=device))

    print("loss:", loss.dtype)

    loss.backward()

    grad_dtypes = {param.grad.dtype for param in model.parameters()}
    print("grad_dtypes:", grad_dtypes)
