from torch import nn
import torch
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, length, stride):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 1, length, stride)
        self.initialise_layer(self.conv1)
        self.conv2 = nn.Conv1d(1, 32, 8, 1)
        self.initialise_layer(self.conv2)
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.conv3 = nn.Conv1d(32, 32, 8, 1)
        self.initialise_layer(self.conv3)
        self.full1 = nn.Linear(192, 100)
        self.full2 = nn.Linear(100, 50)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size = input.shape[0]
        x = torch.flatten(input, 0, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.full1(x))
        x = torch.sigmoid(self.full2(x))
        x = torch.reshape(x, (batch_size, 10, 50))
        x = torch.mean(x, dim=1)
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)