from torch import nn
import torch
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.elu = torch.nn.ELU()

        self.conv1 = nn.Conv2d(1, 64, (3,3), padding='same')
        self.initialise_layer(self.conv1)
        self.pool1 = nn.MaxPool2d((2,2))

        self.conv2 = nn.Conv2d(64, 128, (3,3), 1, padding='same')
        self.initialise_layer(self.conv2)
        self.pool2 = nn.MaxPool2d((3,3))
        
        self.conv3 = nn.Conv2d(128, 128, (3,3), 1, padding='same')
        self.initialise_layer(self.conv3)
        self.pool3 = nn.MaxPool2d((4,4))

        self.conv4 = nn.Conv2d(128, 128, (3,3), 1, padding='same')
        self.initialise_layer(self.conv4)

        self.gru1 = torch.nn.GRU(15, 15)

        self.gru2 = torch.nn.GRU(15, 15)

        self.fc = torch.nn.Linear(1920, 50)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        print("Input shape ", input.shape)
        batch_size = input.shape[0]
        # x = torch.flatten(input, 0, 2)
        # print("After flatten shape ", x.shape)
        x = self.elu(self.conv1(input))
        print("Conv1 shape ", x.shape)
        x = self.pool1(x)
        print("Pool1 shape ", x.shape)
        x = self.elu(self.conv2(x))
        print("Conv2 shape ", x.shape)
        x = self.pool2(x)
        # print("Pool2 shape ", x.shape)
        x = self.elu(self.conv3(x))
        # print("Conv3 shape ", x.shape)
        x = self.pool3(x)
        # print("Pool3 shape ", x.shape)
        x = self.elu(self.conv4(x))
        # print("Conv4 shape ", x.shape)
        x = self.pool3(x)
        # print("Pool4 shape ", x.shape)
        # print("Before reshape ", x.shape)
        x = torch.flatten(x, -2, -1)
        # x = torch.transpose(x, 1, 2)
        # print("After reshape ", x.shape)
        x, _ = self.gru1(x)
        x = self.elu(x)
        # print("After GRU1 ", x.shape)
        x, _ = self.gru2(x)
        x = self.elu(x)
        # print("After GRU2 ", x.shape)
        x = torch.flatten(x, 1, 2)
        # print("After flatten ", x.shape)
        x = self.fc(x)
        x = torch.sigmoid(x)
        # print("After FF ", x.shape)
        # print(x)

        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)