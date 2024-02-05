import torch
import torch.nn as nn

from arch_config import architecture_config
from utils_class import CNNBlock
from utils import *


framework = "pytorch"
model_type = "yolo"
main_class = "MyModel"
image_size = 448
batch_size = 64
output_classes = 3


def _create_fcs(split_size=7, num_boxes=2, num_classes=3):
    S, B, C = split_size, num_boxes, num_classes
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(1024 * S * S, 496),
        nn.Dropout(0.0),
        nn.LeakyReLU(0.1),
        nn.Linear(496, S * S * (C + B * 5)),
    )  # Original paper uses nn.Linear(1024 * S * S, 4096) not 496. Also
    # the last layer will be reshaped to (S, S, 13) where C+B*5 = 13


# predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)


class MyModel(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(MyModel, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = _create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3]
                    )
                ]
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(x) == list:
                conv1 = x[0]  # Tuple
                conv2 = x[1]  # Tuple
                repeats = x[2]  # Int

                for _ in range(repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)
