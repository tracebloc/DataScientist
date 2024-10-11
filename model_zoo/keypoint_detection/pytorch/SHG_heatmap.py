import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
framework = "pytorch"
model_type = "heatmap"
main_class = "StackHourglass"
# for this model image_size minimum value should be 64
image_size = 64
batch_size = 128
output_classes = 1
category = "keypoint_detection"
num_feature_points = 16

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        skip = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + skip)


class Hourglass(nn.Module):
    def __init__(self, depth, num_channels):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.num_channels = num_channels

        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()

        for _ in range(depth):
            self.downsample_layers.append(ResidualBlock(num_channels, num_channels))
            self.skip_connections.append(ResidualBlock(num_channels, num_channels))
            self.upsample_layers.append(ResidualBlock(num_channels, num_channels))

        self.middle = ResidualBlock(num_channels, num_channels)

    def forward(self, x, original_size):
        skip_connections = []

        for down, skip in zip(self.downsample_layers, self.skip_connections):
            x = down(x)
            skip_connections.append(skip(x))
            x = F.avg_pool2d(x, 2)

        x = self.middle(x)

        for up, skip in zip(self.upsample_layers, reversed(skip_connections)):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
                x = F.pad(
                    x, (0, skip.size(3) - x.size(3), 0, skip.size(2) - x.size(2))
                )
            x = up(x) + skip

        x = F.interpolate(x, size=original_size, mode="bilinear", align_corners=False)

        return x


class StackHourglass(nn.Module):
    def __init__(self, num_stacks=8, stack_channels=256, num_feature_points=16):
        super(StackHourglass, self).__init__()
        self.num_stacks = num_stacks
        self.num_channels = stack_channels
        self.num_feature_points = num_feature_points

        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, stack_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(stack_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(stack_channels, stack_channels),
            nn.MaxPool2d(2, stride=2),
        )

        self.hourglasses = nn.ModuleList(
            [Hourglass(4, stack_channels) for _ in range(num_stacks)]
        )
        self.output_layers = nn.ModuleList(
            [
                nn.Conv2d(stack_channels, num_feature_points, kernel_size=1, stride=1, padding=0)
                for _ in range(num_stacks)
            ]
        )

    def forward(self, x):
        original_size = x.shape[2:]  # Store original input size (height, width)
        x = self.pre_layers(x)

        outputs = []
        for hg, out_layer in zip(self.hourglasses, self.output_layers):
            x = hg(x, original_size)
            out = out_layer(x)
            outputs.append(out)

        return outputs[-1]  # return the heatmaps from the last stack
