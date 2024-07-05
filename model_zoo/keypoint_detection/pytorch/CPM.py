import torch.nn as nn

# Configuration
framework = "pytorch"
model_type = ""
main_class = "CPM"
image_size = 512
batch_size = 16
output_classes = 1
category = "keypoint_detection"
num_keypoints = 4


class CPMStage(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(CPMStage, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class CPM(nn.Module):
    def __init__(self, in_channels=3, num_keypoints=num_keypoints, num_stages=6):
        super(CPM, self).__init__()
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.num_stages = num_stages

        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        self.stages = nn.ModuleList()
        for i in range(num_stages):
            stage = CPMStage(128, 128)
            self.stages.append(stage)

        self.conv_out = nn.Conv2d(
            128, num_keypoints * 3, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        batch_size, _, height, width = x.size()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        for stage in self.stages:
            out = stage(out)

        out = self.conv_out(out)
        out = out.view(batch_size, self.num_keypoints, 3, height, width)
        out = out.mean(dim=(-2, -1))  # Take the mean across height and width dimensions
        return out
