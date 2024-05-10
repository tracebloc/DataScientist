import torch
import torch.nn as nn

framework = "pytorch"
model_type = ""
main_class = "OpenPoseKeypointDetector"
image_size = 32
batch_size = 16
output_classes = 1
category = "keypoint_detection"


class VGGBackbone(nn.Module):
    def __init__(self):
        super(VGGBackbone, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class StageModule(nn.Module):
    def __init__(self, in_channels: int, num_keypoints: int):
        super(StageModule, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_keypoints * 3, kernel_size=1)
        )

    def forward(self, x):
        return self.module(x)


class OpenPoseKeypointDetector(nn.Module):
    def __init__(self, num_keypoints: int=4, stages: int = 3):
        super(OpenPoseKeypointDetector, self).__init__()
        # Backbone
        self.backbone = VGGBackbone()

        # Initial stage module (backbone output to heatmaps)
        self.initial_stage = StageModule(256, num_keypoints)

        # Additional stage modules for refinement
        self.stages = nn.ModuleList(
            [StageModule(256 + num_keypoints * 3, num_keypoints) for _ in range(stages - 1)]
        )

    def forward(self, x):
        # Extract features using the backbone
        backbone_features = self.backbone(x)

        # First stage output
        stage_output = self.initial_stage(backbone_features)

        # Further stages for refinement
        for stage in self.stages:
            concatenated_input = torch.cat([backbone_features, stage_output], dim=1)
            stage_output = stage(concatenated_input)

        # Pool spatially to produce the final (batch_size, num_keypoints, 3) tensor
        batch_size, _, height, width = stage_output.shape
        output = nn.functional.adaptive_avg_pool2d(stage_output, (1, 1)).view(batch_size, -1, 3)

        return output

