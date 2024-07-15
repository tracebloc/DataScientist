import torch
import torch.nn as nn

# Configuration
framework = "pytorch"
model_type = ""
main_class = "HRNetKeypointDetection"
image_size = 64
batch_size = 16
output_classes = 1
category = "keypoint_detection"
num_keypoints = 16

class HRNetKeypointDetection(nn.Module):
    def __init__(self, num_keypoints: int = num_keypoints, input_channels: int = 3):
        super(HRNetKeypointDetection, self).__init__()
        self.num_keypoints = num_keypoints

        # Example HRNet-style backbone: Modify it to fit your specific needs
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Final layer to predict keypoint coordinates (x, y) and visibility (v)
        self.fc = nn.Linear(512, num_keypoints * 3)

    def forward(self, x: torch.Tensor):
        # Input shape: (batch_size, input_channels, height, width)
        batch_size = x.size(0)

        # Extract features using the HRNet-like backbone
        features = self.backbone(x)

        # Global Average Pooling to reduce the spatial dimensions
        pooled_features = nn.functional.adaptive_avg_pool2d(features, (1, 1)).view(
            batch_size, -1
        )

        # Final layer to get the keypoints and visibility
        keypoints = self.fc(pooled_features).view(batch_size, self.num_keypoints, 3)

        return keypoints
