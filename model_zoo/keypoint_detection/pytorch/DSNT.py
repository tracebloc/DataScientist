import torch
import torch.nn as nn
import torch.nn.functional as F


# Configuration
framework = "pytorch"
model_type = ""
main_class = "KeypointDetectionModel"
image_size = 64
batch_size = 16
output_classes = 1
category = "keypoint_detection"
num_keypoints = 16

class KeypointDetectionModel(nn.Module):
    def __init__(self, num_keypoints=16):
        super(KeypointDetectionModel, self).__init__()
        self.num_keypoints = num_keypoints
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Output layer for coordinates
        self.coords_fc = nn.Linear(self.determine_flattened_size(), num_keypoints * 2)

    def determine_flattened_size(self):
        # This should be set appropriately based on the output size of conv4
        return 256 * 4 * 4  # Placeholder value

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)

        x_flat = torch.flatten(x, start_dim=1)
        coords = self.coords_fc(x_flat)
        coords = coords.view(-1, self.num_keypoints, 2)  # Reshape to [batch_size, num_keypoints, 2]

        return coords