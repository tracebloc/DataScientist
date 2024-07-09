import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
framework = "pytorch"
model_type = ""
main_class = "DeepPoseModel"
image_size = 64
batch_size = 16
output_classes = 1
category = "keypoint_detection"
num_keypoints = 16  # Updated to 16 keypoints

class DeepPoseModel(nn.Module):
    def __init__(self, num_keypoints: int = num_keypoints, image_shape=image_size):
        super(DeepPoseModel, self).__init__()
        self.num_keypoints = num_keypoints

        # Define the CNN backbone layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)

        # Adaptive pooling layer to fix the output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Outputs a fixed 1x1 feature map per channel

        # Flatten size adjusted for adaptive pooling
        self.fc1 = nn.Linear(1024, 1024)  # 1024 channels from conv5, each reduced to 1x1
        self.fc2 = nn.Linear(1024, num_keypoints * 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers for keypoint prediction
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Reshape to the output format: (batch_size, num_keypoints, 3)
        x = x.view(-1, self.num_keypoints, 3)

        return x