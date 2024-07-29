import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
framework = "pytorch"
model_type = ""
main_class = "MyModel"
image_size = 64
batch_size = 16
output_classes = 1
category = "keypoint_detection"
num_keypoints = 16

class MyModel(nn.Module):
    def __init__(self, num_keypoints=16):
        super(MyModel, self).__init__()
        self.num_keypoints = num_keypoints
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Initialize these here but set their values dynamically later
        self.flattened_size = None
        self.coords_fc = None

    def compute_flattened_size(self, x):
        """Compute the flattened feature size dynamically."""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        return x.numel() // x.shape[0]  # Compute total features per batch example

    def initialize_fc_layers(self):
        """Initialize fully connected layers once flattened size is known."""
        self.coords_fc = nn.Linear(self.flattened_size, self.num_keypoints * 3).to(self.device)

    def forward(self, x):
        if self.flattened_size is None:
            # Dummy pass to set up layers based on input dimensions
            self.flattened_size = self.compute_flattened_size(x)
            self.initialize_fc_layers()

        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)

        # Flatten the features
        x_flat = torch.flatten(x, start_dim=1)

        # Calculate coordinates
        coords = self.coords_fc(x_flat)
        coords = coords.view(-1, self.num_keypoints, 3)  # Reshape to [batch_size, num_keypoints, 2]

        return coords
