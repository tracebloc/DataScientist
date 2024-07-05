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
num_keypoints = 4

class DSNTLayer(nn.Module):
    def __init__(self):
        super(DSNTLayer, self).__init__()

    def forward(self, heatmap):
        heatmap = F.softmax(heatmap.view(heatmap.size(0), -1), dim=-1).view_as(heatmap)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, heatmap.size(-2)),
            torch.linspace(-1, 1, heatmap.size(-1)),
            indexing="ij",
        )
        grid_x = grid_x.to(heatmap.device)
        grid_y = grid_y.to(heatmap.device)
        x = torch.sum(heatmap * grid_x, dim=(-2, -1))
        y = torch.sum(heatmap * grid_y, dim=(-2, -1))
        coords = torch.stack([x, y], dim=-1)
        return coords


class KeypointDetectionModel(nn.Module):
    def __init__(self, num_keypoints=num_keypoints):
        super(KeypointDetectionModel, self).__init__()
        self.num_keypoints = num_keypoints
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Placeholder until we determine the flattened feature size dynamically
        self.flattened_size = None

        # DSNT Layer
        self.dsnt_layer = DSNTLayer()

        # Visibility fully connected layer (initialized later)
        self.visibility_fc = None

    def compute_flattened_size(self, x):
        """
        Compute the flattened feature size dynamically.
        Call this method with a dummy input during initialization.
        """
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        return torch.flatten(x, start_dim=1).size(1)

    def initialize_fc_layers(self, flattened_size):
        """
        Initialize the fully connected layers with the determined flattened size.
        """
        self.fc1 = nn.Linear(flattened_size, 1024)
        self.fc2 = nn.Linear(1024, self.num_keypoints * 64 * 64)
        self.visibility_fc = nn.Linear(flattened_size, self.num_keypoints)

    def forward(self, x):
        # Ensure fully connected layers are initialized
        if self.flattened_size is None:
            self.flattened_size = self.compute_flattened_size(x)
            self.initialize_fc_layers(self.flattened_size)

        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)

        x_flat = torch.flatten(x, start_dim=1)

        # Heatmap generation
        heatmap = F.relu(self.fc1(x_flat))
        heatmap = self.fc2(heatmap).view(-1, self.num_keypoints, 64, 64)

        # Get keypoint coordinates
        coords = self.dsnt_layer(heatmap)

        # Predict visibility
        visibility = torch.sigmoid(self.visibility_fc(x_flat))

        # Concatenate coordinates and visibility
        output = torch.cat([coords, visibility.unsqueeze(-1)], dim=-1)

        return output
