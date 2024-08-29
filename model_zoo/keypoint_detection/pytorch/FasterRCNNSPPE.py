import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn


# Configuration
framework = "pytorch"
model_type = ""
main_class = "FasterRCNNSPPE"
image_size = 64
batch_size = 128
output_classes = 1
category = "keypoint_detection"
num_keypoints = 16

class FasterRCNNSPPE(nn.Module):
    def __init__(self, num_keypoints=num_keypoints):
        super(FasterRCNNSPPE, self).__init__()
        self.num_keypoints = num_keypoints

        # Load the Faster R-CNN model to get the backbone (ResNet-50)
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        backbone = model.backbone

        # Assume the feature extractor provides a feature map, which is what we use here
        self.feature_extractor = backbone

        # Create a pooling layer compatible with the Faster R-CNN backbone output
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Adjust the feature map size to the input of the final fully connected layer
        num_features = (
            256  # Feature size; confirm actual dimensions from the backbone output
        )
        self.fc = nn.Linear(num_features, num_keypoints * 3)

    def forward(self, x):
        # Ensure x is a tensor and process through the backbone
        features = self.feature_extractor(x)

        # Depending on the backbone structure, you may need to specify the output layer or pick one feature
        if isinstance(features, dict):
            # Pick a particular layer output (e.g., '0') based on your feature extractor
            x = features["0"]  # Replace with the appropriate key

        # Apply adaptive average pooling to match expected fully connected layer input size
        x = self.global_avg_pool(x)

        # Flatten pooled output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Pass through the fully connected layer for keypoint prediction
        x = self.fc(x)

        # Reshape to (batch_size, num_keypoints, 3)
        x = x.view(-1, self.num_keypoints, 3)
        return x
