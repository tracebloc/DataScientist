import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork


# Configuration
framework = "pytorch"
model_type = "heatmap"
main_class = "CascadedPyramidNetwork"
image_size = 64
batch_size = 128
output_classes = 1
category = "keypoint_detection"
num_feature_points = 16



class KeypointHead(nn.Module):
    def __init__(self, in_channels, num_feature_points):
        super(KeypointHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, num_feature_points, kernel_size=1, stride=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class CascadedPyramidNetwork(nn.Module):
    def __init__(self, num_feature_points=num_feature_points):
        super(CascadedPyramidNetwork, self).__init__()

        # Load a pretrained ResNet backbone
        backbone = models.resnet50(pretrained=True)

        # Initial layers for feature extraction
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Initialize the FPN with suitable input channels
        in_channels = [256, 512, 1024, 2048]
        self.fpn = FeaturePyramidNetwork(in_channels, out_channels=256)

        # Keypoint Head for final prediction
        self.keypoint_head = KeypointHead(256, num_feature_points)

        # Upsample layer to match the input image size
        self.upsample = nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=False)

    def forward(self, x):
        # Process the input tensor through the initial ResNet layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Generate feature maps using intermediate layers
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        # Create a dictionary of feature maps for the FPN
        features = {"c1": c1, "c2": c2, "c3": c3, "c4": c4}

        # Aggregate features using the FPN
        fpn_output = self.fpn(features)
        fpn_out = fpn_output["c4"]

        # Apply the keypoint detection head to get heatmaps
        keypoint_heatmaps = self.keypoint_head(fpn_out)

        # Upsample the heatmaps to the input image size
        keypoint_heatmaps = self.upsample(keypoint_heatmaps)

        return keypoint_heatmaps
