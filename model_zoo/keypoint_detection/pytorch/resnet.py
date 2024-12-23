# file: models/simple_baseline.py

import torch
import torch.nn as nn
import torchvision.models as models

# Configuration
framework = "pytorch"
model_type = ""
main_class = "SimpleBaseline"
image_size = 64
batch_size = 128
output_classes = 1
category = "keypoint_detection"
num_feature_points = 16

class SimpleBaseline(nn.Module):
    def __init__(self, num_feature_points=num_feature_points):
        super(SimpleBaseline, self).__init__()
        # Load a pre-trained ResNet backbone
        backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            *(list(backbone.children())[:-2])
        )  # Remove the classification layers

        # Deconvolutional layers for upsampling
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(
                2048, 256, kernel_size=4, stride=2, padding=1, output_padding=0
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                256, 256, kernel_size=4, stride=2, padding=1, output_padding=0
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                256, 256, kernel_size=4, stride=2, padding=1, output_padding=0
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        # Final layer to produce the heatmap
        self.final_layer = nn.Conv2d(256, num_feature_points, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.deconv_layers(x)
        heatmap = self.final_layer(x)

        # Extract keypoints by finding the maximum locations in the heatmaps
        batch_size, num_feature_points, _, _ = heatmap.size()
        keypoints = torch.zeros((batch_size, num_feature_points, 3), device=heatmap.device)

        for i in range(num_feature_points):
            heatmap_i = heatmap[:, i, :, :].reshape(batch_size, -1)
            maxvals, idxs = torch.max(heatmap_i, dim=1)
            keypoints[:, i, 0] = idxs % heatmap.size(3)  # X-coordinate
            keypoints[:, i, 1] = idxs // heatmap.size(3)  # Y-coordinate
            keypoints[:, i, 2] = maxvals  # Visibility

        return keypoints
