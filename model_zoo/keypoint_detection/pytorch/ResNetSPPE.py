import torch
import torch.nn as nn
import torchvision.models as models


# Configuration
framework = "pytorch"
model_type = ""
main_class = "ResNetSPPE"
image_size = 64
batch_size = 16
output_classes = 1
category = "keypoint_detection"

class ResNetSPPE(nn.Module):
    def __init__(self, num_keypoints=4, input_channels=3):
        super(ResNetSPPE, self).__init__()
        self.num_keypoints = num_keypoints

        # Load a pre-trained ResNet model (here we use ResNet-50)
        resnet = models.resnet50(pretrained=True)

        # Modify the first convolution layer to accommodate different input channels
        if input_channels != 3:
            resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Extract all layers except the fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Define a global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Add a fully connected layer to predict keypoints' x, y coordinates and visibility
        num_features = resnet.fc.in_features
        self.fc = nn.Linear(num_features, num_keypoints * 3)

    def forward(self, x):
        # Pass the input through the ResNet backbone
        x = self.backbone(x)

        # Apply global average pooling
        x = self.global_avg_pool(x)

        # Flatten the pooled output
        x = x.view(x.size(0), -1)

        # Pass through the fully connected layer
        x = self.fc(x)

        # Reshape to (batch_size, num_keypoints, 3)
        x = x.view(-1, self.num_keypoints, 3)
        return x
