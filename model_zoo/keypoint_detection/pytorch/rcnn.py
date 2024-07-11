import torch
import torch.nn as nn
from torchvision.models.detection import keypointrcnn_resnet50_fpn as kprcnn

framework = "pytorch"
model_type = "rcnn"
main_class = "MyModel"
image_size = 64
batch_size = 8
output_classes = 1
category = "keypoint_detection"
num_keypoints = 16

class MyModel(nn.Module):
    def __init__(self, num_keypoints=num_keypoints):
        super(MyModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.model = kprcnn(
            pretrained=False,
            pretrained_backbone=True,
            num_keypoints=num_keypoints,
            num_classes=output_classes,
        )

    def forward(self, x, targets=None):
        return self.model(x, targets=targets)
