import torch.nn as nn
from torchvision.models.detection import keypointrcnn_resnet50_fpn as kprcnn

framework = "pytorch"
model_type = "rcnn"
main_class = "MyModel"
image_size = 64
batch_size = 128
output_classes = 1
category = "keypoint_detection"
num_feature_points = 16

class MyModel(nn.Module):
    def __init__(self, num_feature_points=num_feature_points):
        super(MyModel, self).__init__()

        self.model = kprcnn(
            pretrained=False,
            pretrained_backbone=True,
            num_feature_points=num_feature_points,
            num_classes=output_classes,
        )

    def forward(self, x, targets=None):
        return self.model(x, targets=targets)
