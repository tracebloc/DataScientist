import torch.nn as nn
from torchvision.models.detection import keypointrcnn_resnet50_fpn as kprcnn

framework = "pytorch"
model_type = "rcnn"
main_class = "MyModel"
image_size = 448
batch_size = 16
output_classes = 1
category = "keypoint_detection"


class MyModel(nn.Module):
    def __init__(self, num_keypoints=4):
        super(MyModel, self).__init__()

        self.model = kprcnn(pretrained=False,
                            pretrained_backbone=True,
                            num_keypoints=num_keypoints,
                            num_classes=output_classes)

    def forward(self, x, targets=None):
        return self.model(x, targets=targets)
