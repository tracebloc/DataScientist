import torch
import torchvision
import torch.nn as nn

framework = "pytorch"
main_class = "MyModel"
model_type = ""
image_size = 224
batch_size = 16
output_classes = 3


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = torchvision.models.vit_b_16(
            pretrained=False, progress=True, num_classes=output_classes
        )

    def forward(self, x):
        return self.model(x)
