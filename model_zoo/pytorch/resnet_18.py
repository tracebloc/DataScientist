import torch
import torch.nn as nn

framework = "pytorch"
main_class = "MyModel"
image_size = 224
batch_size = 16



class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False,
         num_classes=2)

    def forward(self, x):
        return self.model(x)