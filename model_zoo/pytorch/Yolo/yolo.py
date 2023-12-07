import torch
import torchvision
import torch.nn as nn

framework = "pytorch"
model_type = "yolo"
main_class = "YoloV1"
image_size = 448
batch_size = 64


class CNNBlock(nn.Module):
    def __init__(self,in_channels, out_channels ,kernel_size = 3, stride =1  , padding = 1):## if groups = in_channels then this is depth_wise convolution operation
        super(CNNBlock , self).__init__()
        self.conv2d = nn.Conv2d(in_channels = in_channels , out_channels = out_channels , kernel_size = kernel_size , stride = stride , padding = padding , bias = False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)
    def forward(self,x):
        return self.leaky_relu(self.batchnorm(self.conv2d(x)))


class YoloV1(nn.Module):
    def __init__(self, grid_size=7, n_box=2, n_class=3,
                 input_channels=3):  # S = grid_size , n_box = Bounding boxes , n_class = how many classes for classification!
        super(YoloV1, self).__init__()
        ## This will be a long one so you can pass it if you want. Basicly creating the same model on the pic. above.

        self.darknet = nn.Sequential(
            CNNBlock(in_channels=input_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CNNBlock(in_channels=64, out_channels=192, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CNNBlock(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CNNBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CNNBlock(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=2),
            CNNBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=2),
            CNNBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear((1024 * grid_size * grid_size), 4096),  # output of darknet will be S x S x 1024
            nn.Dropout(0.4),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, grid_size * grid_size * (n_class + n_box * 5))
            # there will be n_class + n_box*5 outputs for each grid (we multiplt n_box with  5 because each box contains; midpoint_x , midpoint_y , width , height ,confidence  )
        )

    def forward(self, x):
        x = self.darknet(x)
        return self.fc(x)
