import torch.nn as nn

framework = "pytorch"
model_type = ""
main_class = "SimpleCNN"
image_size = 69
batch_size = 4
output_classes = 2
category = "generic_classification"
num_feature_points = 69

class SimpleCNN(nn.Module):
    def __init__(self, input_size=69):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * input_size, 128)  # input_size is the total number of features
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension for Conv1D (batch_size, channels, input_size)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten before feeding into FC layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x