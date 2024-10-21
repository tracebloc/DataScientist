import torch.nn as nn

framework = "pytorch"
model_type = ""
main_class = "SimpleLSTM"
image_size = 69
batch_size = 4
output_classes = 2
category = "generic_classification"
num_feature_points = 69

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=69, hidden_size=128, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm(x)  # LSTM returns the output and hidden states
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

