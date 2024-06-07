import torch
import torch.nn as nn
import torch.nn.functional as F

framework = "pytorch"
main_class = "RNN"
category = "text_classification"
model_type = ""
input_shape = 512
batch_size = 16
output_classes = 2
n_hidden = 128

class RNN(nn.Module):
    def __init__(self, input_size=input_shape, hidden_size=n_hidden, output_size=output_classes):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        hidden = torch.zeros(1, n_hidden).cuda()
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
