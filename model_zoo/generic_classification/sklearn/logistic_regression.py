import torch.nn as nn
from sklearn.linear_model import LogisticRegression

framework = "sklearn"
model_type = "linear"
main_method = "MyModel"
image_size = 69
batch_size = 4
output_classes = 2
category = "generic_classification"
num_feature_points = 69

def MyModel():
    return LogisticRegression(random_state=42)