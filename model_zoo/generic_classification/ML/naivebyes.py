import torch.nn as nn
from sklearn.naive_bayes import GaussianNB

framework = "pytorch"
model_type = "ml"
main_class = "MyModel"
image_size = 69
batch_size = 4
output_classes = 2
category = "generic_classification"
num_feature_points = 69

def MyModel():
    return GaussianNB()