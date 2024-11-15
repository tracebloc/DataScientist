import torch.nn as nn
from sklearn.svm import SVC

framework = "sklearn"
model_type = "tree"
main_method = "MyModel"
image_size = 69
batch_size = 4
output_classes = 2
category = "generic_classification"
num_feature_points = 69

def MyModel():
    return SVC(kernel='linear')