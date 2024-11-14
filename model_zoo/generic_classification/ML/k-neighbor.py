import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier

framework = "sklearn"
model_type = "knn"
main_class = "MyModel"
image_size = 69
batch_size = 4
output_classes = 2
category = "generic_classification"
num_feature_points = 69

def MyModel():
    return KNeighborsClassifier(n_neighbors=5)