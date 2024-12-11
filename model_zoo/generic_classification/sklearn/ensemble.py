from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

framework = "sklearn"
model_type = "tree"
main_method = "MyModel"
image_size = 69
batch_size = 4
output_classes = 2
category = "generic_classification"
num_feature_points = 69

def MyModel():
    gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    svm = SVC(probability=True, kernel="rbf", random_state=42)

    stacked_model = StackingClassifier(
        estimators=[('gbm', gbm), ('svm', svm)],
        final_estimator=LogisticRegression(random_state=42)
    )
    return stacked_model