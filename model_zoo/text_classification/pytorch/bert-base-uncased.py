from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig

model_id = 'bert-base-uncased'
framework = "pytorch"
main_class = "MyModel"
category = "text_classification"
model_type = ""
batch_size = 16
output_classes = 134

def MyModel(num_classes=output_classes):

    return AutoModelForSequenceClassification.from_pretrained(model_id,
        num_labels=num_classes,
        ignore_mismatched_sizes=True)
