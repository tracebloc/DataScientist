from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig

model_id = "bert-base-uncased"
hf_token = "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"
framework = "pytorch"
main_class = "MyModel"
category = "text_classification"
model_type = ""
batch_size = 16
output_classes = 2


def MyModel(num_classes=2):
    config = AutoConfig.from_pretrained(model_id, num_labels=num_classes)
    return AutoModelForSequenceClassification.from_config(config)
