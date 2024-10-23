from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
hf_token = "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"
framework = "pytorch"
main_class = "MyModel"
category = "text_classification"
model_type = ""
batch_size = 16
output_classes = 134


def MyModel(num_classes=output_classes):
    config = AutoConfig.from_pretrained(model_id, num_labels=num_classes)
    return AutoModelForSequenceClassification.from_config(config)
