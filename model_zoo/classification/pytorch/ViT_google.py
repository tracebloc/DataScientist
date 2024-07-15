import torch
from torch import nn
from transformers import ViTForImageClassification, ViTConfig


framework = "pytorch"
main_class = "VisionTransformer"
image_size = 224
batch_size = 16
category = "image_classification"
output_classes = 3


class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize the model with a specified number of output labels for classification
        config = ViTConfig.from_pretrained(
            "google/vit-base-patch16-224", num_labels=output_classes
        )
        self.vit = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", config=config, ignore_mismatched_sizes=True
        )

    def forward(self, pixel_values):
        # The model will output a dictionary with various keys.
        outputs = self.vit(pixel_values=pixel_values)
        # The logits are now directly available from the output's 'logits' key.
        logits = outputs.logits
        return logits
