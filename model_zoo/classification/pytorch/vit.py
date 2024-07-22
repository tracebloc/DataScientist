import torch
from torch import nn
from transformers import ViTModel, ViTConfig

framework = "pytorch"
main_class = "VisionTransformer"
image_size = 224
batch_size = 16
category = "image_classification"
output_classes = 3


class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize the configuration for ViT
        self.config = ViTConfig.from_pretrained(
            "google/vit-base-patch16-224", num_labels=output_classes
        )

        # Initialize the ViT model
        self.vit = ViTModel(self.config)

        # Here you can add more layers if you want, for example a classification head
        self.classification_head = nn.Linear(self.config.hidden_size, output_classes)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        logits = self.classification_head(outputs.last_hidden_state[:, 0])
        return logits
