import torch
from torch import nn
from transformers import ViTModel, ViTConfig

framework = "pytorch"
main_class = "VisionTransformer"
image_size = 224
batch_size = 16
category = "image_classification"


class VisionTransformer(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        # Initialize the configuration for ViT
        self.config = ViTConfig.from_pretrained(
            "google/vit-base-patch16-224", num_labels=num_labels
        )

        # Initialize the ViT model
        self.vit = ViTModel(self.config)

        # Here you can add more layers if you want, for example a classification head
        self.classification_head = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        logits = self.classification_head(outputs.last_hidden_state[:, 0])
        return logits
