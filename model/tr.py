import torch
from transformers import RobertaConfig, RobertaForTokenClassification

from .cnn import CNN

class CNNTr(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = CNN()
        config_dict = {
            "architectures": ["RobertaForTokenClassification"],
            "num_labels": 100,
            "attention_probs_dropout_prob": 0.25,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.25,
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 768,
            "layer_norm_eps": 1e-05,
            "max_position_embeddings": 514,
            "model_type": "roberta",
            "num_attention_heads": 8,
            "num_hidden_layers": 3,
            "pad_token_id": 0,
            "type_vocab_size": 1,
            "vocab_size": 256,
        }
        config = RobertaConfig(**config_dict)
        self.encoder = RobertaForTokenClassification(config)

    def forward(self, images, attention_mask=None):
        image_embeddings = self.cnn(images)
        outputs = self.encoder(
            inputs_embeds=image_embeddings, attention_mask=attention_mask
        )
        return outputs.logits
