import torch
from torch import nn
from transformers import DebertaV2ForTokenClassification, DebertaV2Config


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Feature2Embedding(nn.Module):
    """
    Convert [B, C, H, W] image feature tensor to [B, seq_len, D]
    (B, C, H, W) -> (B, W, H, C)
    """

    def forward(self, x):
        n, c, h, w = x.shape
        return x.permute(0, 3, 2, 1).reshape(n, -1, c)

class BaseCNN(nn.Module):

    def __init__(
        self,
        vocab_size: int = 100,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.image_embeddings = None
        self.lm_head = None

    def block(self, in_channels, out_channels, st=2):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=st, padding=1),
            nn.BatchNorm2d(out_channels),
            Swish(),
        )

    def forward(self, images, *args, **kwargs):
        embedding = self.image_embeddings(images)
        return embedding

    def lm(self, embedding):
        return self.lm_head(embedding)


class CNNSmallDropout(BaseCNN):

    def __init__(
        self,
        vocab_size: int = 100,
        p: float = 0.15,
    ):
        super().__init__(vocab_size=vocab_size)

        self.image_embeddings = nn.Sequential(
            self.block(1, 64, st=(2, 2)),
            nn.Dropout2d(p),
            self.block(64, 128, st=(2, 1)),
            nn.Dropout2d(p),
            self.block(128, 256, st=(2, 1)),
            nn.Dropout2d(p),
            self.block(256, 512, st=(4, 1)),
            nn.Dropout2d(p),
            Feature2Embedding(),
        )
        self.lm_head = nn.Linear(512, self.vocab_size)

    def forward(self, images, *args, **kwargs):
        embeddings = self.image_embeddings(images)
        return embeddings


class DebertaEncoder(nn.Module):
    def __init__(
        self, vocab_size: int = 100, config_dict: dict = {}, tokenizer=None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        config_dict = self._get_config_dict(config_dict)
        config = DebertaV2Config(**config_dict)
        self.encoder = DebertaV2ForTokenClassification(config)

    def _get_config_dict(self, config_dict):
        base_config_dict = {
            "model_type": "deberta-v2",
            "architectures": ["DebertaV2ForTokenClassification"],
            "num_labels": self.vocab_size,
            "model_type": "deberta-v2",
            "attention_probs_dropout_prob": 0.15,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.15,
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 768,  
            "max_position_embeddings": 212, 
            "relative_attention": True,
            "position_buckets": 32,  
            "norm_rel_ebd": "layer_norm",
            "share_att_key": True,
            "pos_att_type": "p2c|c2p",
            "layer_norm_eps": 1e-7,
            "max_relative_positions": -1,
            "position_biased_input": True,
            "num_attention_heads": 8,
            "num_hidden_layers": 3,
            "type_vocab_size": 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.pad_token_id,
            "vocab_size": self.vocab_size,
        }
        base_config_dict.update(config_dict)
        return base_config_dict

    def forward(self, image_embeddings, attention_mask=None):
        outputs = self.encoder(
            inputs_embeds=image_embeddings, attention_mask=attention_mask
        )
        return outputs.logits


class OCRModel(nn.Module):
    def __init__(self, visual_model, rec_model: DebertaEncoder):
        super().__init__()
        self.visual_model = visual_model
        self.rec_model = rec_model

    def forward(self, images, attention_mask=None):
        features = self.visual_model(images)
        logits = self.rec_model(features, attention_mask=attention_mask)
        return logits

    def cnn_lm(self, embedding):
        return self.visual_model.lm(embedding)
