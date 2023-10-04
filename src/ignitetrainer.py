from dataclasses import dataclass
from typing import Any

import torch
from ignite.engine import (
    Engine,
    Events,
)
from ignite.handlers import Checkpoint
from ignite.contrib.handlers import global_step_from_engine
from ignite.contrib.handlers import ProgressBar

from .igmetrics import ExactMatch


@dataclass
class IgniteTrainer:
    model: Any
    dm: Any
    tokenizer: Any
    decoder: Any
    save_dir: str
    max_epochs: int = 30
    start_from_checkpoint: str = None

    def __post_init__(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-4, weight_decay=0
        )

        self.val_loader = self.dm.val_dataloader()
        self.train_loader = self.dm.train_dataloader()

        self.steps = len(self.train_loader) * self.max_epochs
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.steps, 1e-8
        )

        self.criterion = torch.nn.CTCLoss(
            blank=self.tokenizer.pad_token_id, zero_infinity=True
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.trainer = Engine(self.train_step)
        self.validation_evaluator = Engine(self.val_step)

    def train_step(self, engine, batch):
        self.model.train()
        self.optimizer.zero_grad()

        images, labels, attention_mask, attention_image = [
            x.to(self.device) for x in batch
        ]

        logits = self.model(images, attention_mask=attention_image)

        input_length = attention_image.sum(-1)
        target_length = attention_mask.sum(-1)

        logits = logits.permute(1, 0, 2)
        logits = logits.log_softmax(2)

        loss = self.criterion(logits, labels, input_length, target_length)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()
        self.lr_scheduler.step()
        return loss.item()

    def val_step(self, engine, batch):
        self.model.eval()
        images, labels, attention_mask, attention_image = [
            x.to(self.device) for x in batch
        ]
        with torch.no_grad():
            logits = self.model(images, attention_image)

        y_pred, y = self.get_preds_from_logits(logits, attention_image, labels)
        return y_pred, y

    def get_preds_from_logits(self, logits, attention_image, labels):
        decoded_ids = logits.argmax(-1).squeeze(0)
        if len(decoded_ids.shape) == 1:
            decoded_ids = decoded_ids.unsqueeze(0)
        decoded = [
            self.decoder(dec, att)
            for dec, att in zip(decoded_ids, attention_image)
        ]
        y_pred = self.tokenizer.batch_decode(decoded, skip_special_tokens=True)
        y = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        return y_pred, y

    def log_validation_results(self, engine):
        self.validation_evaluator.run(self.val_loader)
        metrics = self.validation_evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        print(
            f"Validation Results - Epoch: {engine.state.epoch}  Avg accuracy: {avg_accuracy:.3f}"
        )

    def fit(self):
        self.trainer.add_event_handler(
            Events.EPOCH_COMPLETED, self.log_validation_results
        )
        ExactMatch().attach(self.validation_evaluator, "accuracy")

        # best model checkpointing
        to_save = {"model": self.model}
        handler = Checkpoint(
            to_save,
            self.save_dir,
            n_saved=1,
            filename_prefix="best",
            score_name="accuracy",
            global_step_transform=global_step_from_engine(self.trainer),
        )
        self.validation_evaluator.add_event_handler(Events.COMPLETED, handler)

        # iteration checkpointing
        to_save = {
            "model": self.model,
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,
            "trainer": self.trainer,
        }
        handler = Checkpoint(
            to_save,
            self.save_dir,
            n_saved=4,
        )
        if self.start_from_checkpoint is not None:
            print("Loading last good checkpoint", self.start_from_checkpoint)
            checkpoint = torch.load(self.start_from_checkpoint)
            Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

        self.trainer.add_event_handler(
            Events.ITERATION_COMPLETED(every=5_000), handler
        )

        pbar = ProgressBar()
        pbar.attach(self.trainer, output_transform=lambda x: {"loss": x})
        self.trainer.run(self.train_loader, max_epochs=self.max_epochs)
        return handler
