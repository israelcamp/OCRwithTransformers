import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger

from .tuner import TRTuner
from .datamodule import SROIETask2DataModule
from .model import CNNTr

if __name__ == "__main__":

    pl.seed_everything(0)

    tokenizer_file = "nm_spm.model"
    max_epochs = 30

    dm = SROIETask2DataModule(
        root_dir="SROIETask2/data/",
        label_file="SROIETask2/data.json",
        tokenizer_file=tokenizer_file,
        height=32,
        num_workers=8,
        train_bs=16,
        valid_bs=64,
        max_width=None,
        do_pool=True,
    )

    dm.setup("fit")
    steps = len(dm.train_dataloader()) * max_epochs

    model = TRTuner(
        CNNTr(),
        {
            "tokenizer_file": tokenizer_file,
            "lr": 1e-4,
            "optimizer": "Adam",
            "steps": steps,
        },
    )

    FAST_DEV_RUN = False
    trainer_params = {
        "max_epochs": max_epochs,
        "gpus": [1],
        "log_every_n_steps": 1,
        "fast_dev_run": FAST_DEV_RUN,
        "log_gpu_memory": True,
    }

    if not FAST_DEV_RUN:
        neptune_logger = NeptuneLogger(
            project_name="", # FILL HERE
            experiment_name="",  # Optional,
            tags=[], # Optional
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filename=f"{neptune_logger.version}"
            + "{epoch}-{val_loss:.4f}-{val_em:.4f}-{val_f1:.4f}",
            monitor="val_f1",
            mode="max",
            save_top_k=1,
        )

        trainer_params.update(
            {
                "logger": neptune_logger,
                "callbacks": [checkpoint_callback],
            }
        )

    trainer = pl.Trainer(**trainer_params)

    trainer.fit(model, dm)
