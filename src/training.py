from transformers import AutoTokenizer


from .ctc import GreedyDecoder
from .ignitetrainer import IgniteTrainer
from .datamodule import SynthDataModule, SynthDataset
from .model import (
    CNNSmallDropout,
    DebertaEncoder,
    OCRModel,
)

if __name__ == "__main__":
    IMAGES_DIR = "../data/synth/mnt/90kDICT32px/"
    TRAIN_ANNOTATION_FILE = "../data/synth/mnt/annotation_train_good.txt"
    VAL_ANNOTATION_FILE = "../data/synth/mnt/annotation_val_good.txt"

    tokenizer = AutoTokenizer.from_pretrained(
        "./synth-36char-tokenizers/tokenizer-36char"
    )
    decoder = GreedyDecoder(tokenizer.pad_token_id)

    train_dataset = SynthDataset(IMAGES_DIR, TRAIN_ANNOTATION_FILE)
    val_dataset = SynthDataset(IMAGES_DIR, VAL_ANNOTATION_FILE)

    datamodule = SynthDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        train_bs=16,
        valid_bs=16,
        num_workers=4,
    )

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    vis_model = CNNSmallDropout(vocab_size=tokenizer.vocab_size, p=0.15)
    tr_model = DebertaEncoder(
        vocab_size=tokenizer.vocab_size + 1, tokenizer=tokenizer
    )
    model = OCRModel(vis_model, tr_model)

    ignite_trainer = IgniteTrainer(
        model=model,
        dm=datamodule,
        tokenizer=tokenizer,
        decoder=decoder,
        save_dir="./synth-tr-models",
        max_epochs=2,
    )
    ignite_trainer.fit()
