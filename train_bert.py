"""
Step 2b: Fine-tune BERT (or RoBERTa) on Reddit 6-class emotion. Saves model to outputs/models.
"""
import os
import logging
import warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import config
from data_load import get_cached


class RedditDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[i], dtype=torch.long),
        }


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    acc = (preds == labels).mean()
    from sklearn.metrics import f1_score
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


def main():
    reddit, _ = get_cached()
    X_train = reddit["train"]["texts"]
    y_train = reddit["train"]["labels"]
    X_val = reddit["validation"]["texts"]
    y_val = reddit["validation"]["labels"]

    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BERT_MODEL_NAME, num_labels=len(config.SIX_LABELS)
    )

    train_ds = RedditDataset(X_train, y_train, tokenizer, config.MAX_LENGTH)
    val_ds = RedditDataset(X_val, y_val, tokenizer, config.MAX_LENGTH)

    training_args = TrainingArguments(
        output_dir=os.path.join(config.MODELS_DIR, "bert_reddit_run"),
        num_train_epochs=config.BERT_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.BERT_LR,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=config.RANDOM_SEED,
        log_level="warning",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(os.path.join(config.MODELS_DIR, "bert_reddit"))
    tokenizer.save_pretrained(os.path.join(config.MODELS_DIR, "bert_reddit"))
    print("Saved bert_reddit")


if __name__ == "__main__":
    main()
