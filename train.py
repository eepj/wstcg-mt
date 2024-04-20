# Import torch
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# Import huggingface pre-trained models
from transformers import MarianTokenizer, MarianMTModel

# Import additional libraries
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
import warnings

from dataset import CardTextDataset

# Training device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define and set random states
RANDOM_STATE_DEFAULT = 42
# Default hyperparameters
LR_DEFAULT = 1e-4
EPOCH_DEFAULT = 5

# Pre-trained model name
HF_PRETRAINED_NAME = "Helsinki-NLP/opus-mt-ja-en"

# Additional tokens
TOKENIZER_EXTRA_TOKENS = [
    # Substitutes for character name and trait references
    "<TRAIT>", "<NAME>",
    # Substitutes for trigger icons
    "<SOUL>", "<CHOICE>", "<TREASURE>", "<SALVAGE>", "<STANDBY>",
    "<GATE>", "<BOUNCE>", "<STOCK>", "<SHOT>", "<DRAW>",
    # Tokens for keywords
    "【", "】", "AUTO", "ACT", "CONT", "COUNTER", "CLOCK",
    "トリガー",
]


def main():
    # Parse the input arguments
    parser = argparse.ArgumentParser()
    # Dataset CSV file paths
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    # Japanese text column name
    parser.add_argument("--ja", required=True)
    # English text column name
    parser.add_argument("--en", required=True)
    # Model and tokenizer path
    parser.add_argument("--model", default=None)
    # Model export path
    parser.add_argument("--export", required=True)

    # Training hyperparameters
    # Training epochs
    parser.add_argument("--epochs", default=10, type=int)
    # Batch size
    parser.add_argument("--batch_size", default=1, type=int)
    # Initial learning rate
    parser.add_argument("--lr", default=LR_DEFAULT, type=float)

    # Optional checkpoint every n epochs
    parser.add_argument("--checkpoint", default=-1, type=int)
    # Optional random seed
    parser.add_argument("--seed", default=RANDOM_STATE_DEFAULT, type=int)

    args = parser.parse_args()

    # Seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(args.seed)

    # Load pre-trained model from huggingface
    model_name = HF_PRETRAINED_NAME if args.model is None else args.model
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name, truncation=True)
    # Add tokens and resize embeddings accordingly
    if args.model is None:
        tokenizer.add_tokens(TOKENIZER_EXTRA_TOKENS)
        model.resize_token_embeddings(len(tokenizer))

    # Load optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.5)

    # Load train set
    train_set = CardTextDataset(args.train_csv, args.ja, args.en)
    train_dataloader = DataLoader(train_set, args.batch_size, shuffle=True)

    # Load test set
    val_set = CardTextDataset(args.val_csv, args.ja, args.en)
    val_dataloader = DataLoader(val_set, args.batch_size)

    # Epoch and epoch losses
    training_epochs = range(1, args.epochs + 1)
    train_losses, val_losses = [], []

    # Begin training
    if args.checkpoint <= 0 or args.checkpoint > args.epochs:
        warnings.warn("Checkpoint saving disabled")
    print(f"Traingin for {args.epochs} epochs on device: {DEVICE}")
    model.to(DEVICE)

    try:
        for epoch in training_epochs:
            # Enter training mode
            model.train()
            train_loss = []
            tqdm_train_loader = tqdm(train_dataloader, leave=False)

            for src_str, tgt_str in tqdm_train_loader:
                src_tokenized = tokenizer(src_str, max_length=512, padding="max_length")
                tgt_tokenized = tokenizer(tgt_str, max_length=512, padding="max_length")

                kwargs = {
                    "input_ids": torch.tensor(src_tokenized["input_ids"], device=DEVICE),
                    "attention_mask": torch.tensor(src_tokenized["attention_mask"], device=DEVICE),
                    "labels": torch.tensor(tgt_tokenized["input_ids"], device=DEVICE),
                }

                optimizer.zero_grad()
                # Forward pass
                output = model(**kwargs)
                # Backward pass
                output.loss.backward()
                optimizer.step()

                # Save training batch loss and average
                train_loss.append(output.loss.item())
                tqdm_train_loader.set_description(desc=f"Epoch {epoch}: {np.average(train_loss):5f}")

            # Save train loss
            avg_train_loss = np.average(train_loss)

            # Update scheduler once per epoch
            scheduler.step()

            # Export if required
            if args.checkpoint > 0 and epoch % args.checkpoint == 0:
                model.save_pretrained(os.path.join(args.export, "checkpoints", f"model_checkpoint_{epoch}"))
                tokenizer.save_pretrained(os.path.join(args.export, "checkpoints", f"model_checkpoint_{epoch}"))

            # Enter evaluation mode
            model.eval()
            val_loss = []

            with torch.no_grad():
                for src_str, tgt_str in val_dataloader:
                    src_tokenized = tokenizer(src_str, max_length=512, padding="max_length")
                    tgt_tokenized = tokenizer(tgt_str, max_length=512, padding="max_length")

                    kwargs = {
                        "input_ids": torch.tensor(src_tokenized["input_ids"], device=DEVICE),
                        "attention_mask": torch.tensor(src_tokenized["attention_mask"], device=DEVICE),
                        "labels": torch.tensor(tgt_tokenized["input_ids"], device=DEVICE),
                    }

                    # Forward pass
                    output = model(**kwargs)

                    # Save validation loss and batch average
                    val_loss.append(output.loss.item())

            # Save validation loss
            avg_val_loss = np.average(val_loss)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # Report epoch loss
            print(f"Epoch {epoch}: train_loss={avg_train_loss: 5f}, val_loss={avg_val_loss: 5f}")

    except KeyboardInterrupt:
        print("Keyboard interrupt detected, aborting training")

    # Export last model and tokenizer
    model.save_pretrained(os.path.join(args.export, "checkpoints", "model_last"))
    tokenizer.save_pretrained(os.path.join(args.export, "checkpoints", "model_last"))


if __name__ == "__main__":
    main()
