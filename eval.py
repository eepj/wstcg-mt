# Import torch
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# Import huggingface pre-trained models and metrics
from transformers import MarianTokenizer, MarianMTModel
import evaluate

# Import additional libraries
import argparse
import os
import pandas as pd
import re
from tqdm import tqdm

from dataset import CardTextDataset

# Evaluation device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # Parse the input arguments
    parser = argparse.ArgumentParser()
    # Dataset CSV file path
    parser.add_argument("--test_csv", required=True)
    # Japanese text column name
    parser.add_argument("--ja", required=True)
    # English text column name
    parser.add_argument("--en", required=True)
    # Model and tokenizer path
    parser.add_argument("--model", required=True)
    # CSV export path
    parser.add_argument("--export", required=True)

    args = parser.parse_args()

    # Load trained models from paths
    model = MarianMTModel.from_pretrained(args.model)
    tokenizer = MarianTokenizer.from_pretrained(args.model)

    # Load BLEU evaluation metric
    metric_bleu = evaluate.load("bleu")
    metric_chrf = evaluate.load("chrf")

    # Load test set
    test_set = CardTextDataset(args.test_csv, args.ja, args.en)

    print(f"Evaluating {len(test_set)} instances on {DEVICE}")
    results = []
    model.to(DEVICE)
    model.eval()

    try:
        with torch.no_grad():
            for sample in tqdm(test_set):
                src_str, tgt_str = sample
                src_tensors = tokenizer(src_str, return_tensors="pt", padding=True)

                trans_tensor = model.generate(**src_tensors.to(DEVICE))
                # Convert output tensor to string
                trans_str = "".join([tokenizer.decode(t, skip_special_tokens=True) for t in trans_tensor])
                metric_bleu.add_batch(predictions=[trans_str], references=[tgt_str])
                metric_chrf.add_batch(predictions=[trans_str], references=[tgt_str])
                results.append({"src": src_str, "tgt": tgt_str, "pred": trans_str})

    except KeyboardInterrupt:
        print("Keyboard interrupt detected, aborting evaluation")

    finally:
        bleu = metric_bleu.compute()
        chrf = metric_chrf.compute()

        print(bleu)
        print(chrf)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.export, "result.csv"))


if __name__ == "__main__":
    main()
