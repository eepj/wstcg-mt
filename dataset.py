from torch.utils.data import DataLoader, Dataset
import pandas as pd


class CardTextDataset(Dataset):
    def __init__(self, csv_path: str, src_col: str, tgt_col: str):
        super().__init__()
        df = pd.read_csv(csv_path)
        self.src = list(df[src_col])
        self.tgt = list(df[tgt_col])

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index: int) -> tuple[list[str], list[str]]:
        return self.src[index], self.tgt[index]
