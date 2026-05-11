import pandas as pd

train_df = pd.read_csv("data/sarcasm_train.csv")
val_df = pd.read_csv("data/sarcasm_val.csv")
test_df = pd.read_csv("data/sarcasm_test.csv")

print(train_df.head())
print(train_df.shape)

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
sample = tokenizer(
    "great another monday",
    padding="max_length",
    truncation=True,
    max_length=128
)

print(sample)


import torch
from torch.utils.data import Dataset

class SarcasmDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            str(self.texts[idx]),
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
        
        
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)

torch.save(model.state_dict(), "models/sarcasm_bert.pt")