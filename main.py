import pandas as pd
import re

# load headline sarcasm
df1 = pd.read_json(
    "data/Sarcasm_Headlines_Dataset.json",
    lines=True
)

df2 = pd.read_json(
    "data/Sarcasm_Headlines_Dataset_v2.json",
    lines=True
)

# rename columns
df1 = df1.rename(columns={
    "headline":"text",
    "is_sarcastic":"label"
})[["text","label"]]

df2 = df2.rename(columns={
    "headline":"text",
    "is_sarcastic":"label"
})[["text","label"]]

# load reddit sarcasm
df3 = pd.read_csv(
    "data/train-balanced-sarc.csv",
    sep="\t",
    engine="python",
    on_bad_lines="skip",
    header=None
)

# inspect first!
print(df3.head())

# assign names
df3.columns = [
    "label","text1","author","subject",
    "score","col5","col6","date",
    "timestamp","text"
]

df3 = df3[["text","label"]]

# merge
sarcasm_df = pd.concat(
    [df1,df2,df3],
    ignore_index=True
)

# clean
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+","",text)
    text = re.sub(r"<.*?>","",text)
    text = re.sub(r"\s+"," ",text).strip()
    return text

sarcasm_df["text"] = sarcasm_df["text"].apply(clean_text)

print(sarcasm_df.head())
print(sarcasm_df.shape)

#splicting of data set into train and test + valid(80+10+10)
from sklearn.model_selection import train_test_split

# 80% training
train_df, temp_df = train_test_split(
    sarcasm_df,
    test_size=0.2,
    stratify=sarcasm_df["label"],
    random_state=42
)

# split remaining 20% → 10% validation + 10% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)

# save
train_df.to_csv("data/sarcasm_train.csv", index=False)
val_df.to_csv("data/sarcasm_val.csv", index=False)
test_df.to_csv("data/sarcasm_test.csv", index=False)

print("Saved successfully")