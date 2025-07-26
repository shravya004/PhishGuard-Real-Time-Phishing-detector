import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from datasets import Dataset
import evaluate
import torch
df = pd.read_csv("clean_dataset.csv")
safe_urls = [
    "https://www.google.com", "https://www.gmail.com", "https://www.chat.openai.com",
    "https://www.youtube.com", "https://www.amazon.in", "https://www.facebook.com",
    "https://www.linkedin.com", "https://www.microsoft.com", "https://www.netflix.com",
    "https://www.instagram.com"
]
safe_df = pd.DataFrame({'text': safe_urls, 'label': 0})
df = pd.concat([df, safe_df], ignore_index=True)

# 2. Shuffle and split
df = df.sample(frac=0.5, random_state=42).reset_index(drop=True)
df["label"] = df["label"].astype(int)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 3. ML Model Inputs (hashed features)
def text_hash(x): return hash(x) % 10**8
train_df["features"] = train_df["text"].apply(text_hash)
test_df["features"] = test_df["text"].apply(text_hash)
X_train_ml = train_df["features"].values.reshape(-1, 1)
y_train_ml = train_df["label"]
X_test_ml = test_df["features"].values.reshape(-1, 1)
y_test_ml = test_df["label"]

# 4. ML Models
ml_models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(probability=True),
    "Gradient Boosting": GradientBoostingClassifier()
}
for name, model in ml_models.items():
    model.fit(X_train_ml, y_train_ml)
    y_pred = model.predict(X_test_ml)
    print(f"\n📊 ML Model Report ({name})")
    print(classification_report(y_test_ml, y_pred))

# 5. BERT-Tiny Transformer Model
train_hf = Dataset.from_pandas(train_df[["text", "label"]])
test_hf = Dataset.from_pandas(test_df[["text", "label"]])
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

train_hf = train_hf.map(tokenize_function, batched=True)
test_hf = test_hf.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="no"
)

accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return accuracy.compute(predictions=predictions.numpy(), references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_hf,
    eval_dataset=test_hf,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()
preds = trainer.predict(test_hf)
transformer_preds = torch.argmax(torch.tensor(preds.predictions), dim=-1).numpy()

print("\n🤖 Transformer Model Report (BERT-Tiny)")
print(classification_report(test_df["label"].values, transformer_preds))
