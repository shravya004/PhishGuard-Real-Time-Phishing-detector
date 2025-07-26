import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv("final.csv")  # Update with actual filename if different

# Step 2: Keep only the 'URL' and 'label' columns
df = df[["URL", "label"]]

# Step 3: Rename for compatibility with training script
df = df.rename(columns={"URL": "text", "label": "label"})

# Step 4: Drop any rows with missing or empty URLs
df = df.dropna(subset=["text", "label"])
df = df[df["text"].str.strip() != ""]

# Step 5: Convert label to int if it's not already
df["label"] = df["label"].astype(int)

# Step 6: Save the clean dataset
df.to_csv("clean_phishing_dataset.csv", index=False)
print("✅ Clean dataset saved as clean_phishing_dataset.csv")
print(df.head())
