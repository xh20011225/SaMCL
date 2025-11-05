import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
import os

data_path = "Dataset.csv"
test_size = 0.2
n_splits = 5
random_state = 42
output_dir = "dataset_splits"
os.makedirs(output_dir, exist_ok=True)


df_pos = pd.read_csv(data_path)
df_pos["interaction"] = 1
print(f"Positive: {len(df_pos)}")

def generate_neg_samples(pos_df, n_samples, random_state=42):
    rng = np.random.default_rng(random_state)
    peptides = df_pos[["peptide sequence", "peptide label", "peptide com_seq"]].drop_duplicates().reset_index(drop=True)
    proteins = df_pos[["protein sequence", "protein label", "protein com_seq"]].drop_duplicates().reset_index(drop=True)
    pos_pairs = set(zip(pos_df["protein sequence"], pos_df["peptide sequence"]))

    neg_list = []
    tries = 0
    max_tries = n_samples * 50
    while len(neg_list) < n_samples and tries < max_tries:
        tries += 1
        p_idx = rng.integers(0, len(proteins))
        pep_idx = rng.integers(0, len(peptides))
        p_row = proteins.iloc[p_idx]
        pep_row = peptides.iloc[pep_idx]
        pair = (p_row["protein sequence"], pep_row["peptide sequence"])
        if pair in pos_pairs:
            continue

        pep_seq = str(pep_row["peptide sequence"])
        pro_seq = str(p_row["protein sequence"])
        pep_label = "0" * len(pep_seq)
        pro_label = "0" * len(pro_seq)

        neg_entry = {
            "PDB ID": f"NEG_{len(neg_list)+1}",
            "peptide sequence": pep_seq,
            "peptide label": pep_label,
            "peptide com_seq": pep_row["peptide com_seq"],
            "protein sequence": pro_seq,
            "protein label": pro_label,
            "protein com_seq": p_row["protein com_seq"],
            "interaction": 0
        }
        neg_list.append(neg_entry)

    if len(neg_list) < n_samples:
        raise  RuntimeError(f"Runtime Error !")

    return pd.DataFrame(neg_list)


print("Generating negatives...")
neg_df = generate_neg_samples(df_pos, len(df_pos), random_state=random_state)
all_df = pd.concat([df_pos, neg_df], ignore_index=True)
all_df = all_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
train_df, test_df = train_test_split(
    all_df, test_size=test_size, stratify=all_df["interaction"], random_state=random_state
)
test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["interaction"]), start=1):
    train_fold = train_df.iloc[train_idx].reset_index(drop=True)
    val_fold = train_df.iloc[val_idx].reset_index(drop=True)
    train_fold.to_csv(os.path.join(output_dir, f"train_fold_{fold}.csv"), index=False)
    val_fold.to_csv(os.path.join(output_dir, f"val_fold_{fold}.csv"), index=False)

print("Done!")
