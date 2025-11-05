import pandas as pd

dataset_df = pd.read_csv("bidingsite.csv")
foldseek_df = pd.read_csv("foldseek.csv")

split_cols = dataset_df["Binding ID"].str.split("_", expand=True)
dataset_df["pdbid"] = split_cols[0].str.upper()   # PDB ID 转大写
dataset_df["pep_chain"] = split_cols[1]
dataset_df["pro_chain"] = split_cols[2]

pep_df = foldseek_df.rename(columns={"Combined_Seq": "pep Combined_seq", "AA_Seq": "pep_AA_Seq"})
pro_df = foldseek_df.rename(columns={"Combined_Seq": "pro Combined_seq", "AA_Seq": "pro_AA_Seq"})

merged_df = pd.merge(
    dataset_df,
    pep_df[["PDB ID", "Chain", "pep_AA_Seq", "pep Combined_seq"]],
    left_on=["pdbid", "pep_chain"],
    right_on=["PDB ID", "Chain"],
    how="left",
    suffixes=("", "_pep")
)
merged_df = merged_df[merged_df["Donor sequence"].astype(str) == merged_df["pep_AA_Seq"]]

merged_df = pd.merge(
    merged_df,
    pro_df[["PDB ID", "Chain", "pro_AA_Seq", "pro Combined_seq"]],
    left_on=["pdbid", "pro_chain"],
    right_on=["PDB ID", "Chain"],
    how="left",
    suffixes=("", "_pro")
)
merged_df = merged_df[merged_df["Complex protein chain sequence"].astype(str) == merged_df["pro_AA_Seq"]]

final_df = merged_df[[
    "Binding ID",
    "Donor sequence",
    "Binding label",
    "pep Combined_seq",
    "Complex protein chain sequence",
    "Complex protein chain sequence label",
    "pro Combined_seq"
]]

final_df = final_df.dropna(subset=["pep Combined_seq", "pro Combined_seq"])
final_df.to_csv("Dataset.csv", index=False)

print(f"Save as Dataset.csv，data num: {len(final_df)}")





