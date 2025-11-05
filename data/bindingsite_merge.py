import pandas as pd

donor_df = pd.read_csv("DonorSeqLabel_peptide_5.0_Any.csv")
pro_df = pd.read_csv("ProSeqLabel_peptide_5.0_Any.csv")

merged_df = pd.merge(donor_df, pro_df, on="Binding ID", how="inner")
final_df = merged_df[[
    "Binding ID",
    "Donor sequence",
    "Binding label",
    "Complex protein chain sequence",
    "Complex protein chain sequence label"
]]
final_df.to_csv("bindingsite.csv", index=False)

print("Save as bindingsite.csv !")

