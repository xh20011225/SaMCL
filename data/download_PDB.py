import pandas as pd
import requests
import os
from time import sleep
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

df = pd.read_csv("PDB_IDs.csv", encoding="utf-8-sig", header=None)
df.columns = ["PDB ID"]
df["PDB ID"] = df["PDB ID"].str.lower()

ids = ",".join(df["PDB ID"].astype(str))
with open("PDB_IDs.txt", "w") as f:
    f.write(ids)

output_dir = "pdb_files"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("PDB_IDs.csv", encoding="utf-8-sig", header=None)
df.columns = ["PDB ID"]
pdb_ids = df["PDB ID"].str.upper().tolist()

url_template = "https://files.rcsb.org/download/{}.cif"

for pdb_id in pdb_ids:
    cif_path = os.path.join(output_dir, f"{pdb_id}.cif")

    if os.path.exists(cif_path):
        print(f"{pdb_id}.cif already download!")
        continue

    url = url_template.format(pdb_id)

    try:
        resp = requests.get(url, timeout=15, verify=False)
        if resp.status_code == 200:
            with open(cif_path, "wb") as f:
                f.write(resp.content)
            print(f"Successful: {pdb_id}.cif")
        else:
            print(f"Error: {pdb_id}.cif，Code: {resp.status_code}")
    except requests.RequestException as e:
        print(f"Request Eroor! {pdb_id}.cif: {e}")

    sleep(0.5)
