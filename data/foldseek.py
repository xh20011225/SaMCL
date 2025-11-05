import os
import pandas as pd
from utils.foldseek_util import get_struc_seq

pdb_dir = "pdb_files"
output_file = "foldseek.csv"
results = []
count_file = 0
count_chain = 0

for filename in os.listdir(pdb_dir):
    if filename.endswith(".cif"):
        pdb_path = os.path.join(pdb_dir, filename)
        pdb_id = os.path.splitext(filename)[0]
        count_file += 1

        try:
            parsed_seqs = get_struc_seq("bin/foldseek", pdb_path, plddt_mask=False)

            for chain, (aa_seq, struct_seq, combined_seq) in parsed_seqs.items():

                results.append({
                    "PDB ID": pdb_id,
                    "Chain": chain,
                    "AA_Seq": aa_seq,
                    "Struct_Seq": struct_seq,
                    "Combined_Seq": combined_seq
                })

                count_chain += 1

        except Exception as e:
            print(f"Error {filename}: {e}")

        print(f"file_count: {count_file}，data_count: {count_chain}")

df = pd.DataFrame(results)
df.to_csv(output_file, index=False)