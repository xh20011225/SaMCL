import csv
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import PepProDataset, collate_fn
from saved_model.SaMCL import SaMCL
from saprot_loader import load_saprot_with_lora


def mask_to_bin_seq(mask_arr):
    mask_arr = np.array(mask_arr, dtype=int)
    return "".join(map(str, mask_arr.tolist()))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_results = []

    for batch in tqdm(loader, desc="Testing"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        output = model(batch)

        inter_logits = output["inter_logits"]
        inter_probs = torch.softmax(inter_logits, dim=1)[:, 1].cpu().numpy()
        inter_preds = (inter_probs >= 0.5).astype(int)

        pep_logits = output["pep_logits"]
        pro_logits = output["pro_logits"]

        pep_pred = model.decode_pep(pep_logits, batch["pep_mask"]).cpu().numpy()
        pro_pred = model.decode_pro(pro_logits, batch["pro_mask"]).cpu().numpy()

        pdb_ids = batch["pdb_id"]
        pep_mask_bool = batch["pep_mask"].cpu().numpy()
        pro_mask_bool = batch["pro_mask"].cpu().numpy()

        for i, pdb_id in enumerate(pdb_ids):
            pep_len = int(pep_mask_bool[i].sum())
            pro_len = int(pro_mask_bool[i].sum())

            if inter_preds[i] == 0:
                pep_mask_trimmed = [0] * pep_len
                pro_mask_trimmed = [0] * pro_len
            else:
                pep_mask_trimmed = pep_pred[i][:pep_len].tolist()
                pro_mask_trimmed = pro_pred[i][:pro_len].tolist()

            result = {
                "pdb_id": pdb_id,
                "pred_inter_prob": float(inter_probs[i]),
                "pred_inter_label": int(inter_preds[i]),
                "pep_pred_mask": pep_mask_trimmed,
                "pro_pred_mask": pro_mask_trimmed
            }
            all_results.append(result)

    return all_results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    example_csv = "example.csv"
    model_path = "saved_model/best_model.pt"
    save_path = "example_pred.csv"

    saprot_model, saprot_tokenizer = load_saprot_with_lora(
        base_model_path="weights/PLMs/SaProt_650M_PDB",
        lora_weights_path="saprot_finetuned",
        device=device
    )

    model = SaMCL().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_dataset = PepProDataset(example_csv, saprot_model, saprot_tokenizer, device=device)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    preds = predict(model, test_loader, device)

    with open(example_csv, "r") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)

    drop_cols = {"peptide com_seq", "protein com_seq"}
    header = [h for h in header if h not in drop_cols]

    new_cols = ["pred_inter_prob", "pred_inter_label", "pep_pred_seq", "pro_pred_seq"]
    out_header = header + new_cols

    pdb2pred = {p["pdb_id"]: p for p in preds}
    output_rows = []

    for row in rows:
        pdb_id = row.get("PDB ID") or row.get("pdb_id")
        if pdb_id is None:
            continue

        pred = pdb2pred.get(pdb_id)
        if pred is None:
            continue

        pep_mask = pred["pep_pred_mask"]
        pro_mask = pred["pro_pred_mask"]

        row["pep_pred_seq"] = mask_to_bin_seq(pep_mask)
        row["pro_pred_seq"] = mask_to_bin_seq(pro_mask)
        row["pred_inter_prob"] = pred["pred_inter_prob"]
        row["pred_inter_label"] = pred["pred_inter_label"]

        for c in drop_cols:
            row.pop(c, None)

        output_rows.append(row)


    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_header)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nPredictions saved to: {save_path}")


if __name__ == "__main__":
    main()
