import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F

# one-hot encoding
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def one_hot_encode(seq, max_len):
    encoding = torch.zeros(max_len, len(AMINO_ACIDS))
    for i, aa in enumerate(seq[:max_len]):
        if aa in AA_TO_IDX:
            encoding[i, AA_TO_IDX[aa]] = 1
    return encoding



# Physico encoding
PHYSICOCHEM = {
    "A": [1.8,  0,  0],
    "C": [2.5,  0,  1],
    "D": [-3.5, -1, 1],
    "E": [-3.5, -1, 1],
    "F": [2.8,  0,  0],
    "G": [-0.4, 0,  0],
    "H": [-3.2, 1,  1],
    "I": [4.5,  0,  0],
    "K": [-3.9, 1,  1],
    "L": [3.8,  0,  0],
    "M": [1.9,  0,  0],
    "N": [-3.5, 0,  1],
    "P": [-1.6, 0,  0],
    "Q": [-3.5, 0,  1],
    "R": [-4.5, 1,  1],
    "S": [-0.8, 0,  1],
    "T": [-0.7, 0,  1],
    "V": [4.2,  0,  0],
    "W": [-0.9, 0,  0],
    "Y": [-1.3, 0,  1],
}

def physico_encode(seq, max_len):
    dim = len(next(iter(PHYSICOCHEM.values())))
    encoding = torch.zeros(max_len, dim)
    for i, aa in enumerate(seq[:max_len]):
        if aa in PHYSICOCHEM:
            encoding[i] = torch.tensor(PHYSICOCHEM[aa])
    return encoding



class PepProDataset(Dataset):
    def __init__(self, csv_file, saprot_model, saprot_tokenizer, pro_max_len=800, pep_max_len=50, device="cuda"):
        self.data = pd.read_csv(csv_file)
        self.pro_max_len = pro_max_len
        self.pep_max_len = pep_max_len
        self.saprot_model = saprot_model.to(device)
        self.saprot_model.eval()
        self.saprot_tokenizer = saprot_tokenizer
        self.device = device


    def __len__(self):
        return len(self.data)


    def __getitem__(self, item):
        row = self.data.iloc[item]

        pdb_id = row["PDB ID"]
        pep_seq = row["peptide sequence"]
        pro_seq = row["protein sequence"]
        pep_com_seq = row["peptide com_seq"]
        pro_com_seq = row["protein com_seq"]

        pep_label = torch.tensor([int(x) for x in str(row["peptide label"])], dtype=torch.long)
        pro_label = torch.tensor([int(x) for x in str(row["protein label"])], dtype=torch.long)
        interaction_raw = str(row["interaction"]).strip()
        interaction = torch.tensor(int(interaction_raw), dtype=torch.long)

        pep_onehot = one_hot_encode(pep_seq, self.pep_max_len)
        pro_onehot = one_hot_encode(pro_seq, self.pro_max_len)
        pep_phys = physico_encode(pep_seq, self.pep_max_len)
        pro_phys = physico_encode(pro_seq, self.pro_max_len)

        pep_inputs = self.saprot_tokenizer(pep_com_seq, return_tensors="pt",
                                           truncation=True, padding=True, max_length=self.pep_max_len)
        pro_inputs = self.saprot_tokenizer(pro_com_seq, return_tensors="pt",
                                           truncation=True, padding=True, max_length=self.pro_max_len)
        pep_inputs = {k: v.to(self.device) for k, v in pep_inputs.items()}
        pro_inputs = {k: v.to(self.device) for k, v in pro_inputs.items()}

        with torch.no_grad():
            pep_emb = self.saprot_model.get_hidden_states(pep_inputs, reduction=None)
            pro_emb = self.saprot_model.get_hidden_states(pro_inputs, reduction=None)

        if isinstance(pep_emb, (list, tuple)):
            pep_emb = pep_emb[0]
        if isinstance(pro_emb, (list, tuple)):
            pro_emb = pro_emb[0]

        pep_emb = (pep_emb - pep_emb.mean(dim=-1, keepdim=True)) / (pep_emb.std(dim=-1, keepdim=True) + 1e-6)
        pro_emb = (pro_emb - pro_emb.mean(dim=-1, keepdim=True)) / (pro_emb.std(dim=-1, keepdim=True) + 1e-6)

        return {
            "pdb_id": pdb_id,
            "pep_seq": pep_seq,
            "pro_seq": pro_seq,
            "pep_onehot": pep_onehot,
            "pro_onehot": pro_onehot,
            "pep_phys": pep_phys,
            "pro_phys": pro_phys,
            "pep_emb": pep_emb,
            "pro_emb": pro_emb,
            "pep_label": pep_label,
            "pro_label": pro_label,
            "interaction": interaction,
        }

def collate_fn(batch, pep_max_len=50, pro_max_len=800):
    d_pep = batch[0]["pep_emb"].shape[1]
    d_pro = batch[0]["pro_emb"].shape[1]

    def pad_or_truncate(x, max_len, d):
        device = x.device  # 获取原 tensor 所在设备
        seq_len = x.shape[0]
        if seq_len >= max_len:
            padded = x[:max_len]
            mask = torch.ones(max_len, dtype=torch.bool, device=device)
        else:
            pad_len = max_len - seq_len
            padded = torch.cat([x, torch.zeros(pad_len, d, device=device)], dim=0)
            mask = torch.cat([torch.ones(seq_len, dtype=torch.bool, device=device),
                              torch.zeros(pad_len, dtype=torch.bool, device=device)])
        return padded, mask

    pep_emb_list, pep_mask_list = zip(*[pad_or_truncate(x["pep_emb"], pep_max_len, d_pep) for x in batch])
    pro_emb_list, pro_mask_list = zip(*[pad_or_truncate(x["pro_emb"], pro_max_len, d_pro) for x in batch])
    pep_emb = torch.stack(pep_emb_list)
    pro_emb = torch.stack(pro_emb_list)
    pep_mask = torch.stack(pep_mask_list)
    pro_mask = torch.stack(pro_mask_list)

    def pad_label(x, max_len):
        device = x.device
        seq_len = x.shape[0]
        if seq_len >= max_len:
            return x[:max_len]
        else:
            return torch.cat([x, torch.zeros(max_len - seq_len, dtype=torch.long, device=device)], dim=0)

    pep_labels = torch.stack([pad_label(x["pep_label"], pep_max_len) for x in batch])
    pro_labels = torch.stack([pad_label(x["pro_label"], pro_max_len) for x in batch])

    pep_onehot = torch.stack([x["pep_onehot"] for x in batch])
    pro_onehot = torch.stack([x["pro_onehot"] for x in batch])
    pep_phys = torch.stack([x["pep_phys"] for x in batch])
    pro_phys = torch.stack([x["pro_phys"] for x in batch])

    interactions = torch.tensor([int(x["interaction"]) for x in batch], dtype=torch.long)
    pdb_ids = [x["pdb_id"] for x in batch]

    return {
        "pdb_id": pdb_ids,
        "pep_onehot": pep_onehot,
        "pro_onehot": pro_onehot,
        "pep_phys": pep_phys,
        "pro_phys": pro_phys,
        "pep_emb": pep_emb,
        "pro_emb": pro_emb,
        "pep_label": pep_labels,
        "pro_label": pro_labels,
        "pep_mask": pep_mask,
        "pro_mask": pro_mask,
        "interaction": interactions
    }