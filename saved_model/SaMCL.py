import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

class MSCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.kernal_sizes = kernel_sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, k, padding=k//2) for k in kernel_sizes
        ])
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        out = torch.stack(conv_outs, dim=0).sum(dim=0)
        out = self.bn(out)
        out = F.relu(out)
        return out.transpose(1, 2)


class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn = torch.matmul(Q, K.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        return out


class BiLSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn = SelfAttention(hidden_dim * 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out = self.attn(lstm_out)
        return attn_out

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x1, x2):
        Q = self.query(x1)
        K = self.key(x2)
        V = self.value(x2)
        attn = torch.matmul(Q, K.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        return out + x1


class SaMCL(nn.Module):
    def __init__(self, input_dim=1303, hidden_dim=256, num_tags=2, dropout_rate=0.3):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.pep_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.pro_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.pep_cnn = MSCNN(hidden_dim, hidden_dim)
        self.pro_cnn = MSCNN(hidden_dim, hidden_dim)

        self.pep_lstm = BiLSTMAttention(hidden_dim, hidden_dim)
        self.pro_lstm = BiLSTMAttention(hidden_dim, hidden_dim)

        self.global2local = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.cross_attn_pep = CrossAttention(hidden_dim)
        self.cross_attn_pro = CrossAttention(hidden_dim)

        # interaction branch
        self.inter_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2)
        )

        self.pep_ff = nn.Sequential(
            nn.Linear(hidden_dim, num_tags),
            nn.Dropout(dropout_rate)
        )
        self.pro_ff = nn.Sequential(
            nn.Linear(hidden_dim, num_tags),
            nn.Dropout(dropout_rate)
        )

        self.task_log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, batch, labels=None):
        pep_feat = torch.cat([batch["pep_onehot"], batch["pep_phys"], batch["pep_emb"]], dim=-1)
        pro_feat = torch.cat([batch["pro_onehot"], batch["pro_phys"], batch["pro_emb"]], dim=-1)

        pep_feat = self.pep_proj(pep_feat)
        pro_feat = self.pro_proj(pro_feat)

        pep_local = self.pep_cnn(pep_feat)
        pep_global = self.pep_lstm(pep_feat)
        pep_global = self.global2local(pep_global)
        pep_fused_base = pep_local + pep_global

        pro_local = self.pro_cnn(pro_feat)
        pro_global = self.pro_lstm(pro_feat)
        pro_global = self.global2local(pro_global)
        pro_fused_base = pro_local + pro_global

        pep_fused = self.cross_attn_pep(pep_fused_base, pro_fused_base)
        pro_fused = self.cross_attn_pro(pro_fused_base, pep_fused_base)

        pep_mask = batch.get("pep_mask", None)
        pro_mask = batch.get("pro_mask", None)

        if pep_mask is not None:
            pep_seq_feat = (pep_fused * pep_mask.unsqueeze(-1)).sum(dim=1) / (pep_mask.sum(dim=1, keepdim=True) + 1e-6)
        else:
            pep_seq_feat = pep_fused.mean(dim=1)

        if pro_mask is not None:
            pro_seq_feat = (pro_fused * pro_mask.unsqueeze(-1)).sum(dim=1) / (pro_mask.sum(dim=1, keepdim=True) + 1e-6)
        else:
            pro_seq_feat = pro_fused.mean(dim=1)

        pair_feat = torch.cat([pep_seq_feat, pro_seq_feat], dim=-1)
        inter_logits = self.inter_mlp(pair_feat)

        pep_logits = self.pep_ff(pep_fused)
        pro_logits = self.pro_ff(pro_fused)

        output = {
            "inter_logits": inter_logits,
            "pep_logits": pep_logits,
            "pro_logits": pro_logits
        }

        if labels is not None:
            interaction = labels["interaction"]
            inter_loss = F.cross_entropy(inter_logits, interaction)

            active_mask = (interaction == 1)
            if active_mask.sum() > 0:
                pep_loss = self.compute_residue_loss(
                    logits=pep_logits,
                    labels=labels["pep_label"],
                    mask=pep_mask,
                    sample_mask=active_mask
                )
                pro_loss = self.compute_residue_loss(
                    logits=pro_logits,
                    labels=labels["pro_label"],
                    mask=pro_mask,
                    sample_mask=active_mask
                )
            else:
                pep_loss = torch.zeros(1, device=pep_logits.device)
                pro_loss = torch.zeros(1, device=pro_logits.device)

            losses = torch.stack([inter_loss, pep_loss, pro_loss])
            precisions = torch.exp(-self.task_log_vars)
            weighted = precisions * losses + self.task_log_vars
            total_loss = weighted.sum()

            output.update({
                "loss": total_loss,
                "inter_loss": inter_loss.detach(),
                "pep_loss": pep_loss.detach(),
                "pro_loss": pro_loss.detach(),
            })

        return output

    def compute_residue_loss(self, logits, labels, mask=None, sample_mask=None):
        B, L, C = logits.shape
        per_pos_loss = F.cross_entropy(logits.view(-1, C), labels.view(-1), reduction="none")
        per_pos_loss = per_pos_loss.view(B, L)

        if mask is not None:
            mask_f = mask.float()
            per_pos_loss = per_pos_loss * mask_f
            valid_per_sample = mask_f.sum(dim=1)
        else:
            valid_per_sample = torch.full((B,), L, device=logits.device, dtype=torch.float)

        eps = 1e-6
        per_sample_loss = per_pos_loss.sum(dim=1) / (valid_per_sample + eps)

        if sample_mask is not None:
            sel = sample_mask.bool()
            if sel.sum() == 0:
                return torch.zeros(1, device=logits.device)
            loss = per_sample_loss[sel].mean()
        else:
            loss = per_sample_loss.mean()

        return loss

    def decode_pep(self, pep_logits, pep_mask=None):
        preds = torch.argmax(pep_logits, dim=-1)
        if pep_mask is not None:
            preds = preds * pep_mask.long()
        return preds

    def decode_pro(self, pro_logits, pro_mask=None):
        preds = torch.argmax(pro_logits, dim=-1)
        if pro_mask is not None:
            preds = preds * pro_mask.long()
        return preds