import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import EsmTokenizer
from model.saprot.base import SaprotBaseModel


def load_saprot_with_lora(base_model_path, lora_weights_path, device):
    config = {
        "task": "base",
        "config_path": base_model_path,
        "load_pretrained": True,
    }
    saprot_base = SaprotBaseModel(**config).to(device)
    tokenizer = EsmTokenizer.from_pretrained(lora_weights_path)  # 使用微调后tokenizer

    target_modules = []
    for i in range(33):
        target_modules.append(f"esm.encoder.layer.{i}.attention.self.query")
        target_modules.append(f"esm.encoder.layer.{i}.attention.self.value")

    peft_config = LoraConfig(
        task_type="FEATURE_EXTRACTION",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
    )

    saprot_lora = get_peft_model(saprot_base.model, peft_config)

    lora_state_dict = torch.load(f"{lora_weights_path}/best_model_weights.pth", map_location=device)
    set_peft_model_state_dict(saprot_lora, lora_state_dict)

    print("SaProt + LoRA successfully load !")

    saprot_base.model = saprot_lora

    return saprot_base, tokenizer
