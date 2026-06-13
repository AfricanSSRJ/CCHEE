import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFGenerator:
    def __init__(self, model_name, lora_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        if lora_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.to(self.device)
        self.model.eval()

    def _format(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def generate(self, messages, max_new_tokens=512, temperature=0.0):
        prompt = self._format(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        do_sample = temperature > 0
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)


class SFTDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=768):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]["messages"]
        target = self.data[idx]["output"]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        target_ids = target_ids + [self.tokenizer.eos_token_id]
        input_ids = (prompt_ids + target_ids)[: self.max_len]
        labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_len]
        return {"input_ids": input_ids, "labels": labels}


def sft_collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, mask = [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        mask.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(mask),
    }


def build_lora_model(model_name, rank=16, alpha=32, dropout=0.05):
    from peft import LoraConfig, get_peft_model

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def train_sft(model_name, train_samples, output_dir, cfg):
    from functools import partial

    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = build_lora_model(model_name, cfg["lora_rank"], cfg["lora_alpha"], cfg["lora_dropout"])
    model.to(device)
    model.train()

    dataset = SFTDataset(train_samples, tokenizer, cfg["max_len"])
    collate = partial(sft_collate, pad_id=tokenizer.pad_token_id)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    accum = cfg.get("grad_accum", 1)
    for _ in range(cfg["epochs"]):
        optimizer.zero_grad()
        for step, batch in enumerate(loader):
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            (out.loss / accum).backward()
            if (step + 1) % accum == 0:
                optimizer.step()
                optimizer.zero_grad()
        optimizer.step()
        optimizer.zero_grad()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
