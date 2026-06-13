from .type_resources import build_definitions


TRIGGER_CFG = {"lr": 2e-5, "batch_size": 16, "max_len": 256, "max_span_len": 6, "epochs": 15}

ARG_CFG = {"lr": 2e-5, "batch_size": 16, "max_len": 256, "max_span_len": 10, "neg_ratio": 3, "epochs": 15}

BIO_CFG = {"lr": 2e-5, "batch_size": 16, "max_len": 256, "epochs": 15}

SFT_CFG = {
    "lr": 2e-4,
    "batch_size": 4,
    "grad_accum": 8,
    "max_len": 768,
    "epochs": 10,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
}


def build_definitions_safe(type_system, override_path):
    return build_definitions(type_system, override_path)
