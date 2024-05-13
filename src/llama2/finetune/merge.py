import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from src.utils.defaults import DATA_DIR, LLAMA_DIR, LLAMA_HUGGINGFACE_CHECKPOINT

device_map = {"": 0}

# for NYU HPC
# DATA_DIR = Path("/scratch/sjb8193")
# LLAMA_HUGGINGFACE_CHECKPOINT = Path("/scratch/sjb8193/models_hf/7B")

model_path = str(LLAMA_HUGGINGFACE_CHECKPOINT)
adapter_path = str(LLAMA_DIR / "7B_finetuned")

base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
model = model.merge_and_unload()

merged_model_path = DATA_DIR / "doc2query-llama-2-7b-merged"
model.save_pretrained(merged_model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained(merged_model_path)
