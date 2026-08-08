from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("stanford-crfm/music-small-800k")

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                        # the rank — the bottleneck width
    lora_alpha=16,              # scaling (~2×r)
    lora_dropout=0.05,
    target_modules=["c_attn"],  # GPT-2's fused query/key/value attention projection
)

model = get_peft_model(base, lora_cfg)
model.print_trainable_parameters()