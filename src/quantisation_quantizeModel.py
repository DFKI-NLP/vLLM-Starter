from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import os


from other.utils import load_model

cache_dir = "/netscratch/thomas/models/"

model_path = 'mistralai/Mistral-7B-Instruct-v0.2'
quant_model = 'mistral-instruct-v0.2-awq'
quant_path = os.path.join(cache_dir, quant_model)

quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(load_model(model_path, model_path=cache_dir), trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')