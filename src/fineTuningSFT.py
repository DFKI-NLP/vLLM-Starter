"""
srun --partition=RTXA6000-SLT \
     --job-name=vllm-test \
     --export=ALL,HF_HUB_CACHE=/ds/models/hf-cache-slt/ \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=1 \
     --mem-per-cpu=4G \
    python fineTuningSFT.py


srun --partition=RTXA6000-SLT \
     --job-name=vllm-test \
     --export=ALL,HF_HUB_CACHE=/ds/models/hf-cache-slt/ \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=1 \
     --mem-per-cpu=4G \
    python offline_simpleInference.py --model_name="/ds/models/hf-cache-slt/myAwesomeModel" --prompt="Is 9.11 larger than 9.9?"
"""
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
import os

# 1. Load model for PEFT
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
    device_map="auto",  # Auto-allocate devices
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
    use_rslora=True,
    use_gradient_checkpointing="unsloth"
)

model.print_trainable_parameters()  # No need for print()

# 2. Prepare data and tokenizer, convert dataset to chatml format
tokenizer = get_chat_template(
    tokenizer,
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    chat_template="chatml",
)


def apply_template(examples):
    return {"text": [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
                     for msg in examples["conversations"]]}


dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = dataset.map(apply_template, batched=True, remove_columns=dataset.column_names)

print("Sample from dataset after template application:")
print(dataset[0]["text"])

# 3. Training
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        seed=0,
    ),
)

trainer.train()

# 4. Load model for inference
model = FastLanguageModel.for_inference(model)

# Make sure you're using the same template format as during training
messages = [{"from": "human", "value": "Is 9.11 larger than 9.9?"}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,  # First get the raw text
    add_generation_prompt=True
)
print("Generated prompt:", prompt)  # Inspect the actual prompt

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

# Generate with appropriate parameters
outputs = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.0,
    top_p=0.9,
    do_sample=True
)

# Decode only the new tokens, not the entire sequence
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("Response:", response)

print("Messages:")
print(messages)
print("Outputs:")
print(outputs)
print("Decoded texts")
print(response)

print("---")

# 5. Save model (Ensure directory exists)
save_path = "/ds/models/hf-cache-slt/myAwesomeModel"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained_merged(save_path, tokenizer, save_method="merged_16bit")
