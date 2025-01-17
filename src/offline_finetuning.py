from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B",
    load_in_4bit = True,
    max_seq_length = 1024,
    dtype = None
)

#Lora Parameters https://github.com/unslothai/unsloth/wiki#lora-parameters-encyclopedia
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 3407,
    use_rslora = False, # rank stabilized LoRA
    loftq_config = None
)