from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt
import os
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from unsloth.chat_templates import get_chat_template
import time

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 4bit for 405b!
    "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!

    "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2 models
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",

    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit" # NEW! Llama 3.3 70B!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct", # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

#LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


#We use Maxime Labonne's FineTome-100k dataset in ShareGPT style.
# But we convert it to HuggingFace's normal multiturn format ("role", "content") instead of ("from", "value")/ Llama-3 renders multi turn conversations like below:
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
pass

dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
#{'conversations': [{'from': 'human', 'value': 'Explain what boolean operators are, what they do, and provide examples of how they can be used in programming. Additionally, describe the concept of operator precedence and provide examples of how it affects the evaluation of boolean expressions. Discuss the difference between short-circuit evaluation and normal evaluation in boolean expressions and demonstrate their usage in code. \n\nFurthermore, add the requirement that the code must be written in a language that does not support short-circuit evaluation natively, forcing the test taker to implement their own logic for short-circuit evaluation.\n\nFinally, delve into the concept of truthiness and falsiness in programming languages, explaining how it affects the evaluation of boolean expressions. Add the constraint that the test taker must write code that handles cases where truthiness and falsiness are implemented differently across different programming languages.'}, {'from': 'gpt', 'value': 'Boolean operators are logical operators used in programming to manipulate boolean values. They operate on one or more boolean operands and return a boolean result. The three main boolean operators are "AND" (&&), "OR" (||), and "NOT" (!).\n\nThe "AND" operator returns true if both of its operands are true, and false otherwise. For example:\n\n```python\nx = 5\ny = 10\nresult = (x > 0) and (y < 20)  # This expression evaluates to True\n```\n\nThe "OR" operator returns true if at least one of its operands is true, and false otherwise. For example:\n\n```python\nx = 5\ny = 10\nresult = (x > 0) or (y < 20)  # This expression evaluates to True\n```\n\nThe "NOT" operator negates the boolean value of its operand. It returns true if the operand is false, and false if the operand is true. For example:\n\n```python\nx = 5\nresult = not (x > 10)  # This expression evaluates to True\n```\n\nOperator precedence refers to the order in which operators are evaluated in an expression. It ensures that expressions are evaluated correctly. In most programming languages, logical AND has higher precedence than logical OR. For example:\n\n```python\nresult = True or False and False  # This expression is evaluated as (True or (False and False)), which is True\n```\n\nShort-circuit evaluation is a behavior where the second operand of a logical operator is not evaluated if the result can be determined based on the value of the first operand. In short-circuit evaluation, if the first operand of an "AND" operator is false, the second operand is not evaluated because the result will always be false. Similarly, if the first operand of an "OR" operator is true, the second operand is not evaluated because the result will always be true.\n\nIn programming languages that support short-circuit evaluation natively, you can use it to improve performance or avoid errors. For example:\n\n```python\nif x != 0 and (y / x) > 10:\n    # Perform some operation\n```\n\nIn languages without native short-circuit evaluation, you can implement your own logic to achieve the same behavior. Here\'s an example in pseudocode:\n\n```\nif x != 0 {\n    if (y / x) > 10 {\n        // Perform some operation\n    }\n}\n```\n\nTruthiness and falsiness refer to how non-boolean values are evaluated in boolean contexts. In many programming languages, non-zero numbers and non-empty strings are considered truthy, while zero, empty strings, and null/None values are considered falsy.\n\nWhen evaluating boolean expressions, truthiness and falsiness come into play. For example:\n\n```python\nx = 5\nresult = x  # The value of x is truthy, so result is also truthy\n```\n\nTo handle cases where truthiness and falsiness are implemented differently across programming languages, you can explicitly check the desired condition. For example:\n\n```python\nx = 5\nresult = bool(x)  # Explicitly converting x to a boolean value\n```\n\nThis ensures that the result is always a boolean value, regardless of the language\'s truthiness and falsiness rules.'}], 'source': 'infini-instruct-top-500k', 'score': 5.212620735168457}

dataset = standardize_sharegpt(dataset)
#Changed cinversations 'from' to 'role' and 'value' to 'content' and 'roletype' to 'user' instead of 'human' and 'assistant' and 'gpt'
#{'conversations': [{'content': 'How do astronomers determine the original wavelength of light emitted by a celestial body at rest, which is necessary for measuring its speed using the Doppler effect?', 'role': 'user'}, {'content': 'Astronomers make use of the unique spectral fingerprints of elements found in stars. These elements emit and absorb light at specific, known wavelengths, forming an absorption spectrum. By analyzing the light received from distant stars and comparing it to the laboratory-measured spectra of these elements, astronomers can identify the shifts in these wavelengths due to the Doppler effect. The observed shift tells them the extent to which the light has been redshifted or blueshifted, thereby allowing them to calculate the speed of the star along the line of sight relative to Earth.', 'role': 'assistant'}], 'source': 'WebInstructSub_axolotl', 'score': 5.025244235992432}

dataset = dataset.map(formatting_prompts_func, batched = True,)
#{'Conversations': [{'Content': 'How Do Astronomers Determine The Original Wavelength Of Light Emitted By A Celestial Body At Rest, Which Is Necessary For Measuring Its Speed Using The Doppler Effect?', 'Role': 'User'}, {'Content': 'Astronomers Make Use Of The Unique Spectral Fingerprints Of Elements Found In Stars. These Elements Emit And Absorb Light At Specific, Known Wavelengths, Forming An Absorption Spectrum. By Analyzing The Light Received From Distant Stars And Comparing It To The Laboratory-Measured Spectra Of These Elements, Astronomers Can Identify The Shifts In These Wavelengths Due To The Doppler Effect. The Observed Shift Tells Them The Extent To Which The Light Has Been Redshifted Or Blueshifted, Thereby Allowing Them To Calculate The Speed Of The Star Along The Line Of Sight Relative To Earth.', 'Role': 'Assistant'}], 'Source': 'Webinstructsub_Axolotl', 'Score': 5.025244235992432, 'Text': '<|Begin_Of_Text|><|Start_Header_Id|>System<|End_Header_Id|>\N\Ncutting Knowledge Date: December 2023\Ntoday Date: 26 July 2024\N\N<|Eot_Id|><|Start_Header_Id|>User<|End_Header_Id|>\N\Nhow Do Astronomers Determine The Original Wavelength Of Light Emitted By A Celestial Body At Rest, Which Is Necessary For Measuring Its Speed Using The Doppler Effect?<|Eot_Id|><|Start_Header_Id|>Assistant<|End_Header_Id|>\N\Nastronomers Make Use Of The Unique Spectral Fingerprints Of Elements Found In Stars. These Elements Emit And Absorb Light At Specific, Known Wavelengths, Forming An Absorption Spectrum. By Analyzing The Light Received From Distant Stars And Comparing It To The Laboratory-Measured Spectra Of These Elements, Astronomers Can Identify The Shifts In These Wavelengths Due To The Doppler Effect. The Observed Shift Tells Them The Extent To Which The Light Has Been Redshifted Or Blueshifted, Thereby Allowing Them To Calculate The Speed Of The Star Along The Line Of Sight Relative To Earth.<|Eot_Id|>'}
#Added 'Text' key with the text of the conversation


#Train

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB
    ),
)

#We also use Unsloth's train_on_completions method to only train on the assistant outputs and ignore the loss on the user's inputs.
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)


print(tokenizer.decode(trainer.train_dataset[5]["input_ids"]))
space = tokenizer(" ", add_special_tokens = False).input_ids[0]
print(tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]]))

trainer_stats = trainer.train()

#Inference
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [
    {"role": "user", "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True,
                         temperature = 1.5, min_p = 0.1)
print(tokenizer.batch_decode(outputs))

# 5. Save model (Ensure directory exists)
save_path = "/ds/models/hf-cache-slt/myAwesomeModel"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained_merged(save_path, tokenizer, save_method="merged_16bit")