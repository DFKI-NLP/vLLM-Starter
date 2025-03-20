# vLLM-Starter
This repository provides a starting point for using the vLLM library on the cluster.
vLLM supports both offline and online usage:

- **Offline Mode**: Load the model once and process your data locally.
- **Online Mode**: Start a service to generate text via an endpoint, similar to OpenAI's GPT API.

This guide focuses on **offline usage**, as the online service may block cluster resources.

## Installation
```bash
conda create -n vLLM-Starter python=3.11 -y
conda activate vLLM-Starter
pip install vllm
```

## Download LLM models
It is recommended to store the LLM models in a central location, as these large files can be shared across different users.
By default, we use `/ds/models/llms/cache` as the storage location, where several LLMs are already stored. 
To set this directory as the default cache location, you need to configure the environment variable `HF_HUB_CACHE`.
In the examples we define the cache location in the srun command.

## Run script on cluster
Please replace **{script.py}** with your script.
```bash
srun --partition=RTXA6000-SLT \
     --job-name=vllm-test \
    --export=ALL,HF_HUB_CACHE=/ds/models/hf-cache-slt/ \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=1 \
     --mem-per-cpu=4G \
     --time=1-00:00:00 \
    python {script.py}
```

## Offline use

### Simple example
A simple example demonstrating how to use vLLM in offline mode can be found in the file [offline_simpleInference.py](./src/offline_simpleInference.py). 
This script loads a model and generates text based on a prompt.
<details>
    <summary>Example</summary>

```bash
srun --partition=RTXA6000-SLT \
     --job-name=vllm-test \
     --export=ALL,HF_HUB_CACHE=/ds/models/hf-cache-slt/ \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=1 \
     --mem-per-cpu=4G \
    python offline_simpleInference.py
```
</details>


### Chat-style example
The following script demonstrates how to load a model and generate text based on a chat-style prompt. 
You can find the more complex example in the file [offline_chatstyle.py](./src/offline_chatstyle.py).
<details>
    <summary>Example</summary>

```bash
srun --partition=RTXA6000-SLT \
     --job-name=vllm-test \
     --export=ALL,HF_HUB_CACHE=/ds/models/hf-cache-slt/ \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=1 \
     --mem-per-cpu=4G \
    python offline_chatstyle.py
```
</details>

### Structured output example
The `GuidedDecodingParams` class in vLLM allows you to define the output structure for tasks that require a predefined format, such as Named Entity Recognition (NER). 
You can use various methods to guide the decoding process, including regular expressions, JSON objects, grammar, or simple binary choices.
The example can be found in the file [offline_structuredOutput.py](./src/offline_structuredOutput.py).
<details>
    <summary>Example</summary>

```bash
srun --partition=RTXA6000-SLT \
     --job-name=vllm-test \
     --export=ALL,HF_HUB_CACHE=/ds/models/hf-cache-slt/ \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=1 \
     --mem-per-cpu=4G \
    python offline_structuredOutput.py
```
</details>

### Vision example
vLLM can also be applied to vision tasks, such as generating captions for images. 
When using vision LLMs, you have to use the specific prompt-template for the model and provide `stop_token_ids`.
Please check the official GitHub repository for the specific prompt-template and `stop_token_ids` [here](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language.py).
The example can be found in the file [offline_visionExample.py](./src/offline_visionExample.py), which loads the image in [data/example.jpg](./data/example.jpg) and generates a caption for it.
<details>
    <summary>Example</summary>

```bash
srun --partition=RTXA6000-SLT \
     --job-name=vllm-test \
     --export=ALL,HF_HUB_CACHE=/ds/models/hf-cache-slt/ \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=1 \
     --mem-per-cpu=4G \
    python offline_visionExample.py
```
</details>


### Vision example (slightly more complex)
As we have seen in the previous example, vLLM requires for vision tasks the correct prompt-template and the stop_token_ids.
In the [original examples](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language.py) from the vLLM repository, 
the code loads the LLM for each question. 
I have modified the code to load the LLM only once and then use it for all questions.
<details>
    <summary>Example</summary>

```bash
srun --partition=RTXA6000-SLT \
     --job-name=vllm-test \
     --export=ALL,HF_HUB_CACHE=/ds/models/hf-cache-slt/ \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=1 \
     --mem-per-cpu=4G \
    python offline_visionImproved.py --model=LLAVANext
```
</details>    

### Quantization example
LLMs are known for their substantial size, which makes them highly memory-intensive. 
Quantization is a technique designed to reduce a model's memory footprint. 
Quantized models can be loaded just like their standard counterparts in vLLM. 
In cases where a quantized version of your model isn’t readily available, you can perform the quantization yourself. 
Beyond the general concept, there are various methods and tools available for quantizing models. 
If you are interested in model quantization for vLLM, we refer to the vLLM [documentation](https://docs.vllm.ai/en/v0.6.1/quantization/bnb.html).
As I currently lack experience with quantization, I cannot provide insights into best practices.

<details>
  <summary>AWQ quantization</summary>
You can find an example for [AWQ quantization](https://arxiv.org/abs/2106.04561) in the file `quantisation_quantizeModel.py`.
AWQ quantisation uses calibration instances and you should therefore execute with GPU resources. 


```bash
pip install autoawq
```

```bash
srun --partition=RTXA6000-SLT \
     --job-name=quantisation \
     --export=ALL,HF_HUB_CACHE=/ds/models/hf-cache-slt/ \
     --nodes=1 \
     --ntasks=1 \
     --gpus-per-task=1 \
     --cpus-per-task=3 \
     --mem=50G \
     --time=1-00:00:00 \
     python quantisation_quantizeModel.py
```
</details>

### Fine-tuning example
I followed the  [unsloth tutorial](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb) to fine tune a Llama 3.2 model on the [Tome-100k dataset](https://huggingface.co/datasets/mlabonne/FineTome-100k).
The current example is a simple fine-tuning example, which can be found in  [fineTuningSFT.py](fineTuningSFT.py).
Training is currently only 60 steps and the model is saved to the `/ds/models/hf-cache-slt/myAwesomeModel` directory.

```bash
pip install unsloth
```

<details>
    <summary>Example</summary>

Fine-Tuning:
```bash
srun --partition=RTXA6000-SLT \
     --job-name=fine-tuning \
     --export=ALL,HF_HUB_CACHE=/ds/models/hf-cache-slt/ \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=1 \
     --mem-per-cpu=4G \
    python offline_visionExample.py
```

Inference using vLLM
```bash
srun  --partition=RTXA6000 \
      --job-name=vllm-test \
      --export=ALL,HF_HUB_CACHE=/ds/models/hf-cache-slt/ \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task=6 \
      --gpus-per-task=1 \
      --mem-per-cpu=4G \
      python offline_simpleInference.py --model_name=/ds/models/hf-cache-slt/myAwesomeModel/ --prompt="Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,
```

</details>



### How to Run a Multi-GGUF Model (e.g., DeepSeek)
vLLM currently supports only single-GGUF models.
To use multi-GGUF models, you need to download the individual GGUF files and merge them into a single file.
This tutorial uses DeepSeek-R1 with 671B parameters as an example.

<details>
  <summary>DeepSeek Tutorial</summary>

#### Step 1: Download the GGUF files
```bash
huggingface-cli download "unsloth/DeepSeek-R1-Q2_K" --local-dir /ds/models/hf-cache-slt/ --include='*-Q2_K-*'
```

#### Step 2: Convert to a single GGUF model
To merge GGUF files, we'll use LLAMA.cpp:

> **Note:** I was unable to run LLAMA.cpp on our cluster and had to perform the merging on a local machine. 
The resulting GGUF file was then transferred using rsync.. 
If you manage to get GGUF-merging working on the cluster, please submit a pull request.

```bash
# Clone and build LLAMA.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release

# Merge the GGUF files
./build/bin/llama-gguf-split --merge ~/deepseek/DeepSeek-V3-Q2_K_XS/DeepSeek-V3-Q2_K_XS-00001-of-00005.gguf ~/deepseek/oneModel/output.gguf
```

You can find the DeepSeek-R1 model here: `/ds/models/hf-cache-slt/deepseek/DeepSeek-R1.gguf`

#### Step 3: DeepSeek specific things
There are some DeepSeek specific changes for this project, following this pull-request https://github.com/vllm-project/vllm/pull/13167:

##### 3.1: Install vLLM nightly build
```bash
pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

##### 3.2: Download additionally needed files
No need to download. 
All these files can be found here: `/ds/models/hf-cache-slt/deepseek/deepseek-config`

From: https://huggingface.co/deepseek-ai/DeepSeek-R1/tree/main
- generation_config.json
- tokenizer_config.json
- tokenizer.json
- model.safetensors.index.json

From: https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main
- config.json
- configuration_deepseek.py 
- modeling_deepseek.py

Don't forget to change torch_dtype in config.json

##### 3.3: Disable MLA
```bash
export VLLM_MLA_DISABLE=1
```

#### Step 4: Run the model
For running DeepSeek-R1, please use this code and disable MLA. 
However, for regular models you can provide the GGUF-model to the Simple example script.
```bash
srun --partition=H100 \
     --job-name=deepseek-r1 \
     --export=ALL,HF_HUB_CACHE=/ds/models/hf-cache-slt/ \
     --nodes=1 \
     --ntasks=1 \
     --gpus-per-task=4 \
     --cpus-per-task=12 \
     --mem=64G \
     --time=1-00:00:00 \
     python deepseek.py     
```
</details>

## Online use
You can also use vLLM in online/interactive mode. 
As previously mentioned, this mode is not recommended for production use, but it is useful for testing and debugging.
Please ensure that you shutdown the service after you are done with it, as it consumes allocated resources even when not in use.
This mode starts a service on the cluster, which you can access via a REST interface.
This is similar to the tutorial from [perseus-textgen](https://github.com/DFKI-NLP/perseus-textgen), but in my personal experience less brittle.

**Steps:**
1. Start the service 
2. Retrieve the node name using `squeue -u $USER`.
3. Access the service documentation at `http://$NODE.kl.dfki.de:8000/docs`.

### 1. Start service

#### When in local mode (requires GPU)
```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

#### On DFKI-cluster
Please set download-dir accordingly.
```bash
srun --partition=RTXA6000-SLT \
     --job-name=vllm_serve \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=1 \
     --mem-per-cpu=4G \
     --time=1-00:00:00 \
     vllm serve "Qwen/Qwen2.5-1.5B-Instruct" \
                --download-dir=/ds/models/llms/cache \
                --port=8000
```

### 2. Retrieve the node name using
Call this on the head node to get the list of your running jobs:

```bash
squeue -u $USER
```
Then, you can access the API documentation at the following endpoint (replace $NODE with the node name):
http://$NODE.kl.dfki.de:8000/docs

### 3. Access service
Replace $NODE with the node name.
```bash
curl http://${NODE}.kl.dfki.de:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

#### Optional: Forward port to local machine
If you want to access the service from your local machine, you can forward the port using SSH.

```bash
ssh -L 5001:<$NODE>:8000 <username>@<loginnode>
```

Then you can access the service on your local machine at `http://localhost:5001`.
```bash
curl http://localhost:5001/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

### Access services using generation-like endpoint
Check the example in `online/remoteGeneration.py`.

### Access services using completion-like endpoint
Check the example in `online/remoteChat.py`.