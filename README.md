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
To be added.
```bash
pip install unsloth
```


### Classification example
To be added.

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