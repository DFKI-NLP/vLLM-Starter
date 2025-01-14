# vLLM-Starter
vLLM-Startercode

You can use vLLM for offline and online use.
Both have their advantages and disadvantages.
The focus is on offline use, as online starts a service which might block resources on the cluster.
Either way, you should store the LLM models (/netscratch/{$USER}), as it is a large file and might fill your personal storage very quickly.
Please have a look at the offline/online use to see how this was solved.

## Installation
```bash
conda create -n vLLM-Starter python=3.11 -y
conda activate vLLM-Starter
pip install vllm
```

## Run script on cluster
Please replace **{script.py}** with your script.
```bash
srun --partition=RTXA6000-SLT \
     --job-name=vllm-test \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=1 \
     --mem-per-cpu=4G \
     --time=1-00:00:00 \
    python {script.py}
```

## Offline use

Please see the method `load_model` in the `other/utils` folder. 
This method helps you in loading Hugging Face models and storing them to your specified directory.

### Simple example
A very simple example of how to use vLLM in offline mode can be found in the file `offline_simpleInference.py`.
This script loads a model and generates text based on a prompt.
<details>
    <summary>Example</summary>

```bash
srun --partition=RTXA6000-SLT \
     --job-name=vllm-test \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=1 \
     --mem-per-cpu=4G \
    python offline_simpleInference.py
```
</details>


### Chat-style example
A more complex example can be found in the file `offline_chatstyle.py`.
This script loads a model and generates text based on a **chat-style** prompt.
<details>
    <summary>Example</summary>

```bash
srun --partition=RTXA6000-SLT \
     --job-name=vllm-test \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=1 \
     --mem-per-cpu=4G \
    python offline_chatstyle.py
```
</details>

### Structured output example
For some tasks, you might want to have a structured output.
Examples are tasks like named entity recognition, where you want to define the output structure.
The vLLM class `GuidedDecodingParams` allows a variety of options to guide the decoding process.
For example, you can use regular expressions, JSON objects, grammar, or simple choices (e.g., True/False) to guide the decoding process.
The example can be found in the file `offline_structuredOutput.py`.
<details>
    <summary>Example</summary>

```bash
srun --partition=RTXA6000-SLT \
     --job-name=vllm-test \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=1 \
     --mem-per-cpu=4G \
    python offline_structuredOutput.py
```
</details>

### Vision example
You can also use vLLM for vision tasks.
When using vision LLMs, you have to use the specific prompt-template for the model and provide `stop_token_ids`.
Please check the official GitHub repository for the specific prompt-template and `stop_token_ids` [here](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language.py).
The example can be found in the file `offline_visionExample.py`, which loads the image in `data/example.jpg` and generates a caption for it.
<details>
    <summary>Example</summary>

```bash
srun --partition=RTXA6000-SLT \
     --job-name=vllm-test \
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
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=1 \
     --mem-per-cpu=4G \
    python offline_visionImproved.py
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

### Classification example
To be added.

## Online use
You can also use vLLM in online mode.
This starts a service on the cluster and you can access it via REST interface.

### Start service

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
                --download-dir=/netscratch/thomas/vllm \
                --port=8000
```

First, you need to know the node your job is running on. Call this on the head node to get the list of your running jobs:
```bash
squeue -u $USER
```
Then, you can access the API documentation at the following endpoint (replace $NODE with the node name):
http://$NODE.kl.dfki.de:5000/docs

#### Optional: Forward port to local machine
To be added.

### Access services using curl
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

### Access services using generation-like endpoint
Check the example in `online/remoteGeneration.py`.

### Access services using completion-like endpoint
Check the example in `online/remoteChat.py`. 