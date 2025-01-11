# vLLM-Starter
vLLM-Startercode

You can use vLLM for offline and online use.
Both have their advantages and disadvantages.
The fokus is on offline use, as online starts a service which might block resources on the cluster.
Either way, you should  store the LLM models (/netscratch/), as it is a large file and might fill your personal storage very quickly.
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

Please see the the methhod load_model in the other/utils folder. 
This method helps you in loading huggingface models and stroring them to your specified directory.

### Simple example
A very simple example of how to use vLLM in offline mode can be found in the fle offline_simpleInference.py.
This script loads a model and generates text based on a prompt.

### Chatstyle example
A more complex example can be found in the file offline_chatstyle.py.
This script loads a model and generates text based on a **chatstyle** prompt.

### Structured output example
FOr some tasks, you might want to have a structured output.
Examples are tasks like named entity recognition, where you want to define the output structure.
The vllm class GuidedDecodingParams allows a variety of options to guide the decoding process.
For example, you can use regular expressions, JSON objects, grammar, or simple choices (e.g., True/False) to guide the decoding process.
The example can be found in the file offline_structuredOutput.py.

### Vision example
You can also use vLLM for vision tasks.
When using vision LLMs, you have to use the specific prompt-template for the model and provide stop_token_ids.
Please check the official github repository for the specific prompt-template and stop_token_ids here.
The example can be found in the file offline_visionExample.py, which loads the image in data/example.jpg and generates a caption for it.

### Quantization example


### Fine-tuning example

## Online use
You can also use vLLM in online mode.
This starts a service on the cluster and you can access it via REST.


### Start service

#### When in local mode (requires GPU)
```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

#### On DFKI-cluster
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

#### Optional: Forward port to local machine
#TODO 
#ssh -L 5001:serv-9217:8000 <user>@login1.pegasus.kl.dfki.de

### Access services using curl

```bash
curl http://serv-9219.kl.dfki.de:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```
### Access services using generation-like endpoint
Check example in online/remoteGeneration.py

### Access services using completion-like endpoint
Check example in online/remoteChat.py