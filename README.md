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

Please see the the methhod load_mode in the other/utils folder. 
This method helps you in loading huggingface models and stroring them to your specified directory.

### Simple example

### Chatstyle example

### Structured output example

### Quantization example

### Multi-modal example

## Online use

### Start service

### Access services using generation-like endpoint

### Access services using completion-like endpoint