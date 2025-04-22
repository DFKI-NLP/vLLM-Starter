# "Manual" on how to use RolmOCR on the cluster

## 1. Convert PDFs to an imgage

In order to convert PDFs to an image, use the pdf_to_image.py script. Angenommen, the PDFs are within a folder called "testdaten", the script iterates through it and generated an image for each page of the PDF-file. All images are saved within the same folder. In the end it also generates a .json file with the names of the pages which will be needed for the further conduction of OCR.

## 2. Start vLLM server

In oder to use rolmOCR one has to start a server as described on their github. For this, you can use launch_vllm.sh on the cluster. A running server will be generated afterwards. The following partitions are included "A100-PCI,H200-DA,RTX3090-MLT", but this can be changed/adapted as wished. 

## 3. Prepare script for OCR

query_rolmocr.py is a prepared script for the OCR process. Theoretically, it can be used as it is, but in case you have another forlder stucture, it should be adapted. 

## 4. Run Query

Before running the query, you need to adapt the file `run_query.sh`:

```
#!/bin/bash
#SBATCH --job-name=rolm-query
#SBATCH --partition=<PARTITION>
#SBATCH --nodelist=<NODE>
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:10:00
#SBATCH --output=rolm_query.out
#SBATCH --error=rolm_query.err

# DIRECTLY use the python binary from your conda env
/netscratch/<USERNAME>/miniconda3/envs/<VIRTUAL_ENVIRONMENT_NAME>/bin/python <SCRIPT_NAME>
```

- <PARTITION>: Insert the name of the partition, the server is running on.
- <NODE>: With the following command you can find out on which node the server is running on: `squeue -u $USER`. The output should include "NODELIST". Insert that node name for <NODE>
- <USERNAME>, <VIRTUAL_ENVIRONMENT_NAME>, and <SCRIPT_NAME> are self-explained.


## CURRENT ISSUES
The script runs but does not return an output-file as it should. Needs to be debugged still.
