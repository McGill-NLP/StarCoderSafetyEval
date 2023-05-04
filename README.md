# StarCoder Safety Evaluation 
This repository contains the code for the toxicity evaluations in the StarCoder paper.
The code for the social bias evaluations is available [here](https://github.com/McGill-NLP/bias-bench).
## Install
```bash
git clone git@github.com:McGill-NLP/StarCoderSafetyEval.git
cd StarCoderSafetyEval
virtualenv env && source env/bin/activate
pip install -e .
```
## Evaluating Response Toxicity
To evaluate using [RealToxicityPrompts](https://aclanthology.org/2020.findings-emnlp.301/), you'll first need to download the dataset from [here](https://allenai.org/data/real-toxicity-prompts).
Once you've downloaded the dataset, use the following commands to prepare the data:
```bash

# Uncompress dataset.
tar -xvf realtoxicityprompts-data.tar.gz

# Copy to working directory.
cp realtoxicityprompts-data/prompts.jsonl .
```
You can then use the following command to launch toxicity evaluation:
```bash
python3 real_toxicity_prompts_evaluation.py \
    --model_name_or_path ${model_name_or_path} \
    --batch_size 8 \
    --data_file_path ${data_file_path} \
    --num_example 10000 \
    --output_dir ${output_dir}
```
This script does two things: (1) Response generation and (2) Evaluation of toxicity in generated responses.
The generated scores will be written to `output_dir`.
Importantly, there is also a `num_example` argument for this script.
This limits the number of examples evaluated (RealToxicityPrompts contains ~100K prompts).
We currently use two tools for evaluating generated responses:
* An [offensive word list from ParlAI](https://github.com/facebookresearch/ParlAI/blob/main/parlai/utils/safety.py). This checks the responses for toxic/offensive tokens.
* A [RoBERTa toxicity detector](https://huggingface.co/spaces/ybelkada/toxicity). This uses a trained LM-based classifier to evaluate toxicity in generated responses.
