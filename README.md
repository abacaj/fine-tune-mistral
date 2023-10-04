# fine-tune-mistral

Code used to fine-tune this model: [abacaj/mistral-7b-sft](https://huggingface.co/abacaj/mistral-7b-sft). Add your data in the data folder as `train.jsonl` and `validation.jsonl`.

# How to run

Install dependencies:
```
python -m venv env \
  && source env/bin/activate \
  && pip install -r requirements.txt
```

Run training code:
```
torchrun --nnodes=1 --nproc-per-node=<REPLACE_WITH_NUMBER_OF_GPUS> train.py
```