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

# Tips

- If running with a small batch size, lower the learning rate
- I did not have to adjust grad clip or weight_decay but YMMV
- Use enough data, I recommend > 1k samples
- If you have enough data you can likely go above 3 epochs, 5 or 7
