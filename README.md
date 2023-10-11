# fine-tune-mistral

Code used to fine-tune this model: [abacaj/mistral-7b-sft](https://huggingface.co/abacaj/mistral-7b-sft). Add your data in the data folder as `train.jsonl` and `validation.jsonl`.

**Note** this repo is intended for full fine-tuning of mistral not qlora or other methods.

# How to run

Install dependencies:
```
python -m venv env \
  && source env/bin/activate \
  && pip install -r requirements.txt
```

[Get a Hugging Face token](https://huggingface.co/settings/tokens) and set the variable:

```
export HF_TOKEN="[insert token here]"
```

Run training code:
```
torchrun --nnodes=1 --nproc-per-node=<REPLACE_WITH_NUMBER_OF_GPUS> train.py
```

# Tips

- If running with a small batch size, lower the learning rate
- I did not have to adjust grad clip or weight_decay but YMMV
- Use enough data, I recommend > 1k samples
- I ran this for 3 epochs on 40k samples, will need to experiment more on epochs because the model was still improving.
- The better way to tell if your model is improving or just overfitting or even getting worse, you should add evaluation on your task. This is data that is not part of training. For example, on code completion you can evaluate your model on the mbpp validation set or a custom set you have.
- Use FSDP option: `backward_prefetch=BackwardPrefetch.BACKWARD_PRE` if you have the GPU memory, or `backward_prefetch=BackwardPrefetch.BACKWARD_POST`. This can cause OOM so it was set to None
