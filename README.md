Our subsmission for the BabyLM challenge. BabyLM is a shared task for the pre-training of a LLM with a limited dataset.

model.py contains the training pipeline. The same pipeline in notebook version is contained uin Baseline.ipynb. The training data is in data/raw, the tokenizer (which is produced by during as part of the training pipeline) is saved in /artifacts . The model, after training is saved in /BabyLM.
