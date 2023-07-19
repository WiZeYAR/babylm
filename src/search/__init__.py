from typing import Any
from pydantic import BaseModel
from ray import tune
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from data import download_raw
from .trial import start_trial
from .params import Hyperparams


def run_search():
    download_raw()
    tuner = tune.Tuner(
        tune.with_resources(start_trial, dict(gpu=1)),
        tune_config=tune.TuneConfig(
            metric="mean_accuracy",
            mode="max",
            chdir_to_trial_dir=False,
            num_samples=5,
        ),
        param_space={
            "vocab_size": tune.choice([4096,8192]),
            "max_position_embedding": tune.choice([256,512]),
            "num_attention_heads": tune.choice([8,12]),
            "num_hidden_layers": tune.choice([4,6])
        },
    )
    results = tuner.fit()
    print(results.get_best_result())


if __name__ == "__main__":
    run_search()
