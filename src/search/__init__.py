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
            # search_alg=algo, # TODO
        ),
        param_space=dict(
            vocab_size=5000,
        ),
    )
    results = tuner.fit()
    print(results.get_best_result())


if __name__ == "__main__":
    run_search()
