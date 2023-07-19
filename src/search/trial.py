from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from typing import Generator, Any

from transformers import RobertaTokenizerFast
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import Trainer, TrainingArguments, AutoTokenizer
from transformers import PreTrainedTokenizerFast

from torch.utils.data import ConcatDataset, Dataset
import torch.nn as nn
from ray.air import session

from .params import Hyperparams
from eval import run_eval, EvalArgs


def data_paths(raw_data_path: Path = Path("data/raw")) -> Generator[str, None, None]:
    """Gets the data paths for the

    Args:
        raw_data_path (Path, optional): Points to the raw data folder. Defaults to Path("data/raw").

    Raises:
        AssertionError: If the data has not been previously downloaded.

    Yields:
        Generator[str, None, None]: A generator over the train dataset file paths.
    """
    if not Path("data/raw/.SUCCESS").exists():
        raise AssertionError("The dataset has not yet been downloaded")
    yield from (map(str, (raw_data_path / "babylm_data/babylm_10M/").glob("*.train")))


def tokenizer(config: Hyperparams) -> AutoTokenizer:
    """Generates or loads tokenizer for the current trial

    Args:
        config (Hyperparams): Hyperparameter set of the trial

    Returns:
        AutoTokenizer: The tokenizer, used for this trial
    """

    # ---- Getting existing tokenizer
    if (Path(session.get_trial_dir()) / "vocab.json").is_file():
        return RobertaTokenizerFast.from_pretrained(session.get_trial_dir())

    # ---- Generating new tokenizer
    t = ByteLevelBPETokenizer()
    t.train(
        files=list(data_paths()),
        vocab_size=config.vocab_size,
        min_frequency=3,  # TODO: Hyperparameter?
        special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ],
    )
    t.save_model(session.get_trial_dir())
    return tokenizer(config)


def dataset(config: Hyperparams) -> Dataset:
    """Loads dataset

    Args:
        config (Hyperparams): A set of hyperparameters for the trial

    Returns:
        Dataset: Concatenated dataset with all .train files
    """
    return ConcatDataset(
        LineByLineTextDataset(
            tokenizer(config),
            file,
            block_size=128,  # TODO: Hyperparameter?
        )
        for file in data_paths()
    )


def model(config: Hyperparams) -> nn.Module:
    """Creates model

    Args:
        config (Hyperparams): A set of hyperparameters for the trial

    Returns:
        nn.Module: Torch model
    """
    return RobertaForMaskedLM(
        config=RobertaConfig(
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embedding, 
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.num_hidden_layers,
            type_vocab_size=1,  # TODO: Hyperparameter?
        )
    )


def start_trial(trial_config_dict: dict[Any, Any]):
    config = Hyperparams(**trial_config_dict)

    # ---- Training
    trainer = Trainer(
        model=model(config),
        args=TrainingArguments(
            output_dir=session.get_trial_dir(),
            overwrite_output_dir=True,
            num_train_epochs=5,  # TODO: Hyperparameter?
            per_gpu_train_batch_size=64,  # TODO: Hyperparameter?
            save_steps=3000,
            save_total_limit=2,
            prediction_loss_only=True,
        ),
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer(config),
            mlm=True,
            mlm_probability=0.15,  # TODO: Hyperparameter?
        ),
        train_dataset=dataset(config),
    )
    trainer.train()
    trainer.save_model(session.get_trial_dir())

    # ---- Evaluation
    accuracies = run_eval(EvalArgs(model_path=session.get_trial_dir()))
    return dict(mean_accuracy=sum(acc for acc in accuracies.values()) / len(accuracies))
