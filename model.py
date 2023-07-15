# %%
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

# %% [markdown]
# # Create tokenizer

# %%
paths = sorted(str(x) for x in Path("./data/raw").glob("**/*.train"))
paths

# %%
vocab_size=5000

tokenizer = ByteLevelBPETokenizer() # Byte-level byte-pair tokenizer.
tokenizer.train(
    files=paths,
    vocab_size=5000,
    min_frequency=3,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ],
)

# %%
# tokenizer.save_model("./artifacts", "babylm")
tokenizer.save_model("./artifacts")

# %% [markdown]
# # Load tokenizer

# %%
tokenizer = ByteLevelBPETokenizer(
    "./artifacts/vocab.json",
    "./artifacts/merges.txt",
)
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)

# %%
from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("./artifacts")

# %%
from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=paths[8],
    block_size=128,
)

# %%
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# %%
from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=vocab_size,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

# %%
from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)

# %%
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./BabyLM",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# %%
# trainer.train()

# %% [markdown]
# ## TODOS:
# - [x] Baselines
# - [ ] Integrate evaluation pipeline with baselines
# - [ ] Find way to put any torch model into the eval pipeline
# - [ ] Gather all possible hyperparameters from the pipeline
# - [ ] Integrate with Ray Tune


