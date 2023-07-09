#!/bin/sh
ws=/workspaces/babylm
poetry install
poetry run pip install git+https://github.com/babylm/evaluation-pipeline.git#egg=lm_eval[dev]

if nvidia-smi; then
    poetry run pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
fi

wget https://github.com/babylm/evaluation-pipeline/raw/main/filter_data.zip $ws/filter_data.zip
unzip $ws/filter_data.zip
