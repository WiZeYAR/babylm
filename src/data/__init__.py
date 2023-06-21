import requests
import zipfile
from tqdm.auto import tqdm
from pathlib import Path
import lightning as ptl
from pydantic import BaseSettings
from tokenizers import ByteLevelBPETokenizer


class DataConfig(BaseSettings):
    data_folder: Path = Path("data")
    artifact_folder: Path = Path("artifacts")
    download_link: str = (
        "https://github.com/babylm/babylm.github.io/raw/main/babylm_data.zip"
    )


class DataModule(ptl.LightningDataModule):
    def __init__(self, conf: DataConfig = DataConfig()) -> None:
        super().__init__()
        self.conf = conf

    def prepare_data(self) -> None:
        self.__download()
        self.__train_tokenizer()

    def setup(self, stage: str) -> None:
        return super().setup(stage)

    def __train_tokenizer(self):
        """Generates tokenizer from data"""
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=list(map(str, Path(self.conf.data_folder).glob(f"**/*.train"))),
            vocab_size=52_000,
            min_frequency=2,
            special_tokens=[
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "<mask>",
            ],
        )
        tokenizer.save_model(str(self.conf.artifact_folder), "tokenizer")

    def __download(self):
        """Downloads raw data and unzips it"""
        raw_data_folder = Path(self.conf.data_folder) / "raw"
        output_file = Path(self.conf.data_folder) / "raw.zip"

        raw_data_folder.mkdir(parents=True, exist_ok=True)
        if not (raw_data_folder / ".SUCCESS").is_file():
            # ---- Download zip file
            response = requests.get(self.conf.download_link, stream=True, timeout=60)
            file_size = int(response.headers.get("Content-Length", 0))

            with Path(output_file).open("wb") as f:
                progress_bar = tqdm(
                    total=file_size,
                    unit="B",
                    unit_scale=True,
                    desc="Downloading raw data",
                )
                for data in response.iter_content(chunk_size=4096):
                    f.write(data)
                    progress_bar.update(len(data))

            # ---- Inflate zip file
            with zipfile.ZipFile(output_file) as f:
                f.extractall(raw_data_folder)
            (raw_data_folder / ".SUCCESS").touch()


def download_raw(
    url: str = "https://github.com/babylm/babylm.github.io/raw/main/babylm_data.zip",
    data_folder: str = "data",
    lazy: bool = True,
    timeout: int = 30,
) -> dict[str, list[Path]]:
    raw_data_folder = Path(data_folder) / "raw"
    output_file = Path(data_folder) / "raw.zip"

    if lazy and not (raw_data_folder / ".SUCCESS").is_file():
        # ---- Download zip file
        response = requests.get(url, stream=True, timeout=timeout)
        file_size = int(response.headers.get("Content-Length", 0))

        with Path(output_file).open("wb") as f:
            progress_bar = tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                desc="Downloading raw data",
            )
            for data in response.iter_content(chunk_size=4096):
                f.write(data)
                progress_bar.update(len(data))

        # ---- Inflate zip file
        with zipfile.ZipFile(output_file) as f:
            f.extractall(raw_data_folder)
        (raw_data_folder / ".SUCCESS").touch()

    return {
        mode: list(raw_data_folder.glob(f"**/*.{mode}"))
        for mode in ["train", "dev", "test"]
    }
