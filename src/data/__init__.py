import requests
import zipfile
from tqdm.auto import tqdm
from pathlib import Path


def download_raw(
    url: str = "https://github.com/babylm/babylm.github.io/raw/main/babylm_data.zip",
    data_folder: str = "data",
    lazy: bool = True,
    timeout: int = 30,
) -> dict[str, list[Path]]:
    raw_data_folder = Path(data_folder) / "raw"
    output_file = Path(data_folder) / "raw.zip"

    if lazy and not (raw_data_folder / ".SUCCESS").is_file():
        raw_data_folder.mkdir(parents=True, exist_ok=True)

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
