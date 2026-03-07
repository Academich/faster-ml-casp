"""
Module with script to download public data
This module is modified for the needs of this project.
It differs from the corresponding file in the original AiZynthFinder implementation
"""
import argparse
import os
import sys

import requests
import tqdm

FILES_TO_DOWNLOAD = {
    "policy_model_onnx": {
        "filename": "uspto_model.onnx",
        "url": "https://zenodo.org/record/7797465/files/uspto_model.onnx",
    },
    "template_file": {
        "filename": "uspto_templates.csv.gz",
        "url": "https://zenodo.org/record/7341155/files/uspto_unique_templates.csv.gz",
    },
    "ringbreaker_model_onnx": {
        "filename": "uspto_ringbreaker_model.onnx",
        "url": "https://zenodo.org/record/7797465/files/uspto_ringbreaker_model.onnx",
    },
    "ringbreaker_templates": {
        "filename": "uspto_ringbreaker_templates.csv.gz",
        "url": "https://zenodo.org/record/7341155/files/uspto_ringbreaker_unique_templates.csv.gz",
    },
    "stock_zinc": {
        "filename": "zinc_stock.hdf5",
        "url": "https://ndownloader.figshare.com/files/23086469",
    },
    "caspirus10k":  {
        "filename": "caspyrus10k.csv",
        "url": "https://ndownloader.figshare.com/files/43491753?private_link=2eab4132b322229c1efc",
    },
    "stock_paroutes": {
        "filename": "paroutes_n1_stock.hdf5",
        "url": "https://ndownloader.figshare.com/files/43491756?private_link=2eab4132b322229c1efc",
    },
    "filter_policy_onnx": {
        "filename": "uspto_filter_model.onnx",
        "url": "https://zenodo.org/record/7797465/files/uspto_filter_model.onnx",
    },
    "paroutes_n1_routes": {
        "filename": "n1-routes.json",
        "url": "https://zenodo.org/record/6275421/files/n1-routes.json?download=1",
    },
    "paroutes_n1_targets": {
        "filename": "n1-targets.txt",
        "url": "https://zenodo.org/record/6275421/files/n1-targets.txt?download=1",
    },
}

BB_DIRECTORY = "bb_stock"

YAML_TEMPLATE = """expansion:
  uspto:
    - {}
    - {}
  ringbreaker:
    - {}
    - {}
filter:
  uspto: {}
stock:
  zinc: {}
"""


def _download_file(url: str, filename: str) -> None:
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        pbar = tqdm.tqdm(
            total=total_size, desc=os.path.basename(filename), unit="B", unit_scale=True
        )
        with open(filename, "wb") as fileobj:
            for chunk in response.iter_content(chunk_size=1024):
                fileobj.write(chunk)
                pbar.update(len(chunk))
        pbar.close()


def main() -> None:
    """Entry-point for CLI"""
    parser = argparse.ArgumentParser("download_public_data")
    parser.add_argument(
        "path",
        default=".",
        help="the path to download the files",
    )
    path = parser.parse_args().path
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(os.path.join(path, BB_DIRECTORY)):
        os.makedirs(os.path.join(path, BB_DIRECTORY))

    try:
        for key, filespec in FILES_TO_DOWNLOAD.items():
            save_path = os.path.join(path, filespec["filename"])
            if key in ("stock_zinc", "stock_paroutes"):
                save_path = os.path.join(path, BB_DIRECTORY, filespec["filename"])
            _download_file(filespec["url"], save_path)
    except requests.HTTPError as err:
        print(f"Download failed with message {str(err)}")
        sys.exit(1)

    with open(os.path.join(path, "config.yml"), "w") as fileobj:
        path = os.path.abspath(path)
        fileobj.write(
            YAML_TEMPLATE.format(
                os.path.join(path, FILES_TO_DOWNLOAD["policy_model_onnx"]["filename"]),
                os.path.join(path, FILES_TO_DOWNLOAD["template_file"]["filename"]),
                os.path.join(
                    path, FILES_TO_DOWNLOAD["ringbreaker_model_onnx"]["filename"]
                ),
                os.path.join(
                    path, FILES_TO_DOWNLOAD["ringbreaker_templates"]["filename"]
                ),
                os.path.join(path, FILES_TO_DOWNLOAD["filter_policy_onnx"]["filename"]),
                os.path.join(path, BB_DIRECTORY, FILES_TO_DOWNLOAD["stock_zinc"]["filename"]),
            )
        )
    print("Configuration file written to config.yml")


if __name__ == "__main__":
    main()
