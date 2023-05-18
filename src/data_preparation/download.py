import json
import os
import requests
from tqdm import tqdm
import tarfile
from zipfile import ZipFile
import argparse

from configs.globals import ROOT


def download(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


if __name__ == "__main__":
    with open("configs/datasets.json") as f:
        dataset_records = json.load(f)

    parser = argparse.ArgumentParser(
        description="Download dataset by URL.",
    )
    parser.add_argument(
        "--dataset-name",
        default="mvtec_ad",
        choices=list(dataset_records),
        type=str,
        required=True,
    )
    args = parser.parse_args()

    project_name = args.dataset_name

    ds_folder = os.path.join(ROOT, project_name)
    tar_path = os.path.join(ds_folder, dataset_records[project_name]["tar_name"])
    os.makedirs(ds_folder, exist_ok=True)

    print("Downloading dataset...")
    if not os.path.exists(tar_path):
        download(dataset_records[project_name]["url"], tar_path)

    print("Extracting dataset...")
    opener_fn = tarfile.open if ".tar" in tar_path else ZipFile
    with opener_fn(tar_path) as file:
        file.extractall(ds_folder)

    os.remove(tar_path)

    print("Dataset downloaded and extracted successfully!")
