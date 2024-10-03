# modules/dataset_loader.py

from beir import util
from beir.datasets.data_loader import GenericDataLoader

def load_dataset(dataset_name):
    data_path = util.download_and_unzip(f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip", "./datasets")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    return corpus, queries, qrels
