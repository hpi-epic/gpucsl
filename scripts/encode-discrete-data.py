import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

import sys

import argparse


parser = argparse.ArgumentParser(
    description="Encode a discrete dataset with integer encoding. The script needs to be run from the scripts folder."
)
parser.add_argument(
    "dataset",
    action="store",
    help="The name of the dataset to be encoded. The dataset needs to be in the ../data folder.",
)
args = parser.parse_args()

enc = OrdinalEncoder(dtype=int)

dataset_name = args.dataset
save_folder = f"../data"
save_path = f"{save_folder}/{dataset_name}/{dataset_name}_encoded.csv"
print(f"Encoding dataset {save_folder}/{dataset_name}/{dataset_name}.csv")

pd.DataFrame(
    enc.fit_transform(
        pd.read_csv(f"{save_folder}/{dataset_name}/{dataset_name}.csv", header=None)
    )
).to_csv(save_path, header=False, index=False)

print(f"Wrote {save_path}")
