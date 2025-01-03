"""
Parse metadata from OpenDV dataset and create split files.
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict

import pandas as pd


def _path(path: str) -> str:
    path = os.path.expanduser(os.path.expandvars(path))
    return path


def create_split_opendv(path: str) -> Dict[str, Dict[str, float]]:
    """Create split files for OpenDV dataset."""
    db = pd.read_csv(path, sep="\t")
    split_db = defaultdict(list)
    for _, row in db.iterrows():
        video_id = row["video_id"].replace("@", "-") + "." + row["container"]
        split_db[row["split"]].append(video_id)

    return split_db


parser = argparse.ArgumentParser()
parser.add_argument(
    "--file",
    type=_path,
    help="Metadata CSV file.",
)
parser.add_argument(
    "--outdir",
    type=_path,
    help="Output directory for OpenDV split files.",
)
args = parser.parse_args()
split_db = create_split_opendv(args.file)

for split, ids in split_db.items():
    outfile = os.path.join(args.outdir, f"{split.lower()}.json")
    with open(outfile, "w") as f:
        json.dump(ids, f)
