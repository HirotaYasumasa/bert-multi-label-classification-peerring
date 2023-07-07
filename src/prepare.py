import random
import unicodedata
from pathlib import Path
import pandas as pd

from classopt import classopt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import utils as utils


@classopt(default_long=True)
class Args:
    input_dir: Path = "./data"
    output_dir: Path = "./datasets/peerring"
    seed: int = 42
    label2id: dict[str, int] = {
        "treatment": 0,
        "physical": 1,
        "psychological": 2,
        "work-financial": 3,
        "family-friends": 4,
    }


def process_title(title: str) -> str:
    title = unicodedata.normalize("NFKC", title)
    title = title.strip("　").strip()
    return title


# 記事本文の前処理
# 重複した改行の削除、文頭の全角スペースの削除、NFKC正規化を実施
def process_body(body: list[str]) -> str:
    body = [unicodedata.normalize("NFKC", line) for line in body]
    body = [line.strip("　").strip() for line in body]
    body = [line for line in body if line]
    body = "\n".join(body)
    return body


def main(args: Args):
    random.seed(args.seed)

    data = []

    for path in tqdm(list(args.input_dir.glob("*.csv"))):
        if path.name == "LICENSE.txt":
            continue
        
        df = pd.read_csv(path)

        for _, row in df.iterrows():
            data.append(
                {
                    "text": row['content'],
                    "labels": [
                        row['treatment'],
                        row['physical'],
                        row['psychological'],
                        row['work/financial'],
                        row['family/friends']
                    ]
             }
         )

    random.shuffle(data)
    utils.save_jsonl(data, args.output_dir / "all.jsonl")
    utils.save_json(args.label2id, args.output_dir / "label2id.json")

    train, test = train_test_split(data, test_size=0.2, random_state=args.seed)
    utils.save_jsonl(train, args.output_dir / "train_val.jsonl")
    utils.save_jsonl(test, args.output_dir / "test.jsonl")


if __name__ == "__main__":
    args = Args.from_args()
    main(args)
