import json
from argparse import ArgumentParser

from torch.utils.data import Dataset
from transformers import pipeline

label_map = [
    "O",
    "B-MISC",
    "I-MISC",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
]


class SentDataset(Dataset):
    def __init__(self, sentences, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        return self.sentences[i]

    def __iter__(self):
        return iter(self.sentences)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("--cpu", dest="device", action="store_const", const="cpu")
    parser.add_argument("--cuda", dest="device", action="store_const", const="cuda")
    parser.add_argument("--gpu", dest="device", action="store_const", const="cuda")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument(
        "--aggregation",
        type=str,
        default="average",
        choices=("average", "none", "max"),
    )
    parser.add_argument(
        "-nb",
        "--nb",
        "--no_batch",
        dest="batch_size",
        action="store_const",
        const=None,
    )
    parser.add_argument("corpus", type=str)

    args = parser.parse_args()

    match args.device:
        case "cuda":
            device = f"cuda:{args.device_id}"
        case device:
            device = device

    token_classifier = pipeline(
        "ner",
        model="Davlan/distilbert-base-multilingual-cased-ner-hrl",
        aggregation_strategy=args.aggregation,
        device=device,
    )

    with open(args.corpus, "r", encoding="utf-8") as fp:
        sentences = SentDataset([l.strip() for l in fp])

    annot = []
    it = (
        token_classifier(sentences, batch_size=args.batch_size)
        if args.batch_size is not None
        else (token_classifier(s) for s in sentences)
    )
    for out in it:
        s = [
            {
                "label": pred.get("entity") or pred.get("entity_group"),
                "begin": pred["start"],
                "end": pred["end"],
            }
            for pred in out
        ]
        annot.append(s)

    with open("pipelines.json", "w") as fp:
        json.dump(annot, fp, indent=2)
