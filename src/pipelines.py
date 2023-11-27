import json

from itertools import islice

import torch

from torch.utils.data import Dataset

from transformers import (
    DistilBertForTokenClassification,
    DistilBertTokenizerFast,
    pipeline,
)


def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


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


if __name__ == "__main__":
    # model = DistilBertForTokenClassification.from_pretrained(
    #     "Davlan/distilbert-base-multilingual-cased-ner-hrl"
    # )
    # model.eval()
    # tokenizer = DistilBertTokenizerFast.from_pretrained(
    #     "Davlan/distilbert-base-multilingual-cased-ner-hrl",
    # )
    token_classifier = pipeline(
        model="Davlan/distilbert-base-multilingual-cased-ner-hrl",
        # aggregation_strategy="none",
        aggregation_strategy="average",
        # device="cpu",
        device=0,
    )

    with open(
        "/hot_storage/Data/Leipzig/deu/deu_wikipedia_2021_10K-1k_shuf.txt",
        "r",
        encoding="utf-8",
    ) as fp:
        sentences = SentDataset([l.strip() for l in fp])

    # sentences = SentDataset(
    #     [
    #         "My name is Clara and I live in Berkeley, California.",
    #         "I saw Barack Obama at the White House today.",
    #         "Ich habe gestern die Goethe Universität in Frankfurt am Main besucht.",
    #         "Nader Jokhadar had given Syria the lead with a well-struck header in the seventh minute.",
    #     ]
    # )

    annot = []
    # for out in (token_classifier(s) for s in sentences):
    for out in token_classifier(sentences, batch_size=16):
        s = [
            {
                "label": pred.get("entity") or pred.get("entity_group"),
                "begin": pred["start"],
                "end": pred["end"],
            }
            for pred in out
        ]
        if s:
            annot.append(s)
        # tokens = tokenizer(
        #     batch,
        #     truncation=True,
        #     max_length=512,
        #     padding=True,
        #     return_offsets_mapping=True,
        #     return_tensors="pt",
        # )
        # offset_mapping = tokens.pop("offset_mapping")

        # for idx, (ts, os, ps) in enumerate(
        #     zip(tokens.input_ids, offset_mapping, outputs.logits.argmax(2))
        # ):
        #     word_ids = tokens.word_ids(idx)
        #     tt = tokenizer.batch_decode([[t] for t in ts])
        #     annot_s = []
        #     for idx, (i, o, p) in enumerate(zip(ts, os.tolist(), ps)):
        #         if i > 103:  # and p != 0:
        #             if word_ids[idx] == word_ids[idx - 1]:
        #                 annot_s[-1]["end"] = o[1]
        #             else:
        #                 annot_s.append(
        #                     {
        #                         "label": label_map[p],
        #                         "begin": o[0],
        #                         "end": o[1],
        #                     }
        #                 )
        #     annot.append(annot_s)

    print(json.dumps(annot, indent=2))
