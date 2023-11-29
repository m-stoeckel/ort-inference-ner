import json

import torch
from inference import batched, get_arg_parser, parse_args_device, process_outputs
from transformers import (
    BatchEncoding,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizerFast,
)

if __name__ == "__main__":
    args = get_arg_parser().parse_args()

    device = parse_args_device(args)

    config = DistilBertConfig.from_pretrained(
        "Davlan/distilbert-base-multilingual-cased-ner-hrl"
    )
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "Davlan/distilbert-base-multilingual-cased-ner-hrl",
    )
    model = DistilBertForTokenClassification.from_pretrained(
        "Davlan/distilbert-base-multilingual-cased-ner-hrl"
    ).to(device)
    model.eval()

    with open(args.corpus, "r", encoding="utf-8") as fp:
        sentences = [l.strip() for l in fp]

    annot = []
    for batch in batched(sentences, args.batch_size):
        tokens: BatchEncoding = tokenizer(
            batch,
            truncation=True,
            max_length=config.max_position_embeddings,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offset_mapping = tokens.pop("offset_mapping")

        with torch.no_grad():
            outputs = model(
                input_ids=tokens.input_ids.to(device),
                attention_mask=tokens.attention_mask.to(device),
            )
            outputs = outputs.logits.cpu()

        for idx, (ts, os, ps) in enumerate(
            zip(tokens.input_ids, offset_mapping, outputs.argmax(2))
        ):
            annot.append(
                process_outputs(tokens.word_ids(idx), ts, os, ps, args.aggregation)
            )

    with open("baseline.json", "w") as fp:
        json.dump(annot, fp, indent=2)
