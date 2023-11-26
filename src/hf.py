import json

from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast


if __name__ == "__main__":
    model = DistilBertForTokenClassification.from_pretrained(
        "Davlan/distilbert-base-multilingual-cased-ner-hrl"
    )
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "Davlan/distilbert-base-multilingual-cased-ner-hrl",
    )
    sentences = [
        "My name is Clara and I live in Berkeley, California.",
        "I saw Barack Obama at the White House today.",
        "Ich habe gestern die Goethe Universität in Frankfurt am Main besucht.",
        "Nader Jokhadar had given Syria the lead with a well-struck header in the seventh minute.",
    ]
    tokens = tokenizer(
        sentences,
        truncation=True,
        max_length=512,
        padding=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offset_mapping = tokens.pop("offset_mapping")
    outputs = model(**tokens)

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

    tokens.fromkeys
    annot = []
    for idx, (ts, os, ps) in enumerate(
        zip(tokens.input_ids, offset_mapping, outputs.logits.argmax(2))
    ):
        word_ids = tokens.word_ids(idx)
        tt = tokenizer.batch_decode([[t] for t in ts])
        annot_s = []
        for idx, (i, o, p) in enumerate(zip(ts, os.tolist(), ps)):
            if i > 103 and p != 0:
                if word_ids[idx] == word_ids[idx - 1]:
                    annot_s[-1]["end"] = o[1]
                else:
                    annot_s.append(
                        {
                            "label": label_map[p],
                            "begin": o[0],
                            "end": o[1],
                        }
                    )
        annot.append(annot_s)

    print(json.dumps(annot, indent=2))
