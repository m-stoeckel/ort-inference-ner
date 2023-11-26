from transformers import DistilBertForTokenClassification, DistilBertTokenizer


if __name__ == "__main__":
    model = DistilBertForTokenClassification.from_pretrained(
        "Davlan/distilbert-base-multilingual-cased-ner-hrl"
    )
    tokenizer = DistilBertTokenizer.from_pretrained(
        "Davlan/distilbert-base-multilingual-cased-ner-hrl",
        use_fast=True,
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
        return_tensors="pt",
    )
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

    annot = []
    for ts, ps in zip(tokens.input_ids, outputs.logits.argmax(2)):
        s_annot = []
        tt = tokenizer.batch_decode([[t] for t in ts])
        for i, t, p in zip(ts, tt, ps):
            match i:
                case _ if i < 104:
                    continue
                case _:
                    s_annot.append((t, label_map[p]))
        annot.append(s_annot)

    for s_annot in annot:
        print(s_annot)
