import json

import onnxruntime as ort
from inference import batched, get_arg_parser, process_outputs
from tqdm import tqdm
from transformers import BatchEncoding, DistilBertConfig, DistilBertTokenizerFast

if __name__ == "__main__":
    args = get_arg_parser().parse_args()

    match args.device:
        case "cuda":
            providers = ["CUDAExecutionProvider"]
        case "cpu":
            providers = ["CPUExecutionProvider"]
        case _:
            raise ValueError(f"Unknown device {args.device}")

    config = DistilBertConfig.from_json_file("data/config.json")
    tokenizer = DistilBertTokenizerFast.from_pretrained("data/")
    ort_sess = ort.InferenceSession(
        "data/model.onnx",
        providers=providers,
    )

    with open(args.corpus, "r", encoding="utf-8") as fp:
        sentences = [l.strip() for l in fp]

    annot = []
    for batch in batched(tqdm(sentences), args.batch_size):
        tokens: BatchEncoding = tokenizer(
            batch,
            truncation=True,
            max_length=config.max_position_embeddings,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="np",
        )
        offset_mapping = tokens.pop("offset_mapping")

        outputs, *_ = ort_sess.run(
            None,
            {"input_ids": tokens.input_ids, "attention_mask": tokens.attention_mask},
        )

        for idx, (ts, os, ps) in enumerate(
            zip(tokens.input_ids, offset_mapping, outputs.argmax(2))
        ):
            annot.append(
                process_outputs(tokens.word_ids(idx), ts, os, ps, args.aggregation)
            )

    with open("ort-hf.json", "w") as fp:
        json.dump(annot, fp, indent=2)
