import enum
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from itertools import islice
from typing import Final, List


def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


class Entity(enum.StrEnum):
    MISC = "MISC"
    PER = "PER"
    ORG = "ORG"
    LOC = "LOC"


class Label:
    value: str
    is_out = False
    is_begin = False
    is_inside = False

    def short(self) -> str:
        return self.value

    def long(self) -> str:
        return f"{self.value}"


class O(Label):
    value = "O"
    is_out = True


class IO(Label):
    def short(self) -> str:
        return f"{self.entity.value}"

    def long(self) -> str:
        return f"{self.value}-{self.entity.value}"


@dataclass
class B(IO):
    entity: Entity
    value = "B"
    is_begin = True


@dataclass
class I(IO):
    entity: Entity
    value = "I"
    is_inside = True


@dataclass
class Annotation:
    label: Label
    begin: int
    end: int

    def to_json(self, strip_io=True):
        return {
            "label": self.label.short() if strip_io else self.label.long(),
            "begin": self.begin,
            "end": self.end,
        }


LABEL_MAP: Final[List[Label]] = [
    O(),
    B(Entity.MISC),
    I(Entity.MISC),
    B(Entity.PER),
    I(Entity.PER),
    B(Entity.ORG),
    I(Entity.ORG),
    B(Entity.LOC),
    I(Entity.LOC),
]


def get_arg_parser(aggregation_choices=("last", "strict", "none")):
    parser = ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("--cpu", dest="device", action="store_const", const="cpu")
    parser.add_argument("--cuda", dest="device", action="store_const", const="cuda")
    parser.add_argument("--gpu", dest="device", action="store_const", const="cuda")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument(
        "-nb",
        "--nb",
        "--no_batch",
        dest="batch_size",
        action="store_const",
        const=1,
    )
    parser.add_argument(
        "-a",
        "--aggregation",
        type=str,
        default=aggregation_choices[0],
        choices=aggregation_choices,
    )
    parser.add_argument("corpus", type=str)
    return parser


def parse_args_device(args: Namespace) -> str:
    match args.device:
        case "cuda":
            device = f"cuda:{args.device_id}"
        case device:
            device = device
    return device


def process_outputs(
    word_ids: list[int | None],
    token_ids: list[int],
    offsets: list[tuple[int, int]],
    predictions: list[int],
    aggregation: str = "last",
):
    annotations: list[Annotation] = []
    last_annotation = Annotation(O(), 0, 0)
    last_word_id = None
    for idx, (token, offset, pred) in enumerate(
        zip(token_ids, offsets.tolist(), predictions)
    ):
        if token > 103:
            word_id = word_ids[idx]
            label = LABEL_MAP[pred]
            begin, end = offset
            match [aggregation, label, last_annotation.label]:
                case _ if word_id == last_word_id:
                    last_annotation.end = end
                case ["none", _, _]:
                    last_annotation = Annotation(label, begin, end)
                    annotations.append(last_annotation)
                case [
                    "strict",
                    I(entity),
                    B(last_entity) | I(last_entity),
                ] if entity == last_entity:
                    last_annotation.end = end
                case [_, I(_), B(_) | I(_)]:
                    last_annotation.end = end
                case _:
                    last_annotation = Annotation(label, begin, end)
                    annotations.append(last_annotation)
            last_word_id = word_id
    return [
        a.to_json(strip_io=aggregation != "none")
        for a in annotations
        if not a.label.is_out
    ]
