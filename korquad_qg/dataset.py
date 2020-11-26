from typing import List, NamedTuple, Tuple, Optional

import torch
from tokenizers import SentencePieceBPETokenizer
from torch.utils.data import Dataset

import json

GPTDecodingInputType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
GPTInputsType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
GPTFeaturesType = Tuple[List[int], List[float], List[int], List[int]]


class QAExample(NamedTuple):
    context: str
    answer: str
    question: Optional[str] = None


class QGDataset(Dataset):
    def __init__(
        self,
        examples: List[QAExample],
        tokenizer: SentencePieceBPETokenizer,
        max_sequence_length: int,
        is_train: bool = True,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

        self.sos_token = tokenizer.token_to_id("<s>")
        self.eos_token = tokenizer.token_to_id("</s>")

        self.is_train = is_train

    def __getitem__(self, index: int) -> GPTFeaturesType:
        example = self.examples[index]

        context_tokens = self.tokenizer.encode(f"문맥:{example.context}").ids
        answer_tokens = self.tokenizer.encode(f"정답:{example.answer}").ids
        question_tokens = self.tokenizer.encode(f"질문:{example.question}").ids

        full_length = len(context_tokens) + len(answer_tokens) + len(question_tokens) + 2
        if full_length > self.max_sequence_length:
            available_seq_len = self.max_sequence_length - len(answer_tokens) - len(question_tokens) - 2
            context_tokens = context_tokens[:available_seq_len]

        input_ids = [self.sos_token] + context_tokens + answer_tokens + question_tokens + [self.eos_token]
        num_conditional_tokens = len(input_ids) - len(question_tokens) - 1

        labels = input_ids if self.is_train else ([-100] * num_conditional_tokens) + question_tokens
        attention_mask = [1.0] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        return input_ids, attention_mask, token_type_ids, labels

    def __len__(self) -> int:
        return len(self.examples)


def dynamic_padding_collate_fn(features: List[GPTFeaturesType]) -> GPTInputsType:
    max_seq_len = max([len(feature[0]) for feature in features])
    input_ids, attention_mask, token_type_ids, labels = [], [], [], []

    for feature in features:
        padded_input_ids = input_ids + [0] * (max_seq_len - len(feature[0]))
        padded_attention_mask = attention_mask + [0.0] * (max_seq_len - len(feature[1]))
        padded_token_type_ids = token_type_ids + [0] * (max_seq_len - len(feature[2]))
        padded_labels = labels + [-100] * (max_seq_len - len(feature[3]))

        input_ids.append(padded_input_ids)
        attention_mask.append(padded_attention_mask)
        token_type_ids.append(padded_token_type_ids)
        labels.append(padded_labels)

    return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(token_type_ids), torch.tensor(labels)


class QGDecodingDataset(QGDataset):
    def __getitem__(self, index: int) -> GPTDecodingInputType:
        example = self.examples[index]

        context_tokens = self.tokenizer.encode(f"문맥:{example.context}").ids
        answer_tokens = self.tokenizer.encode(f"정답:{example.answer}").ids
        question_prefix_tokens = self.tokenizer.encode("질문:").ids

        input_ids = [self.sos_token] + context_tokens + answer_tokens + question_prefix_tokens
        attention_mask = [1.0] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(token_type_ids)


def load_korquad_dataset(dataset_path: str) -> List[QAExample]:
    with open(dataset_path) as f:
        korquad_raw_data_json = json.load(f)

    examples = []
    for document in korquad_raw_data_json["data"]:
        for paragraph in document["paragraphs"]:
            for qa_pair in paragraph["qas"]:
                example = QAExample(paragraph["context"], qa_pair["answers"][0]["text"], qa_pair["question"])
                examples.append(example)

    return examples
