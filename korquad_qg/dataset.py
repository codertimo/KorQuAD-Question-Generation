import json
from typing import List, NamedTuple, Optional, Tuple

import torch
from tokenizers import SentencePieceBPETokenizer
from torch.utils.data import Dataset

GPTDecodingInputType = Tuple[torch.Tensor, torch.Tensor]
GPTInputsType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
GPTFeaturesType = Tuple[List[int], List[float], List[int]]

MAX_QUESTION_SPACE = 32
MIN_QUESTION_SPACE = 5


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
        self.question_prefix_tokens = self.tokenizer.encode("질문:").ids

        self.is_train = is_train

    def __getitem__(self, index: int) -> GPTFeaturesType:
        example = self.examples[index]

        context_tokens = self.tokenizer.encode(f"문맥:{example.context}").ids
        answer_tokens = self.tokenizer.encode(f"정답:{example.answer}").ids
        question_tokens = self.tokenizer.encode(f"{example.question}").ids

        # [SOS] + 문맥:CONTEXT + 정답:ANSWER + 질문:
        conditional_tokens_len = 1 + len(context_tokens) + len(answer_tokens) + len(self.question_prefix_tokens)
        # QUESTION + [EOS]
        post_tokens_len = len(question_tokens) + 1

        if conditional_tokens_len + post_tokens_len > self.max_sequence_length:
            available_seq_len = (
                self.max_sequence_length - conditional_tokens_len - post_tokens_len + len(context_tokens)
            )
            context_tokens = context_tokens[:available_seq_len]

        conditional_tokens = [self.sos_token] + context_tokens + answer_tokens + self.question_prefix_tokens
        post_tokens = question_tokens + [self.eos_token]
        input_ids = conditional_tokens + post_tokens

        labels = input_ids if self.is_train else ([-100] * len(conditional_tokens)) + post_tokens
        attention_mask = [1.0] * len(input_ids)

        assert len(input_ids) <= self.max_sequence_length

        return input_ids, attention_mask, labels

    def __len__(self) -> int:
        return len(self.examples)


def dynamic_padding_collate_fn(features: List[GPTFeaturesType]) -> GPTInputsType:
    max_seq_len = max([len(feature[0]) for feature in features])
    input_ids, attention_mask, labels = [], [], []

    for feature in features:
        padded_input_ids = feature[0] + [0] * (max_seq_len - len(feature[0]))
        padded_attention_mask = feature[1] + [0.0] * (max_seq_len - len(feature[1]))
        padded_labels = feature[2] + [-100] * (max_seq_len - len(feature[2]))

        input_ids.append(padded_input_ids)
        attention_mask.append(padded_attention_mask)
        labels.append(padded_labels)

    return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(labels)


class QGDecodingDataset(QGDataset):
    def __getitem__(self, index: int) -> GPTDecodingInputType:
        example = self.examples[index]

        context_tokens = self.tokenizer.encode(f"문맥:{example.context}").ids
        answer_tokens = self.tokenizer.encode(f"정답:{example.answer}").ids

        # [SOS] + CONTEXT + ANSWER + 정답:
        conditional_tokens_len = 1 + len(context_tokens) + len(answer_tokens) + len(self.question_prefix_tokens)
        # ANSWER_SEQ + [EOS]
        post_tokens_len = MAX_QUESTION_SPACE + 1
        if conditional_tokens_len + post_tokens_len > self.max_sequence_length:
            available_seq_len = (
                self.max_sequence_length - conditional_tokens_len - post_tokens_len + len(context_tokens)
            )
            context_tokens = context_tokens[:available_seq_len]

        input_ids = [self.sos_token] + context_tokens + answer_tokens + self.question_prefix_tokens
        attention_mask = [1.0] * len(input_ids)

        return torch.tensor(input_ids), torch.tensor(attention_mask)


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
