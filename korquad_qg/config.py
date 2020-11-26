from typing import NamedTuple


class QGConfig(NamedTuple):
    train_dataset: str = "data/train.json"
    dev_dataset: str = "data/dev.json"
    test_dataset: str = "data/test.json"

    gpt_model_hub_name: str = "taeminlee/kogpt2"
    vocab_path: str = "tokenizer/vocab.json"
    tokenizer_merges_path: str = "tokenizer/merges.txt"

    max_sequence_length: int = 512

    epochs: int = 3
    lr: float = 5e-5
    train_batch_size: int = 45
    dev_batch_size: int = 50

    output_dir: str = "outputs/"

    grad_clip: float = 1.0
    warmup_ratio: float = 0.1

    train_log_interval: int = 50
    validation_interval: int = 1000
    save_interval: int = 1000
    random_seed: int = 0
