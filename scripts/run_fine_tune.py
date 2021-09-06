import logging
import math
import os
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
from tokenizers import SentencePieceBPETokenizer
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup

from korquad_qg.config import QGConfig
from korquad_qg.dataset import QGDataset, dynamic_padding_collate_fn, load_korquad_dataset
from korquad_qg.utils import TqdmLoggingHandler

parser = ArgumentParser()
parser.add_argument("--train-dataset", type=str, help="학습 데이터 경로")
parser.add_argument("--dev-dataset", type=str, help="평가 데이터 경로")

parser.add_argument("--epochs", type=int, help="학습 전체를 반복할 횟수")
parser.add_argument("--lr", type=float, help="learning rate")

parser.add_argument("--train-batch-size", type=int, help="학습에 사용할 배치 크기")
parser.add_argument("--eval-batch-size", type=int, help="평가에 사용할 배치 크기")
parser.add_argument("--validation-interval", type=int, help="dev 셋에 대해서 validation 을 수행할 steps")
parser.add_argument("--save-interval", type=int, help="모델을 저장할 steps")

parser.add_argument("--output-dir", type=str, default="artifacts/", help="모델과 학습 로그를 저장할 경로")


def main(config: QGConfig):
    logger = _create_logger(output_dir=config.output_dir)
    logger.info("============================")
    for key, value in config._asdict().items():
        logger.info(f"{key:30}:{value}")
    logger.info("============================")
    torch.manual_seed(config.random_seed)

    tokenizer = SentencePieceBPETokenizer.from_file(
        vocab_filename=config.vocab_path, merges_filename=config.tokenizer_merges_path, add_prefix_space=False
    )

    logger.info("loading train dataset")
    train_examples = load_korquad_dataset(config.train_dataset)
    train_dataset = QGDataset(train_examples, tokenizer, config.max_sequence_length)
    train_dataloader = DataLoader(
        train_dataset, config.train_batch_size, shuffle=True, collate_fn=dynamic_padding_collate_fn
    )

    logger.info("loading dev dataset")
    dev_examples = load_korquad_dataset(config.dev_dataset)
    dev_dataset = QGDataset(dev_examples, tokenizer, config.max_sequence_length, is_train=False)
    dev_dataloader = DataLoader(dev_dataset, config.eval_batch_size, collate_fn=dynamic_padding_collate_fn)

    # model 생성
    model = GPT2LMHeadModel.from_pretrained(config.gpt_model_hub_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = Adam(model.parameters(), lr=config.lr)
    total_steps = len(train_dataloader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    loss_list_between_log_interval = []
    for epoch_id in range(config.epochs):
        for step_index, batch_data in tqdm(
            enumerate(train_dataloader), f"[TRAIN] EP:{epoch_id}", total=len(train_dataloader)
        ):
            global_step = len(train_dataloader) * epoch_id + step_index + 1
            optimizer.zero_grad()

            input_ids, attention_mask, labels = tuple(value.to(device) for value in batch_data)
            model_outputs = model.forward(input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)

            model_outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()

            # for logging
            loss_list_between_log_interval.append(model_outputs.loss.item())

            if global_step % config.train_log_interval == 0:
                mean_loss = np.mean(loss_list_between_log_interval)
                logger.info(
                    f"EP:{epoch_id} global_step:{global_step} "
                    f"loss:{mean_loss:.4f} perplexity:{math.exp(mean_loss):.4f}"
                )
                loss_list_between_log_interval.clear()

            if global_step % config.validation_interval == 0:
                _validate(model, dev_dataloader, device, logger, global_step)

            if global_step % config.save_interval == 0:
                state_dict = model.state_dict()
                model_path = os.path.join(config.output_dir, f"gpt2_step_{global_step}.pth")
                logger.info(f"global_step: {global_step} model saved at {model_path}")
                torch.save(state_dict, model_path)


def _validate(
    model: GPT2LMHeadModel,
    dev_dataloader: DataLoader,
    device: torch.device,
    logger: logging.Logger,
    global_step: int,
):
    model.eval()
    loss_list = []
    for batch_data in tqdm(dev_dataloader, desc="[EVAL]"):
        with torch.no_grad():
            input_ids, attention_mask, labels = tuple(value.to(device) for value in batch_data)
            model_outputs = model.forward(input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
            loss_list.append(model_outputs.loss.item())

    mean_loss = np.mean(loss_list)
    logger.info(f"[EVAL] global_step:{global_step} loss:{mean_loss:.4f} perplexity:{math.exp(mean_loss):.4f}")
    model.train()


def _create_logger(output_dir: str):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s")

    file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(handler)
    return logger


if __name__ == "__main__":
    kwargs = {key: value for key, value in vars(parser.parse_args()).items() if value is not None}

    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    artifacts_dir = os.path.join(kwargs["output_dir"], f"gpt2_{timestamp}")
    os.makedirs(artifacts_dir, exist_ok=True)
    kwargs["output_dir"] = artifacts_dir

    main(QGConfig(**kwargs))
