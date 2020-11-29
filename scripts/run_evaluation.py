import math
from argparse import ArgumentParser

import numpy as np
import torch
from tokenizers import SentencePieceBPETokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel

from korquad_qg.config import QGConfig
from korquad_qg.dataset import QGDataset, dynamic_padding_collate_fn, load_korquad_dataset

parser = ArgumentParser()
parser.add_argument("-m", "--model-path", type=str, required=True)
parser.add_argument("-b", "--batch-size", type=int, default=50)


def main():
    config = QGConfig()
    args = parser.parse_args()

    model = GPT2LMHeadModel.from_pretrained("taeminlee/kogpt2")
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    tokenizer = SentencePieceBPETokenizer.from_file(
        vocab_filename="tokenizer/vocab.json", merges_filename="tokenizer/merges.txt", add_prefix_space=False
    )
    examples = load_korquad_dataset(config.dev_dataset)
    dataset = QGDataset(examples, tokenizer, config.max_sequence_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dynamic_padding_collate_fn)

    model = model.to(device)
    model.eval()

    model.eval()
    loss_list = []
    for batch_data in tqdm(dataloader, desc="[EVAL]"):
        with torch.no_grad():
            input_ids, attention_mask, labels = tuple(value.to(device) for value in batch_data)
            model_outputs = model.forward(input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
            loss_list.append(model_outputs.loss.item())

    mean_loss = np.mean(loss_list)
    print(f"loss:{mean_loss:.4f} perplexity:{math.exp(mean_loss):.4f}")
    model.train()


if __name__ == "__main__":
    main()
