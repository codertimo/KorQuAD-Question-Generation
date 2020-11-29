from argparse import ArgumentParser

import torch
from tokenizers import SentencePieceBPETokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel

from korquad_qg.config import QGConfig
from korquad_qg.dataset import MAX_QUESTION_SPACE, MIN_QUESTION_SPACE, QAExample, QGDecodingDataset

parser = ArgumentParser()
parser.add_argument("-m", "--model-path", type=str, required=True)
parser.add_argument("-s", "--num-samples", type=int)
parser.add_argument("-b", "--num-beams", type=int, default=5)


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
    examples = [
        QAExample(
            "코로나의 온도는 섭씨 수십만~수백만 도로 추정되는데 이는 태양 표면 온도인 5,000~6,000 K의 100배 이상 되는 수치다.[1] 일식 당시 보이는 코로나의 색깔은 백색이나 보라색인데, 섭씨 수십만도 이상은 올라가야 보라색이 된다. 이것은 태양을 점에너지원에서 파생된 단순한 불덩어리로 가정한다면, 즉 봐서는 열역학 제 2법칙에 정면으로 위배되는 것처럼 보이지만 어디 가서 열역학 제 2법칙은 코로나 현상 때문에 위기에 놓여있다는 소리는 하지 말자. 정보가 없을 뿐, 코로나도 열역학 제 2법칙을 정면으로 위배한다는 근거는 어디에도 없다. 코로나의 온도가 태양 표면보다도 높은 기현상을 설명하기 위한 가설로는 크게 2가지 주류설이 있다.",
            "코로나",
        ),
    ]
    dataset = QGDecodingDataset(examples, tokenizer, config.max_sequence_length)
    dataloader = DataLoader(dataset, batch_size=1)

    model = model.to(device)
    model.eval()

    generated_results = []

    for i, batch in tqdm(enumerate(dataloader), desc="generate", total=len(dataloader)):
        input_ids, attention_mask = (v.to(device) for v in batch)
        origin_seq_len = input_ids.size(-1)

        decoded_sequences = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=origin_seq_len + MAX_QUESTION_SPACE,
            min_length=origin_seq_len + MIN_QUESTION_SPACE,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            do_sample=True,
            num_beams=5,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            num_return_sequences=3,
        )

        for decoded_tokens in decoded_sequences.tolist():
            decoded_question_text = tokenizer.decode(decoded_tokens[origin_seq_len:])
            decoded_question_text = decoded_question_text.split("</s>")[0].replace("<s>", "")
            generated_results.append(
                (examples[i].context, examples[i].answer, examples[i].question, decoded_question_text)
            )

    with open("article_qg.tsv", "a") as f:
        for context, answer, question, generated_question in generated_results:
            f.write(f"문맥\t{context}\n")
            f.write(f"답변\t{answer}\n")
            f.write(f"생성된 질문\t{generated_question}\n")
            if question is not None:
                f.write(f"실제 질문\t{question}\n")
            f.write("\n")

            print(f"문맥\t{context}\n")
            print(f"답변\t{answer}\n")
            print(f"생성된 질문\t{generated_question}\n")
            if question is not None:
                print(f"실제 질문\t{question}")
            print()


if __name__ == "__main__":
    main()
