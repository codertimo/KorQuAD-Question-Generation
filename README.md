# Question Generation(QG) Model with KorQuAD

학습된 [SKT-AI/KoGPT2](https://github.com/SKT-AI/KoGPT2) 모델을 기반으로 질문 생성 QG(Question Generation) 모델을 만들었습니다. 
QG 모델을 만들기 위해 Question Answering 데이터셋인 KorQuAD v1.0을 사용하였습니다.

## 사용 방법

### 데이터 준비

학습/평가/생성을 위해서 KorQuAD v1.0 데이터셋을 다운 받습니다.

```shell
make prepare-dataset
```

### 학습

다음 커맨드를 이용해서 학습을 수행할 수 있습니다.

```shell
python -m scripts.run_fine_tune --train-batch-size 16 --eval-batch-size 16 --epochs 5
```

### 성능 평가 (dev 셋 PPL 측정)

```shell
MODEL_PATH = "artifacts/gpt2_xxxxxxxx/gpt2_step_x.pth"
python -m scripts.run_evaluation --model-path $MODEL_PATH --batch-size 50
```

### 질문 생성 (dev 셋에 대해서 질문 생성)

![Decoding 결과](docs/decoded_examples.png)

[Question Generation POC 스프레드 시트](https://docs.google.com/spreadsheets/d/1-PQKFTfBhyH-K0EBa03_KkPMgL86wF_rn_813Hk-LNQ): KorQuAD v1.0 dev 셋에 대해서 decoding 한 결과 입니다.

beam-search 를 기반으로 decoding 되었으며, beam_size 는 5를 사용하였습니다.

```shell
MODEL_PATH = "artifacts/gpt2_xxxxxxxx/gpt2_step_x.pth"
python -m scripts.run_generate --model-path $MODEL_PATH --output-path decoded.tsv
```

### 학습된 QG 모델 다운로드

- [모델 weight](https://drive.google.com/file/d/1t6ChVQwp3MapJDVEBCnyJcdVurIvn7y7/view?usp=sharing)
- [학습 로그](https://drive.google.com/file/d/1bMi_AA5nhTt72iEIzcg5gbyN5AYdZxI8/view?usp=sharing)

## Author

by Junseong Kim (Scatter Lab, Pingpong AI) codertimo@gmail.com
