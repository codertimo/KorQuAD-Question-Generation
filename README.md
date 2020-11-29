# Question Generation(QG) Model with KorQuAD

학습된 [SKT-AI/KoGPT2](https://github.com/SKT-AI/KoGPT2) 모델을 initial weight 로 사용하여, KorQuAD 데이터셋을 fine-tuning 하여 질문 생성(Question Generation) 모델을 만들었습니다.

## 사용 방법

### 데이터 준비

```shell
# KorQuAD 1.0 데이터셋을 다운 받습니다.
make prepare-dataset
```

### 학습

```shell
python -m scripts.run_fine_tune \
    --train-batch-size 16 \
    --eval-batch-size 16 \
    --epochs 5
```

### 성능 평가 (dev 셋 PPL 측정)

```shell
MODEL_PATH = "artifacts/gpt2_xxxxxxxx/gpt2_step_x.pth"
python -m scripts.run_evaluation --model-path $MODEL_PATH --batch-size 50
```

### 결과

![Decoding 결과](docs/decoded_examples.png)

[Question Generation POC 스프레드 시트](https://docs.google.com/spreadsheets/d/1-PQKFTfBhyH-K0EBa03_KkPMgL86wF_rn_813Hk-LNQ): KorQuAD v1.0 dev 셋에 대해서 decoding 한 결과 입니다.

beam-search 를 기반으로 decoding 되었으며, beam_size 는 5를 사용하였습니다.

## 모델 공개

WIP

## Author

by @codertimo
