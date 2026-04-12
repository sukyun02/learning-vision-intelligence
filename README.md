# PyramidNet-272 α200 + ShakeDrop — CIFAR-100

**Team VisionMachine · DAI3004 Supervised Learning Challenge**

## 모델 구성 (Proposal 기준)

| 항목 | 설정 |
|------|------|
| Architecture | PyramidNet-272, α=200, Bottleneck |
| Regularization | ShakeDrop (p linearly increases 0→0.5) |
| Augmentation | AutoAugment + Cutout (16px) + CutMix (α=1.0) |
| Optimizer | SGD lr=0.1, momentum=0.9, weight_decay=5e-4, Nesterov |
| SWA | Last 250 epochs (epoch 75% 이후), swa_lr = lr×0.05 |
| Loss | Hierarchical Loss: 0.2×CE_fine + 0.8×CE_coarse |
| Target | Top-1 ≥ 84–85%, Super-Class ≥ 93% |

## 파일 구조

```
pyramidnet_cifar100/
├── models/
│   └── pyramidnet.py          # PyramidNet + ShakeDrop 구현
├── data/
│   └── cifar100.py            # DataLoader, CutMix, 슈퍼클래스 매핑
├── losses/
│   └── hierarchical_loss.py   # 계층적 손실 함수 (λ=0.8)
├── train.py                   # 전체 훈련 파이프라인
├── evaluate.py                # TTA + 슈퍼클래스 정확도 평가
├── run_seeds.py               # 3-seed 재현성 테스트
└── requirements.txt
```



## 실행 방법
```bash

# 가상환경 생성
python -m venv venv

# 가상환경 활성화
venv\Scripts\activate

# 패키지 다운로드
pip install -r requirements.txt

# 단일 seed 훈련 (.env에 하이퍼파라미터 설정)
python train.py

# 3-seed 재현성 검증
python run_seeds.py

# 평가 (TTA + super-class correction)
python evaluate.py --ckpt checkpoints/swa_final_seed42.pth

# 빠른 스모크 테스트 (5 epoch)
python train.py --seed 42 --epochs 5
```

## 주요 구현 포인트

### ShakeDrop
- 블록 인덱스에 비례해 drop probability가 0→0.5로 선형 증가
- Forward: `(gate + α - gate×α) × x`  (α ∈ [-1, 1])
- Backward: `(gate + β - gate×β) × grad`  (β ∈ [0, 1])

### Super-Class Hierarchical Loss
```
L = 0.2 × CE(fine_logits, fine_labels)
  + 0.8 × NLL(log_softmax_coarse, coarse_labels)
```
- fine logits → coarse logits: log-prob을 슈퍼클래스 멤버별로 합산

### SWA (Stochastic Weight Averaging)
- epoch 75%부터 SWA 시작 (마지막 250 epoch, `SWA_EPOCHS=250`)
- BN 통계는 매 evaluation 전 train set으로 재계산

> 📄 상세 계획·다이어그램은 [docs/pipeline.md](docs/pipeline.md), [docs/코드개발계획.md](docs/코드개발계획.md), [docs/환경세팅.md](docs/환경세팅.md) 참조.

### TTA
- horizontal flip + original 평균 (hflip × 1)
- 슈퍼클래스 logit correction: P(fine) × P(super|fine)
