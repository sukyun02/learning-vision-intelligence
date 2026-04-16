# DeiT-III-Small on CIFAR-100 From Scratch 학습 프로세스

## 1. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 모델 | DeiT-III-Small (21.4M parameters) |
| 데이터셋 | CIFAR-100 (32×32, 100 fine classes, 20 superclasses) |
| 학습 방식 | From scratch (pretrained 없음, 외부 데이터 없음) |
| VRAM 제한 | 12GB |
| 핵심 메트릭 | Superclass Top-5 Density |
| 실험 추적 | Weights & Biases (WandB) |
| 패키지 관리 | uv |

---

## 2. 모델 아키텍처 — DeiT-III-S 직접 구현

timm 등 외부 라이브러리 없이 `torch.nn`만으로 전체 아키텍처를 구현했다.

### 2.1 아키텍처 스펙

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| Embedding Dim | 384 | |
| Depth (Transformer Blocks) | 12 | |
| Attention Heads | 6 | head_dim = 64 |
| MLP Ratio | 4 | hidden = 1536 |
| Patch Size | **4×4** | 원본 DeiT-III는 16×16 (224×224용) |
| Sequence Length | 65 | 8×8 = 64 patches + 1 CLS token |
| Fine Head | Linear(384, 100) | Fine-class classification |
| Coarse Head | Linear(384, 20) | Auxiliary superclass classification |
| Total Parameters | ~21.4M | Fine head + Coarse head 포함 |

### 2.2 CIFAR-100 적응을 위한 핵심 변경

원본 DeiT-III-S는 ImageNet (224×224)용으로 patch_size=16을 사용한다. CIFAR-100 (32×32)에서는 patch_size=16이면 2×2 = 4개 패치밖에 나오지 않아 학습이 불가능하다. 따라서 **patch_size=4**로 변경하여 8×8 = 64개 패치를 생성한다.

```
32×32 이미지 → Conv2d(3, 384, kernel_size=4, stride=4) → 8×8 = 64 patches → (B, 64, 384)
```

### 2.3 DeiT-III 핵심 컴포넌트

**LayerScale** — DeiT-III의 핵심 기여
- 각 Transformer block의 residual branch 출력에 채널별 학습 가능한 스케일링 적용
- 초기값 1e-6 (매우 작은 값) → 학습 초기에 residual을 거의 무시하고 점진적으로 활성화
- 학습 안정성을 크게 향상시킴

```python
# Pre-norm + LayerScale 구조
x = x + drop_path(ls1 * attn(norm1(x)))  # Attention branch
x = x + drop_path(ls2 * mlp(norm2(x)))   # MLP branch
```

**Stochastic Depth (DropPath)**
- 학습 시 일정 확률로 전체 residual branch를 drop
- Drop rate는 layer별로 선형 증가 (0 → drop_path_rate)
- 과적합 방지 + 앙상블 효과

**Attention**
- `F.scaled_dot_product_attention` 사용 → PyTorch native Flash Attention 자동 활용

---

## 3. 데이터 파이프라인

### 3.1 CIFAR-100 데이터셋

| 항목 | Train | Test |
|------|-------|------|
| 이미지 수 | 50,000 | 10,000 |
| 해상도 | 32×32 | 32×32 |
| Fine classes | 100 (클래스당 500장) | 100 (클래스당 100장) |
| Superclasses | 20 (각 5개 fine class 포함) | 20 |

### 3.2 3-Augment (DeiT-III 논문)

기존 ViT 학습에 사용되던 복잡한 augmentation (RandAugment, AutoAugment) 대신, 단 3가지 augmentation만 사용한다:

1. **Grayscale** — 컬러를 흑백으로 변환 후 3채널로 복원
2. **Solarization** — 밝기 128 이상 픽셀 반전
3. **Gaussian Blur** — kernel_size=3 (32×32에 적합)

매 이미지마다 3개 중 1개를 랜덤 선택한다.

**전체 Training Transform 순서:**
```
RandomCrop(32, padding=4) → RandomHorizontalFlip → ThreeAugment → ToTensor → Normalize
```

### 3.3 Mixup / CutMix (직접 구현)

배치 레벨에서 적용. BCE loss와 호환되는 soft target을 생성한다.

| 기법 | Alpha | 설명 |
|------|-------|------|
| Mixup | 0.8 | 두 이미지를 Beta(0.8, 0.8)에서 샘플한 λ로 선형 보간 |
| CutMix | 1.0 | 랜덤 영역을 잘라 다른 이미지에 붙임 |
| Switch Prob | 0.5 | 매 배치마다 Mixup/CutMix 중 하나를 50% 확률로 선택 |

### 3.4 Superclass Top-5 Density 평가

핵심 메트릭은 **Superclass Top-5 Density**이다. 각 샘플에 대해:
1. Fine-class logits에서 Top-5 예측 fine class를 추출
2. 이 중 GT superclass에 속하는 fine class 개수를 세고
3. 해당 개수 / 5 = density score

```python
gt_coarse = fine_to_coarse[fine_targets]          # GT superclass
_, top_k_fine = logits.topk(5, dim=1)              # Top-5 predicted fine classes
top_k_coarse = fine_to_coarse[top_k_fine]          # 각 예측의 superclass
matches = top_k_coarse.eq(gt_coarse.unsqueeze(1))  # GT superclass와 일치 여부
density = matches.float().sum(dim=1) / 5           # per-sample density
# → 전체 평균 = Superclass Top-5 Density
```

**예시**: GT=Maple (Trees), Top-5=[Maple, Oak, Pine, Apple, Rose] → Trees에 속하는 것 3개 → 3/5 = 0.6

---

## 4. 학습 레시피

### 4.1 DeiT-III 학습 특징

| 항목 | DeiT-III 원본 | 본 프로젝트 적용 |
|------|---------------|-----------------|
| Loss | BCE (Binary Cross-Entropy) | BCEWithLogitsLoss + Auxiliary Coarse CE ✅ |
| Optimizer | LAMB | AdamW (single-GPU에 적합) |
| Augmentation | 3-Augment | 3-Augment ✅ |
| Mixup/CutMix | 0.8 / 1.0 | 0.8 / 1.0 ✅ |
| LayerScale | init 1e-6 | init 1e-6 ✅ |
| Stochastic Depth | 모델별 상이 | 0.1 (Small) ✅ |
| LR Schedule | Cosine + Warmup | Cosine + Warmup ✅ |
| AMP | — | fp16 (VRAM 절약) |

**BCE Loss를 사용하는 이유:**
- Mixup/CutMix로 생성된 soft target과 자연스럽게 호환
- 각 클래스를 독립적으로 처리하여 더 세밀한 학습 신호 제공
- DeiT-III 논문의 핵심 기여 중 하나

### 4.2 Auxiliary Superclass Classification Head

Fine-class BCE loss만으로는 모델이 superclass 구조를 충분히 학습하지 못할 수 있다. 이를 보완하기 위해 **auxiliary superclass classification head**를 추가했다.

**구조:**
```
CLS token features (384)
├── fine_head: Linear(384, 100) → fine logits   → BCE loss (soft targets)
└── coarse_head: Linear(384, 20) → coarse logits → CE loss (hard targets)
```

**Total Loss:**
```
total_loss = BCE_fine + α × CE_coarse
```
- `α` (coarse_loss_weight): 기본값 0.5 (`--coarse-loss-weight`로 조절)
- Coarse target은 fine target에서 `fine_to_coarse` 매핑으로 자동 도출
- Mixup/CutMix 적용 전의 원본 fine target으로부터 coarse target 생성 (soft label 아닌 hard label)

**효과:**
- 모델이 "최소한 superclass는 맞추도록" 명시적으로 유도
- CLS token의 latent space에서 같은 superclass에 속하는 fine class들이 가까워짐
- Superclass top-5 density 메트릭에 직접적으로 기여
- Stage 4 (full training)에서만 활성화, Stage 1-3에서는 비활성화

### 4.2 하이퍼파라미터

| Parameter | Value | 근거 |
|-----------|-------|------|
| Batch Size | 512 | 32×32 + AMP로 VRAM 여유 (~1GB at bs=64) |
| Epochs | 300 | CIFAR-100 규모에 적합 |
| Learning Rate | LR Sweep에서 탐색 | Karpathy 방법론 |
| Min LR | 1e-5 | Cosine decay 하한 |
| Weight Decay | 0.05 | ViT 표준 |
| Warmup Epochs | 15 | 전체의 ~5% |
| Drop Path Rate | 0.1 | Small 모델 적정 |
| Coarse Loss Weight (α) | 0.5 | Auxiliary superclass head 가중치 |
| Gradient Clipping | max_norm=1.0 | ViT 학습 안정화 |

---

## 5. Karpathy 방식 단계별 학습 프로세스

Andrej Karpathy의 "A Recipe for Training Neural Networks" 방법론을 따라 4단계로 학습을 진행한다. 각 단계는 이전 단계의 검증이 완료된 후에만 진행한다.

### Stage 1: Overfit Single Batch (`--stage overfit-batch`)

**목적**: 모델 + 옵티마이저 + loss 함수가 정상 동작하는지 검증

**방법**:
- 학습 데이터에서 배치 1개만 추출
- Augmentation, regularization 전부 OFF
- 500 step 반복 학습

**기대 결과**: loss → 0, accuracy → 100% 수렴

**실패 시**: 모델 구조, loss, optimizer에 버그 존재

### Stage 2: LR Sweep (`--stage lr-sweep`)

**목적**: 최적 Learning Rate 탐색

**방법**:
- 9개 LR을 로그 스케일로 테스트: `1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1`
- 각 LR당 10 epoch 학습
- WandB에 별도 run으로 기록

**결과 확인**: WandB에서 LR별 loss curve 비교 → 가장 빠르게 안정적으로 수렴하는 LR 선택

### Stage 3: Baseline (`--stage baseline`)

**목적**: Augmentation 없이 baseline 성능 확인

**방법**:
- Stage 2에서 찾은 최적 LR 사용
- Augmentation / Mixup / CutMix / Stochastic Depth 전부 OFF
- 전체 epoch 학습

**확인 사항**:
- Train vs Val 성능 차이 → 과적합 정도 파악
- 이후 Stage 4에서 regularization 효과를 정량적으로 비교 가능

### Stage 4: Full Training (`--stage full`)

**목적**: 최종 성능 달성

**방법**:
- 3-Augment + Mixup/CutMix + Stochastic Depth + Weight Decay 전부 ON
- 최적 LR + Cosine Schedule + Warmup

**비교**: Stage 3 baseline 대비 regularization 효과 확인

---

## 6. 상세 로깅 체계

WandB에 학습 전 과정을 빠짐없이 기록하여, 로그만으로 전체 과정을 문서화할 수 있다.

### 학습 시작 시 (Run Config)

- 모든 하이퍼파라미터
- 학습 Stage
- 모델 파라미터 수
- GPU 이름, PyTorch/CUDA 버전
- 랜덤 시드

### 매 Step

| 메트릭 | 설명 |
|--------|------|
| `train/loss` | BCE loss |
| `train/lr` | 현재 learning rate |
| `train/grad_norm` | Gradient L2 norm (학습 안정성) |
| `train/throughput` | images/sec |
| `train/gpu_memory_mb` | VRAM 사용량 |
| `grad_norm/<layer>` | Layer별 gradient norm (매 50 step) |

### 매 Epoch

| 메트릭 | 설명 |
|--------|------|
| `val/loss` | Validation CE loss |
| `val/fine_top1` | Fine-class Top-1 Accuracy |
| `val/fine_top5` | Fine-class Top-5 Accuracy |
| `val/sc_top5_density` | **Superclass Top-5 Density (핵심 메트릭)** |
| `val/per_superclass_density` | 20개 superclass별 개별 density (WandB table) |
| `val/confusion_matrix` | Superclass confusion matrix (매 50 epoch) |

### 모델 진단 (매 10 Epoch)

| 메트릭 | 설명 |
|--------|------|
| `weight_norm/<layer>` | Layer별 weight norm |
| `layerscale/<layer>` | LayerScale 값 분포 (mean/std/min/max) |

### 체크포인트

- Best superclass_top5 기준 저장
- 매 50 epoch 중간 체크포인트
- 학습 종료 시 최종 summary (최고 성능 + 해당 epoch + 총 학습 시간)

---

## 7. 프로젝트 구조

```
vision/
├── pyproject.toml              # 의존성 (torch, torchvision, wandb)
├── main.py                     # 진입점: 4-stage 오케스트레이션 + WandB
└── vit/
    ├── __init__.py
    ├── model.py                # DeiT-III-S 아키텍처 (from scratch)
    │   ├── PatchEmbed          # Conv2d 기반 패치 임베딩
    │   ├── Attention           # Multi-head Self-Attention
    │   ├── Mlp                 # Feed-Forward Network
    │   ├── DropPath            # Stochastic Depth
    │   ├── TransformerBlock    # Pre-norm + LayerScale
    │   └── VisionTransformer   # 전체 모델 (fine head + coarse head)
    ├── data.py                 # CIFAR-100 데이터 파이프라인
    │   ├── ThreeAugment        # 3-Augment (DeiT-III)
    │   ├── MixupCutmix         # Mixup + CutMix (직접 구현)
    │   └── CIFAR100WithCoarse  # Superclass target 포함 wrapper
    ├── train.py                # 학습 루프 + step별 로깅
    ├── evaluate.py             # 검증 + superclass 메트릭
    └── superclass.py           # Fine→Coarse 매핑
```

---

## 8. 실행 커맨드

```bash
# 의존성 설치
uv add torch torchvision --default-index https://download.pytorch.org/whl/cu128
uv add wandb --index https://pypi.org/simple

# WandB 로그인
uv run wandb login

# Stage 1: Sanity check
uv run python main.py --stage overfit-batch

# Stage 2: LR 탐색
uv run python main.py --stage lr-sweep

# Stage 3: Baseline (augmentation OFF)
uv run python main.py --stage baseline --lr <최적LR>

# Stage 4: Full training (with auxiliary superclass head)
uv run python main.py --stage full --lr <최적LR> --coarse-loss-weight 0.5
```

---

## 9. 기술적 결정 및 근거

| 결정 | 근거 |
|------|------|
| patch_size=4 (원본 16) | 32×32에서 16이면 패치 4개뿐. 4로 변경하여 64 패치 확보 |
| AdamW (원본 LAMB) | LAMB은 대규모 분산 학습용. Single-GPU + batch 512에서는 AdamW가 안정적 |
| AMP (fp16) | VRAM 절약 + 속도 향상. 12GB 제한 준수 |
| timm 미사용 | 학습 목적으로 아키텍처를 처음부터 구현 |
| BCE Loss | DeiT-III 핵심. Mixup soft target과 호환. CE 대비 더 세밀한 학습 신호 |
| Auxiliary Coarse Head | Superclass 구조를 명시적으로 학습. Density 메트릭 직접 개선 |
| Karpathy 4-stage | 체계적 디버깅. 각 단계에서 문제를 조기 발견 |
