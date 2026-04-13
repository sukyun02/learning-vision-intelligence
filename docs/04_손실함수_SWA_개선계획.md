# 손실함수 · SWA 개선 계획서

**작성 기준일**: 2026-04-13  
**담당 범위**: PyramidNet-272 단일 모델 — 손실함수 및 SWA 하이퍼파라미터 개선  
**목표**: Fine Top-1 ↑ + Super-Class Accuracy 유지 (현 91% → 92%+ 유지하면서 Top-1 개선)

> 본 문서는 `03_코드개발계획.md`의 Phase C (Loss)와 Phase G (SWA)를 이어받아,  
> 아래 세 가지 개선 사항을 구현하기 전에 **설계 의도 · 수식 · 파일 변경 범위**를 먼저 정리한다.

---

## 개선 목록 요약

| # | 개선 항목 | 변경 파일 | 기대 효과 |
|---|---|---|---|
| 1 | Superclass-Aware Label Smoothing | `losses/hierarchical_loss.py` | SC Density 직접 개선 + Top-1 긍정적 |
| 2 | Hierarchical Loss λ 조정 (0.8 → 0.4) | `losses/hierarchical_loss.py`, `train.py` | Top-1에 집중, 총점 향상 |
| 3 | SWA 시점 조정 (마지막 25% → 15%) | `train.py` | 충분히 수렴한 weight만 평균, SWA 품질 향상 |

---

## 개선 1. Superclass-Aware Label Smoothing

### 1-1. 동기

표준 Label Smoothing은 smoothing 확률 ε을 100개 클래스에 **균등 분배**한다.

```
p(i) = (1 - ε) · δ(i == y)  +  ε / 100
```

CIFAR-100처럼 20개 superclass가 존재하는 경우, 같은 super-class 내 sibling 클래스들은 **의미상 가깝다**.  
이들에게 더 많은 smoothing 확률을 분배하면 모델이 "같은 superclass 안에서 헷갈리는 것"을 허용하여  
superclass accuracy를 유지하면서 fine-grained 판별력도 개선된다.

### 1-2. 수식

target class `y`가 superclass `c`에 속하고, 같은 superclass 내 클래스 수를 `n_intra`라 할 때:

```
p(y)                          = 1 - ε
p(i | same superclass, i ≠ y) = ε · intra_ratio / (n_intra - 1)
p(i | different superclass)   = ε · (1 - intra_ratio) / (100 - n_intra)
```

| 파라미터 | 의미 | 기본값 |
|---|---|---|
| `epsilon` | 전체 smoothing 질량 | 0.1 |
| `intra_ratio` | ε 중 같은 superclass에 분배하는 비율 | 0.5 |

CIFAR-100은 superclass당 평균 5개 fine class → `n_intra ≈ 5`.

예시 (n_intra=5, ε=0.1, intra_ratio=0.5):
- `p(y)` = 0.90
- 같은 superclass sibling 4개 각각: 0.1 × 0.5 / 4 = **0.0125**
- 다른 superclass 95개 각각: 0.1 × 0.5 / 95 ≈ **0.000526**

### 1-3. 구현 위치

**파일**: `losses/hierarchical_loss.py`

| 추가 함수/변경 | 역할 |
|---|---|
| `build_sc_aware_soft_targets(epsilon, intra_ratio)` | (100, 100) soft target 행렬 생성 (학습 전 1회) |
| `HierarchicalLoss._get_soft_matrix(device)` | 행렬을 device로 lazy-init |
| `HierarchicalLoss._ce_fine(logits, targets)` | `F.log_softmax` × soft_target 합산으로 CE 계산 |

CutMix 대응:
```python
loss_fine = lam * _ce_fine(logits, la) + (1 - lam) * _ce_fine(logits, lb)
```

### 1-4. HierarchicalLoss 생성자 변경

| 파라미터 | 이전 | 이후 |
|---|---|---|
| `lam_coarse` | 0.8 | 0.4 (개선 2와 연동) |
| `label_smoothing` | 0.0 (uniform, F.cross_entropy에 넘김) | **제거** |
| `epsilon` | (없음) | **0.1** (SC-aware smoothing 총량) |
| `intra_ratio` | (없음) | **0.5** (절반을 intra-class에 분배) |

---

## 개선 2. Hierarchical Loss λ 조정 (0.8 → 0.4)

### 2-1. 동기

현재 SC Density(Super-Class Accuracy)가 이미 91%로 높다.  
총점 = Top-1 + SC Density 이므로 Top-1을 올리는 것이 더 효과적이다.

| 상태 | λ 해석 |
|---|---|
| `λ = 0.8` | CE_coarse 80%, CE_fine 20% → coarse에 너무 치우침 |
| `λ = 0.4` | CE_coarse 40%, CE_fine 60% → fine 학습 비중 증가 |

### 2-2. 손실 수식 변경

```
이전: L = 0.2 · CE_fine  +  0.8 · CE_coarse
이후: L = 0.6 · CE_fine  +  0.4 · CE_coarse
```

### 2-3. 변경 범위

| 파일 | 변경 내용 |
|---|---|
| `losses/hierarchical_loss.py` | `HierarchicalLoss.__init__` 기본값 `lam_coarse=0.4` |
| `train.py` | `HierarchicalLoss(lam_coarse=0.4, epsilon=0.1, intra_ratio=0.5)` |
| `train.py` wandb config | `"loss": "HierarchicalLoss λ=0.4 + SC-LS ε=0.1"` |

---

## 개선 3. SWA 시점 조정 (마지막 25% → 15%)

### 3-1. 동기

| 상태 | SWA 시작 시점 | SWA 구간 |
|---|---|---|
| 이전 (1800ep 기준) | epoch 1350 | 450ep (25%) |
| 이후 (1800ep 기준) | epoch 1530 | 270ep (15%) |
| 이후 (300ep 예시) | epoch 255 | 45ep (15%) |

SWA는 "수렴 이후" weight를 평균내야 효과적이다.  
너무 일찍 시작하면 아직 최적화 중인 weight가 섞여 품질이 떨어진다.  
마지막 15%는 충분히 수렴한 구간이라 SWA 평균의 품질이 높아진다.

### 3-2. 구현 방식

`--swa_start_ratio` CLI 인자 추가 (default: 0.85).

```python
# train.py
parser.add_argument("--swa_start_ratio", type=float,
                    default=float(os.getenv("SWA_START_RATIO", 0.85)),
                    help="SWA 시작 비율. 0.85 = 전체 epoch의 마지막 15%%부터 SWA 적용")
```

`swa_start` 계산:
```python
swa_start  = max(int(args.epochs * args.swa_start_ratio), 1)
swa_epochs = args.epochs - swa_start   # 로깅용
```

기존 `--swa_epochs` 인자는 **제거하지 않고 유지** (하위호환).  
단, `--swa_start_ratio`가 명시되면 이를 우선 사용.

### 3-3. 변경 범위

| 파일 | 변경 내용 |
|---|---|
| `train.py` | `--swa_start_ratio` 인자 추가, `swa_start` 계산 방식 변경 |
| `train.py` wandb config | `"swa_start_ratio": args.swa_start_ratio` 추가 |

---

## 파일별 변경 범위 요약

### `losses/hierarchical_loss.py`

```
추가:
  + build_sc_aware_soft_targets(epsilon, intra_ratio) → Tensor(100,100)
  + HierarchicalLoss._get_soft_matrix(device)
  + HierarchicalLoss._ce_fine(logits, targets)

변경:
  ~ HierarchicalLoss.__init__  : lam_coarse=0.4, epsilon=0.1, intra_ratio=0.5
  ~ HierarchicalLoss.forward   : _ce → _ce_fine 교체 (CutMix 포함)

제거:
  - label_smoothing 파라미터 (F.cross_entropy의 label_smoothing 인자 방식 폐기)
  - _ce() 헬퍼 (uniform smoothing용)
```

### `train.py`

```
변경:
  ~ criterion = HierarchicalLoss(lam_coarse=0.4, epsilon=0.1, intra_ratio=0.5)
  ~ swa_start 계산 : args.epochs - args.swa_epochs → int(args.epochs * args.swa_start_ratio)
  ~ wandb config  : loss 문자열, swa_start_ratio 항목 추가

추가:
  + --swa_start_ratio 인자 (default 0.85)
```

---

## 검증 계획

### 단위 검증

```python
# soft target 행렬 체크 (rows sum to 1)
mat = build_sc_aware_soft_targets(epsilon=0.1, intra_ratio=0.5)
assert mat.shape == (100, 100)
assert torch.allclose(mat.sum(dim=1), torch.ones(100), atol=1e-6)
assert (mat >= 0).all()

# 같은 superclass intra 확률이 inter보다 큰지 확인
# fine class 0 → superclass 4 (FINE_TO_COARSE[0]=4)
# sibling: 클래스 4,5,6,7,8,9,... 중 superclass==4인 것들
```

### 학습 검증

- 1-epoch 스모크: loss 정상 감소 확인
- 10ep 소규모 run: `epsilon=0.0`(off) vs `epsilon=0.1`(on) 비교
- wandb metric: `val/fineclass_acc`, `val/superclass_acc` 추이 비교

---

## 리스크 및 완화

| 리스크 | 완화 방안 |
|---|---|
| λ 낮추면 SC Density 하락 가능 | `λ=0.4`는 중간값; 0.3까지 내리기 전 10ep 실험 먼저 |
| SC-aware smoothing이 오히려 fine accuracy 방해 | `epsilon=0.05`로 낮춰 보수적으로 시작 가능 |
| SWA 구간 줄이면 평균 품질 부족 가능 | 15%도 300ep 기준 45ep → 충분. 결과 나쁘면 20%로 복원 |

---

## 실행 순서

```bash
# 1. 코드 변경 (losses/hierarchical_loss.py, train.py)

# 2. 단위 검증
python - <<'EOF'
import torch
from losses.hierarchical_loss import build_sc_aware_soft_targets
mat = build_sc_aware_soft_targets(0.1, 0.5)
print(mat.shape, mat.sum(dim=1).min().item(), mat.sum(dim=1).max().item())
EOF

# 3. 1-epoch 스모크
python train.py --epochs 1

# 4. 10ep 비교 실험 (선택)
python train.py --epochs 10 --seed 42

# 5. 본 학습
python train.py --seed 42 --swa_start_ratio 0.85
```
