# PyramidNet-272 학습 최적화 보고서 (2차)

> 이전 보고서 이후 추가 논의된 내용만 정리

---

## 1. 앙상블 전략

### 1.1 모델 조합 선정

PyramidNet-272와 앙상블 시 architecture diversity가 핵심이다. CNN끼리(PyramidNet + WRN)는 inductive bias가 같아 오답 패턴이 겹치므로, ViT 계열인 DHVT를 포함해야 상호 보완 효과가 극대화된다.

최종 권장 조합은 **3개 앙상블(PyramidNet-272 + WRN-28-10 + DHVT)**이며, GPU 여유가 있을 경우 EfficientNet-B4를 추가한 4개 앙상블까지 고려할 수 있다.

### 1.2 EfficientNet-B4 추천 근거

- 파라미터 약 19M, 24시간 내 학습 가능
- Compound scaling(depth + width + resolution 동시 조절)으로 PyramidNet(depth 중심), WRN(width 중심)과 feature 추출 방식이 상이
- SE block(channel attention) 포함으로 CNN이면서도 차별화된 오답 패턴 기대

### 1.3 세부 모델 설정

| 모델 | 세부 설정 | 비고 |
|------|-----------|------|
| PyramidNet | 272, alpha=200 | 현재 유지, 시간 대비 표현력 최적 |
| WRN | 28-10 | 가장 널리 검증된 설정, 변경 불필요 |
| DHVT | Small 추천 | Tiny는 capacity 부족, Base는 underfitting 우려 |
| EfficientNet | B4 | B5는 24시간 빠듯, B3는 capacity 아쉬움 |

### 1.4 앙상블 가중치

가중치는 각 모델의 단독 성능에 비례하여 초기값을 설정하고, val set grid search로 최적화한다.

**3개 앙상블 시작점:**
WRN-28-10 (0.40) > PyramidNet-272 (0.35) > DHVT (0.25)

**4개 앙상블 시작점:**
WRN-28-10 (0.35) > PyramidNet-272 (0.25) > DHVT (0.20) = EfficientNet-B4 (0.20)

- DHVT는 단독 성능이 낮아도 유일한 ViT 계열로 diversity 기여가 크므로 제외하지 않는다.
- 미학습 모델(EfficientNet-B4)은 성능 확인 전까지 보수적으로 배정한다.

### 1.5 앙상블 모델 수에 대한 판단

- 3~5개가 실용적 최적, 그 이상은 이득이 급격히 감소
- 단독 성능이 현저히 낮은 모델을 포함하면 오히려 전체 성능이 악화될 수 있음
- **앙상블은 마무리 단계**이며, 각 모델의 개별 성능을 최대한 뽑아낸 후 적용하는 것이 원칙

---

## 2. SWA LR 문제 진단 및 수정

### 2.1 문제 현상

학습 종료 직전 train/loss가 급등하는 현상이 관찰되었다. val acc(superclass ~91, fineclass ~79)에는 큰 영향이 없었으나, SWA averaging 품질 저하 가능성이 존재한다.

### 2.2 원인

| 항목 | 값 |
|------|-----|
| SWA 시작 직전 LR | 0.003 |
| swa_lr 설정값 | 0.005 |

SWA 활성화 시 LR이 0.003 → 0.005로 약 2배 점프하여 weight가 최적점에서 크게 벗어났다.

### 2.3 수정 권장

`swa_lr`을 **0.002**로 변경한다. SWA 시작 시점의 LR보다 같거나 약간 낮아야 weight가 최적점 근처에서 적절히 탐색하면서 안정적인 averaging이 가능하다.
