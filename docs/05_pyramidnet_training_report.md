# PyramidNet-272 학습 최적화 보고서

## 1. 초기 적용 사항

이번 실험에서는 PyramidNet-272 (alpha=200) 모델의 성능 개선을 위해 다음 세 가지를 적용하였다.

### 1.1 Superclass-Aware Label Smoothing

같은 superclass 내 클래스에 smoothing 확률을 집중 배분하는 방식으로, SC Density와 Top-1 정확도 모두에 긍정적 영향을 기대하며 HierarchicalLoss에 통합 구현하였다.

- `epsilon=0.1`: 정답 클래스에 0.9, 나머지 0.1을 분배
- `intra_ratio=0.5`: 0.1 중 절반(0.05)을 같은 superclass 내 클래스에 집중, 나머지 0.05는 전체 균등 배분

### 1.2 Hierarchical Loss λ 조정

기존 `lam_coarse=0.8`에서 `0.4`로 낮추어 적용하였다. SC Density가 이미 91%로 높은 상황에서 Top-1에 더 집중하기 위한 목적이다.

- 적용된 loss 비율: fine_loss 60% + coarse_loss 40%
- Total Score = Top-1 + SC Density 구조상, Top-1 향상이 총점에 더 효과적

### 1.3 SWA 시점 조정

SWA(Stochastic Weight Averaging) 시작 시점을 epoch 225에서 255로 축소하여, 충분히 수렴한 weight만 평균에 포함시키도록 하였다.

---

## 2. 학습 과정 분석

### 2.1 중간 ACC 꺾임 현상

학습 중 val acc가 일시적으로 하락하는 구간이 관찰되었으나, 이는 LR schedule(cosine annealing) 특성상 특정 구간에서 LR이 아직 높아 발생하는 정상적인 현상이다. 꺾임 이후 회복되어 우상향 추세가 유지되었으므로 문제가 아닌 것으로 판단하였다.

### 2.2 LR과 SWA의 관계

- **LR schedule**: optimizer가 weight 업데이트 폭을 결정. Cosine annealing으로 점진 감소.
- **SWA**: LR과 별개로 동작. 학습 후반부 여러 epoch의 model weight를 수집하여 평균을 낸다.
- SWA 활성화 시 기존 cosine schedule을 무시하고 별도의 `swa_lr`로 교체되어, LR이 갑자기 올라가는 것처럼 보이는 현상이 나타난다. 이는 weight가 최적점 근처에서 다양한 위치를 탐색하도록 의도된 설계이다.

### 2.3 SWA 평균 방식

SWA는 수집된 weight의 단순 평균(convex combination)을 사용한다. 중위값 대신 평균을 쓰는 이유는 다음과 같다.

- 수백만 차원의 weight 벡터에 element-wise 중위값을 취하면, 각 파라미터가 서로 다른 시점에서 선택되어 학습 중 실제로 존재한 적 없는 조합이 된다.
- 평균(convex combination)은 결과가 원래 weight들의 convex hull 내부에 위치하므로, loss landscape에서 해석이 자연스럽다.

### 2.4 Train ACC < Val ACC 현상

Train acc(~63%)가 val acc(~79%)보다 낮은 현상이 관찰되었다. 이는 강한 regularization(CutMix, Mixup, RandAugment, label smoothing 등)의 전형적 증상이다.

- Train: augmentation이 적용된 어려운 이미지 + label smoothing으로 confidence 억제
- Val: augmentation 없이 깨끗한 원본 이미지로 평가

다만, train acc가 지나치게 낮으면 모델이 학습 데이터 자체를 충분히 학습하지 못해 val 성능에도 천장이 생길 수 있으므로, augmentation 강도 조절이 필요할 수 있다.

---

## 3. 이전 실행 대비 개선 결과

동일 모델(pyramidnet272_seed42) 기준으로 이전 실행(빨간색)과 현재 실행(초록색)을 비교한 결과:

| 지표 | 이전 실행 | 현재 실행 (학습 중) |
|------|-----------|---------------------|
| val/superclass_acc | ~89 | ~91.07 |
| val/fineclass_acc | ~73 | ~78.91 |
| train/loss | ~10 정체 | ~7.5 이하 |

Label smoothing 적용과 λ 조정이 fine class 구분력 향상에 효과가 있었음을 확인하였다.

---

## 4. 다른 모델과의 비교 및 남은 과제

| 모델 | Superclass | Fineclass (Top-1) |
|------|------------|---------------------|
| ViT | ~80 | ~80 |
| WRN | ~90 | ~85 |
| PyramidNet-272 (현재) | ~91 | ~79 |

PyramidNet은 superclass는 WRN급이나 fineclass가 6%p 낮다. 같은 superclass 내 세부 클래스 구분 능력이 부족한 상태이다.

### 추가 개선 방안

1. **`lam_coarse` 추가 하향**: 0.4 → 0.2로 낮춰 fine class에 capacity 집중
2. **`intra_ratio` 조절**: 0.5 → 0.3 또는 0으로 낮춰, 같은 superclass 내 클래스 간 decision boundary가 흐려지는 문제 방지
3. **Augmentation 강도 조절**: CutMix/Mixup 확률을 줄여 train acc를 75~80 수준으로 올리면서 val acc 동반 상승 여부 확인

---

## 5. PyramidNet-110 전환 검토

팀원으로부터 PyramidNet-110 (alpha=200)으로 전환하면 파라미터가 절반으로 줄어 epoch을 늘릴 수 있다는 제안이 있었다.

### 검토 결과: 비추천

- **파라미터 수 전제가 부정확**: 272는 bottleneck block, 110은 basic block을 사용하여 alpha=200 기준 둘 다 약 26~28M으로 유사하다.
- **표현력 천장 문제**: depth가 줄어들면 표현력의 상한이 낮아지며, epoch 수를 늘려도 이를 보상할 수 없다.
- **현재 overfitting 아님**: train acc < val acc 상태이므로 모델 축소의 근거가 없다.

---

## 6. 24시간 제약 대응 전략

교수님의 24시간 내 학습 완료 지시에 따른 최적 전략을 수립하였다.

### 시간 계산

- 50 epoch = 7시간 → 1 epoch ≈ 8.4분
- 24시간 = 1440분 → 최대 약 **170 epoch** 가능

### 170 epoch 시 예상 성능 (기존 300 epoch schedule 기준)

- Superclass ~87, Fineclass ~72 (300 epoch 대비 각각 4%p, 7%p 하락)
- 성능 하락의 원인은 모델 한계가 아닌, cosine schedule이 300 epoch 기준으로 설계되어 170 시점에 수렴이 덜 된 것

### 최종 권장 설정

| 항목 | 기존 (300 epoch) | 변경 (170 epoch) |
|------|-------------------|-------------------|
| total_epochs | 300 | **170** |
| cosine schedule 기준 | 300 | **170** |
| swa_start | 255 | **145** (마지막 15%) |
| lam_coarse | 0.4 | **0.2** (추가 개선) |

LR schedule을 170 epoch에 맞춰 재설정하면, 해당 시간 내에 cosine이 적절히 감쇠하여 수렴할 수 있으며, PyramidNet-272의 높은 표현력을 최대한 활용할 수 있다.
