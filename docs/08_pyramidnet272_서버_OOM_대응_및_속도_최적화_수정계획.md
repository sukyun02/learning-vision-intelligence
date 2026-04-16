# PyramidNet-272 서버 OOM 대응 및 속도 최적화 수정계획

**작성 기준일**: 2026-04-16  
**대상**: `Pyramidnet272/` 서버 학습 패키지  
**상황**: RTX A5000 24GB에서 `batch_size=256` 학습 중 CUDA OOM 발생  
**목표**: `batch_size=256`을 먼저 살려보되, 실패하면 `batch_size=128`로 안전하게 fallback한다.

---

## 1. 판단 변경 요약

기존 판단은 너무 보수적이었다. 이전 문서에서는 `PyramidNet-272 + batch_size=256` 자체가 A5000 24GB에 과하다고 봤지만, 현재 OOM 스냅샷과 코드 구조를 함께 보면 더 정확한 원인은 다음 조합일 가능성이 높다.

1. validation batch가 training batch의 2배다.
2. training은 AMP를 쓰지만, 현재 `evaluate()`는 autocast 없이 FP32 forward를 돈다.
3. eval 시점에는 모델, optimizer state, SWA model 등이 GPU에 남아 있다.
4. PyTorch reserved memory 단편화로 작은 추가 할당도 실패할 수 있다.

따라서 바로 `batch_size=128`로 내리기보다, 먼저 **validation batch 축소 + 단편화 완화 + gradient memory 정리**를 적용해서 `batch_size=256`을 1 epoch smoke test로 검증하는 전략이 더 합리적이다.

---

## 2. OOM 원인 재평가

### 메모리 스냅샷

| 항목 | 값 |
|------|-----|
| GPU 총 용량 | 약 23.68 GiB |
| 프로세스 사용량 | 약 23.64 GiB |
| PyTorch 할당량 | 약 22.63 GiB |
| 실패한 추가 할당 | 약 88 MiB |
| 드라이버/CUDA 오버헤드 | 약 1.04 GiB |
| 실제 부족분 | 약 80 MiB 수준 |

실패한 추가 할당이 88 MiB 정도라면 `batch_size=256` 자체가 완전히 불가능하다고 단정하기는 이르다. 다만 프로세스가 GPU 전체를 거의 꽉 채운 상태였으므로, 작은 코드 차이와 fragmentation에도 쉽게 터질 수 있는 경계선 상태인 것은 맞다.

### 핵심 트리거 후보: validation batch

현재 `Pyramidnet272/data/cifar100.py`는 validation loader를 다음처럼 만든다.

```python
val_loader = DataLoader(
    val_set, batch_size=batch_size * 2, shuffle=False,
    **base_loader_kwargs,
)
```

따라서:

| train batch | val batch |
|-------------|-----------|
| 128 | 256 |
| 256 | 512 |

`batch_size=256` 학습에서는 validation batch가 512가 된다. 여기에 현재 `train_server.py`의 `evaluate()`는 AMP autocast 없이 FP32로 forward를 수행한다. 즉, training batch 256 AMP보다 validation batch 512 FP32 forward가 peak memory를 더 크게 만들 수 있다.

`torch.no_grad()`는 gradient 저장을 막지만, forward 중 필요한 중간 텐서와 workspace 할당이 사라지는 것은 아니다. PyramidNet-272처럼 깊은 CNN에서는 validation batch 512가 실제 OOM 트리거일 가능성이 충분하다.

### 1 epoch OOM의 해석

이번 OOM이 1 epoch smoke test에서 발생했다는 점은 중요하다. 이는 장기 학습 누적 문제라기보다 **단일 epoch 안의 특정 구간에서 peak memory가 한 번 넘친 문제**일 가능성을 높인다.

단, 1 epoch OOM이라고 해서 validation 원인으로 바로 확정하면 안 된다. OOM 위치에 따라 판단이 달라진다.

| OOM 발생 위치 | 해석 | 우선 조치 |
|---------------|------|-----------|
| `Train` progress 중 | batch 256 training 자체가 경계선 | `set_to_none=True`, `channels_last`, 그래도 실패 시 batch 128 |
| `Eval` progress 시작 후 | val batch 512 또는 FP32 eval이 트리거일 가능성 큼 | `val_batch_mult=1`, `empty_cache()` 우선 |
| 최종 `update_bn`/Final SWA 근처 | SWA BN update 또는 final eval 경로가 트리거 | final eval 전 cache 정리, 필요 시 final eval batch 축소 |
| 로그 위치 불명확 | 구간 마커가 부족함 | `--skip_eval` 진단 옵션과 phase log 추가 |

따라서 Phase 0에는 `batch_size=256`을 무조건 포기하지 않되, OOM 발생 위치를 분리하는 진단 절차를 포함한다.

---

## 3. 수정 전략

전략은 다음 순서로 간다.

```text
Phase 0: batch 256 메모리 안정화 smoke test
  |
  |-- 성공 -> Phase 1 속도 최적화 -> Phase 2A batch 256 본학습
  |
  `-- 실패 -> Phase 2B batch 128 fallback 본학습
```

중요한 원칙:

- `batch_size=256`은 먼저 살려본다.
- validation batch는 기본적으로 train batch와 같게 둔다.
- `batch_size=128`은 fallback이다.
- `torch.compile`은 아직 보류한다.
- 변경 후 반드시 1 epoch 기준으로 시간과 peak memory를 비교한다.
- OOM 위치를 train/eval/final SWA로 구분한 뒤 다음 조치를 선택한다.

---

## 4. Phase 0: batch 256 메모리 안정화

학습 의미론을 거의 건드리지 않는 메모리 안정화만 먼저 적용한다.

### 4.1 validation batch multiplier 추가

수정 파일:

- `Pyramidnet272/data/cifar100.py`
- `Pyramidnet272/train_server.py`
- `Pyramidnet272/run_seeds.py`

`get_dataloaders()`에 `val_batch_multiplier` 인자를 추가한다.

```python
def get_dataloaders(
    data_root="./data",
    batch_size=128,
    num_workers=4,
    use_cutmix=True,
    cutmix_alpha=1.0,
    cutmix_prob=0.5,
    seed=42,
    val_batch_multiplier=1,
    prefetch_factor=2,
):
    ...
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size * val_batch_multiplier,
        shuffle=False,
        **base_loader_kwargs,
    )
```

CLI에는 다음 인자를 추가한다.

```text
--val_batch_mult 1
```

기본값은 `1`로 둔다.

예상 효과:

- `batch_size=256`일 때 val batch 512를 256으로 낮춘다.
- eval peak memory를 크게 낮출 수 있다.
- validation metric 의미는 바뀌지 않는다.

### 4.2 eval 직전 empty_cache

수정 파일:

- `Pyramidnet272/train_server.py`

`should_eval` 블록 진입 직전 또는 내부 평가 직전에 추가한다.

```python
if should_eval and device.type == "cuda":
    torch.cuda.empty_cache()
```

최종 SWA 평가 전에도 추가한다.

```python
if device.type == "cuda":
    torch.cuda.empty_cache()
update_bn(train_loader, swa_model, device=device)
```

주의:

- `empty_cache()`는 살아 있는 tensor를 지우지 않는다.
- reserved cache와 fragmentation 완화에 도움을 줄 수 있다.
- 성능을 조금 희생할 수 있지만, eval interval이 20이면 부담은 작다.

### 4.3 optimizer.zero_grad(set_to_none=True)

수정 파일:

- `Pyramidnet272/train_server.py`

기존:

```python
optimizer.zero_grad()
```

변경:

```python
optimizer.zero_grad(set_to_none=True)
```

효과:

- gradient tensor를 0으로 채우지 않고 None으로 둔다.
- 마지막 train batch 이후 gradient memory가 남아 eval 직전 peak에 영향을 주는 상황을 줄일 수 있다.
- PyTorch 권장 패턴이며 학습 의미론은 유지된다.

### 4.4 OOM 위치 진단용 --skip_eval

수정 파일:

- `Pyramidnet272/train_server.py`

진단 전용 CLI를 추가한다.

```text
--skip_eval
```

동작:

- periodic validation을 건너뛴다.
- final SWA BN update/evaluation도 건너뛴다.
- checkpoint 저장은 evaluation metric이 없으므로 수행하지 않는다.
- 학습 loop 자체가 batch 256으로 통과하는지 확인하는 데만 사용한다.

진단 명령:

```bash
python train_server.py \
  --seed 42 \
  --epochs 1 \
  --batch_size 256 \
  --lr 0.2 \
  --skip_eval
```

판정:

| 결과 | 해석 |
|------|------|
| `--skip_eval`도 OOM | train batch 256 자체가 부족할 가능성 큼 |
| `--skip_eval`은 성공, 일반 smoke test는 OOM | eval/update_bn/final eval 경로가 트리거일 가능성 큼 |

주의:

- `--skip_eval`은 최종 학습용 옵션이 아니다.
- OOM 원인 분리를 위한 smoke test에만 사용한다.

### 4.5 phase log 추가

수정 파일:

- `Pyramidnet272/train_server.py`

OOM 로그 위치를 쉽게 판정하도록 주요 구간 앞에 명확한 print를 추가한다.

```python
print(f"[phase] train epoch={epoch}")
print(f"[phase] eval epoch={epoch}")
print("[phase] final_swa_update_bn")
print("[phase] final_swa_eval")
```

이 로그가 있으면 서버 로그만 보고도 train/eval/final SWA 중 어디에서 터졌는지 판단할 수 있다.

### 4.6 expandable_segments 환경변수

수정 파일:

- `Pyramidnet272/command.txt`
- `Pyramidnet272/setup_server.sh`
- 필요 시 `docs/07_pyramidnet272_서버_학습_환경_구성_가이드.md`

실행 전 다음을 명시한다.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

이 설정은 Python 프로세스 시작 전에 적용되어야 한다.

### 4.7 Phase 0 검증 명령

서버 기본 shell이 `sh`/`dash`라면 venv 활성화는 `source` 대신 `.`를 쓴다.

```bash
cd Pyramidnet272
. venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi

python train_server.py \
  --seed 42 \
  --epochs 1 \
  --batch_size 256 \
  --lr 0.2 \
  --eval_interval 1 \
  --val_batch_mult 1
```

판정:

- 성공하면 Phase 1로 간다.
- OOM이면 먼저 `--skip_eval` 진단 명령으로 train/eval 경로를 분리한다.
- train loop 자체가 OOM이면 `batch_size=128`, `lr=0.1`로 fallback한다.

진단용 eval skip:

```bash
python train_server.py \
  --seed 42 \
  --epochs 1 \
  --batch_size 256 \
  --lr 0.2 \
  --skip_eval
```

---

## 5. Phase 1: 공통 속도 최적화

Phase 0이 성공하든 실패하든, 아래 최적화는 batch size와 무관하게 적용할 가치가 있다.

### 5.1 TF32 활성화

수정 파일:

- `Pyramidnet272/train_server.py`

`main()`에서 device 설정 직후 추가한다.

```python
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

예상 효과:

- A5000(Ampere)에서 5~15% 속도 개선 가능
- 정확도 영향은 일반적으로 작음

### 5.2 --fast_cudnn 플래그

수정 파일:

- `Pyramidnet272/train_server.py`
- `Pyramidnet272/run_seeds.py`

`set_seed()`를 다음처럼 바꾼다.

```python
def set_seed(seed: int, fast_cudnn: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = not fast_cudnn
    torch.backends.cudnn.benchmark = fast_cudnn
```

CLI:

```text
--fast_cudnn
```

설명:

- `benchmark=True`는 고정 입력 크기 CNN에서 빠른 convolution 알고리즘을 찾는 데 유리하다.
- bit-level 완전 재현성은 포기한다.
- seed 자체를 버리는 것은 아니며, 실험 결과의 통계적 재현성은 대체로 유지된다.

### 5.3 non_blocking=True 전송

수정 파일:

- `Pyramidnet272/train_server.py`

train/eval loop의 `.to(device)` 호출에 `non_blocking=True`를 붙인다.

```python
imgs = imgs.to(device, non_blocking=True)
la = la.to(device, non_blocking=True)
lb = lb.to(device, non_blocking=True)
lam = lam.to(device, non_blocking=True)
```

`pin_memory=True`와 함께 CPU to GPU 전송을 조금 더 효율적으로 만든다.

### 5.4 --channels_last 플래그

수정 파일:

- `Pyramidnet272/train_server.py`
- `Pyramidnet272/run_seeds.py`

CLI:

```text
--channels_last
```

model 생성 후:

```python
if args.channels_last:
    model = model.to(memory_format=torch.channels_last)
```

image 전송:

```python
if channels_last:
    imgs = imgs.to(
        device,
        memory_format=torch.channels_last,
        non_blocking=True,
    )
else:
    imgs = imgs.to(device, non_blocking=True)
```

주의:

- ShakeDrop, `F.pad`, SWA와의 호환성을 1 epoch로 확인한다.
- 문제가 생기면 이 옵션만 끄면 된다.

### 5.5 prefetch_factor 파라미터화

수정 파일:

- `Pyramidnet272/data/cifar100.py`
- `Pyramidnet272/train_server.py`
- `Pyramidnet272/run_seeds.py`

`get_dataloaders()` 인자:

```text
prefetch_factor=2
```

CLI:

```text
--prefetch_factor 2
```

기본값은 2로 유지하고, 서버에서 4를 실험한다.

주의:

- `num_workers=0`이면 `prefetch_factor`를 DataLoader에 넘기지 않는다.
- 너무 큰 값은 CPU RAM 사용량을 늘릴 수 있다.

### 5.6 epoch timing과 peak memory 출력

수정 파일:

- `Pyramidnet272/train_server.py`

stdout에는 각 epoch마다 다음을 출력한다.

- 순수 train seconds
- epoch wall-clock seconds
- 순수 train throughput images/sec
- CUDA max allocated GiB
- CUDA max reserved GiB

예시:

```python
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

epoch_t0 = time.time()
...
train_sec = time.time() - epoch_t0
...
epoch_sec = time.time() - epoch_t0
train_imgs_per_sec = (len(train_loader) * args.batch_size) / train_sec

if device.type == "cuda":
    max_alloc = torch.cuda.max_memory_allocated() / 1024**3
    max_reserved = torch.cuda.max_memory_reserved() / 1024**3
```

CSV는 두 종류로 분리한다.

- `log_seed{seed}.csv`: 기존 validation metric 로그. `eval_interval`에 해당하는 epoch만 기록한다.
- `perf_seed{seed}.csv`: 성능 분석 로그. 모든 epoch을 기록한다.

`perf_seed{seed}.csv` 컬럼:

```text
epoch,lr,is_swa,did_eval,train_sec,epoch_sec,train_imgs_per_sec,epoch_imgs_per_sec,cuda_max_alloc_gib,cuda_max_reserved_gib
```

이렇게 분리하면 `eval_interval=20`일 때도 순수 training throughput을 CSV로 비교할 수 있고, validation metric plot 구조는 건드리지 않는다.

---

## 6. Phase 1 벤치마크 순서

### 기준선: Phase 0만 적용

```bash
. venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_server.py \
  --seed 42 \
  --epochs 1 \
  --batch_size 256 \
  --lr 0.2 \
  --eval_interval 1 \
  --val_batch_mult 1
```

### TF32 + fast_cudnn

```bash
python train_server.py \
  --seed 42 \
  --epochs 1 \
  --batch_size 256 \
  --lr 0.2 \
  --eval_interval 1 \
  --val_batch_mult 1 \
  --fast_cudnn
```

### channels_last 추가

```bash
python train_server.py \
  --seed 42 \
  --epochs 1 \
  --batch_size 256 \
  --lr 0.2 \
  --eval_interval 1 \
  --val_batch_mult 1 \
  --fast_cudnn \
  --channels_last
```

### batch 128 비교

```bash
python train_server.py \
  --seed 42 \
  --epochs 1 \
  --batch_size 128 \
  --lr 0.1 \
  --eval_interval 1 \
  --val_batch_mult 1 \
  --fast_cudnn \
  --channels_last
```

비교 기준:

- OOM 여부
- 1 epoch 시간
- images/sec
- peak allocated GiB
- peak reserved GiB
- validation 정상 완료 여부

---

## 7. Phase 2A: batch 256 본학습

Phase 0과 Phase 1 smoke test가 통과하면 batch 256으로 본학습한다.

```bash
. venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python train_server.py \
  --seed 42 \
  --epochs 1800 \
  --batch_size 256 \
  --lr 0.2 \
  --num_workers 8 \
  --eval_interval 20 \
  --fast_cudnn \
  --channels_last \
  --val_batch_mult 1 \
  > logs/seed42.log 2>&1 &
```

운영 규칙:

- `val_batch_mult=1`을 유지한다.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 항상 설정한다.
- epoch별 peak memory를 로그로 본다.
- OOM이 재발하면 batch 128 fallback으로 전환한다.

---

## 8. Phase 2B: batch 128 fallback

Phase 0에서 batch 256이 실패하면 batch 128로 본학습한다.

```bash
. venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python train_server.py \
  --seed 42 \
  --epochs 1800 \
  --batch_size 128 \
  --lr 0.1 \
  --num_workers 8 \
  --eval_interval 20 \
  --fast_cudnn \
  --channels_last \
  --val_batch_mult 1 \
  > logs/seed42.log 2>&1 &
```

설명:

- iteration 수는 늘지만, Phase 1 최적화로 일부 보상한다.
- 안정성이 우선이면 이 경로가 기본 fallback이다.

---

## 9. Phase 3: 선택적 고급 최적화

| 항목 | 설명 | 우선순위 |
|------|------|----------|
| `--accum_steps` | batch 128에서 effective batch 256 효과를 내고 싶을 때 사용 | 낮음 |
| AMP API 현대화 | `torch.cuda.amp` warning 제거 | 낮음 |
| eval autocast | validation memory가 여전히 부족할 때 검토 | 낮음 |
| `torch.compile` | ShakeDrop/SWA/checkpoint 호환성 검증 후 실험 | 보류 |

### eval autocast에 대한 판단

현재 `evaluate()`는 FP32 forward다. validation memory가 계속 부족하면 eval에도 autocast를 적용할 수 있다.

다만 metric 계산값이 아주 미세하게 달라질 수 있으므로 Phase 0 기본 조치에는 넣지 않는다. 우선순위는 `val_batch_mult=1`, `empty_cache()`, `set_to_none=True` 다음이다.

---

## 10. 수정 대상 파일 요약

| 파일 | Phase | 변경 내용 |
|------|-------|-----------|
| `Pyramidnet272/data/cifar100.py` | 0, 1 | `val_batch_multiplier`, `prefetch_factor` 파라미터 |
| `Pyramidnet272/train_server.py` | 0, 1 | `empty_cache()`, `set_to_none=True`, `--skip_eval`, phase log, TF32, `--fast_cudnn`, `non_blocking`, `--channels_last`, `--val_batch_mult`, `--prefetch_factor`, epoch timing, peak memory |
| `Pyramidnet272/run_seeds.py` | 1 | 새 CLI 플래그 forward. 단, `--skip_eval`은 진단 전용이므로 기본 멀티시드에는 사용하지 않음 |
| `Pyramidnet272/command.txt` | 0 | `expandable_segments`, batch 256 우선 smoke test, batch 128 fallback 안내 |
| `Pyramidnet272/setup_server.sh` | 0 | 출력 안내 명령 갱신 |
| `docs/07_pyramidnet272_서버_학습_환경_구성_가이드.md` | 0 | 필요 시 실행 명령 업데이트 |
| `docs/08_pyramidnet272_서버_OOM_대응_및_속도_최적화_수정계획.md` | - | 본 계획 반영 |

---

## 11. 검증 절차

```bash
cd Pyramidnet272
. venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. 문법 확인
python -m py_compile train_server.py data/cifar100.py run_seeds.py

# 2. Phase 0 smoke test
python train_server.py --seed 42 --epochs 1 --batch_size 256 --lr 0.2 --eval_interval 1 --val_batch_mult 1

# 3. OOM 원인 분리: eval 없이 train loop만 확인
python train_server.py --seed 42 --epochs 1 --batch_size 256 --lr 0.2 --skip_eval

# 4. Phase 1 speed test
python train_server.py --seed 42 --epochs 1 --batch_size 256 --lr 0.2 --eval_interval 1 --val_batch_mult 1 --fast_cudnn --channels_last

# 5. fallback 비교
python train_server.py --seed 42 --epochs 1 --batch_size 128 --lr 0.1 --eval_interval 1 --val_batch_mult 1 --fast_cudnn --channels_last
```

최종 판단:

- batch 256이 안정적으로 1 epoch와 validation을 통과하면 Phase 2A로 간다.
- batch 256이 OOM이면 phase log와 `--skip_eval` 결과로 train/eval 원인을 분리한다.
- train loop 자체가 OOM이면 Phase 2B로 간다.
- eval 경로만 OOM이면 `val_batch_mult`, cache 정리, 필요 시 eval autocast를 추가 검토한다.

---

## 12. 결론

이번 OOM은 `batch_size=256` 학습 자체가 절대 불가능하다는 증거라기보다, 현재 서버 패키지의 validation batch 2배 설정과 FP32 evaluation, fragmentation이 겹쳐 생긴 경계선 OOM으로 보는 것이 더 정확하다.

따라서 최종 전략은 다음이다.

1. `batch_size=256`을 먼저 살려본다.
2. validation batch를 train batch와 같게 낮춘다.
3. `empty_cache()`, `set_to_none=True`, `expandable_segments`로 메모리 여유를 만든다.
4. TF32, `fast_cudnn`, `non_blocking`, `channels_last`로 속도를 측정한다.
5. 실패하면 `batch_size=128`로 fallback한다.
6. `torch.compile`은 아직 보류한다.
