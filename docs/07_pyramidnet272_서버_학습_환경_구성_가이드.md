# PyramidNet-272 서버 학습 환경 구성 가이드

**작성 기준일**: 2026-04-16  
**목적**: 로컬에서 가장 최근에 돌린 PyramidNet-272 학습 설정을 동일하게 유지한 채, 서버에서 wandb 없이 독립 실행할 수 있도록 패키징  
**참고**: `Pyramidnet110/` 서버 패키지의 폴더 구조·서버 운영 편의 기능만 참고 (학습 설정 자체는 루트 `train.py` 기준)

### 서버 환경

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA RTX A5000 (24GB) |
| CUDA | 12.2 |
| Python | 3.9.5 |
| pip | 26.0.1 |

---

## 1. 설계 원칙

> **"272 코드는 루트 기준으로 보존하고, 110에서는 서버 운영 편의 기능만 가져온다."**

- 기준 코드: 루트 `train.py` (110의 `train_server.py`가 아님)
- wandb 제거 + CSV 로그 + 서버 setup + plotting만 110에서 참고
- 272 고유 요소 반드시 유지:
  - SC-Aware Label Smoothing (`epsilon`, `intra_ratio`)
  - SC Density 메트릭 (Top-5 기반, 단순 Super-Class Accuracy와 다름)
  - `swa_start_ratio` 기반 SWA 시점 계산
  - `cutmix_prob`, `swa_lr` 등 추가 인자

---

## 2. 110 패키지의 함정 — 그대로 따라가면 터지는 부분

| # | 문제 | 원인 | 해결 |
|---|------|------|------|
| 1 | `run_seeds.py`가 서버에서 실패 | `train.py`를 호출하는데 서버엔 `train_server.py`만 있음 | 호출 대상을 `train_server.py`로 변경 |
| 2 | `setup_server.sh` 에러 | 존재하지 않는 `.env.server`를 복사 시도 | `.env.server` 복사 로직 제거 |
| 3 | plotting CSV 파싱 실패 | 110은 `val_superclass` 컬럼명, 272는 `val_sc_density` | 컬럼명 `val_sc_density`로 통일 |
| 4 | `csv.DictReader` 파싱 에러 | 110의 `#final` 행이 DictReader에서 깨짐 | `#final` 행 사용 안 함, 별도 `summary_seed{N}.txt` 저장 |
| 5 | Git에서 소스 누락 | `.gitignore`의 `data/` 규칙이 `Pyramidnet272/data/*.py`도 무시 | `.gitignore`에 예외 규칙 추가 |
| 6 | 잘못된 checkpoint 평가 | `swa_state` 무조건 우선 로드 → SWA 이전 best에서 깨짐 | `best_state` + `best_is_swa` 플래그로 저장 |

---

## 3. 최종 폴더 구조 및 파일 출처

```
Pyramidnet272/
├── train_server.py          # 새로 작성 (루트 train.py 기반, wandb 제거)
├── evaluate.py              # 루트 evaluate.py 복사 + checkpoint 로딩 수정
├── run_seeds.py             # 루트 run_seeds.py 복사 + 호출 대상 수정
├── setup_server.sh          # Pyramidnet110/ 복사 + 272용 수정
├── requirements_server.txt  # Pyramidnet110/ 기반 (python-dotenv 제거)
├── command.txt              # 새로 작성
├── models/
│   ├── __init__.py          # 루트 models/ 복사
│   └── pyramidnet.py        # 루트 models/ 복사
├── losses/
│   ├── __init__.py          # 루트 losses/ 복사 (110과 동일)
│   └── hierarchical_loss.py # 루트 losses/ 복사 (SC-Aware 포함)
└── data/
    ├── __init__.py          # 루트 data/ 복사 (110과 동일)
    ├── cifar100.py          # 루트 data/ 복사 + 서버 최적화 추가
    └── cifar-100-python/    # Pyramidnet110/data/ 복사 (178MB, 동일 데이터셋)
```

### 파일 출처 요약

| 파일 | 출처 | 방식 | 수정 필요 |
|------|------|------|-----------|
| `models/pyramidnet.py` | 루트 `models/` | 그대로 복사 | 없음 |
| `models/__init__.py` | 루트 `models/` | 그대로 복사 | 없음 |
| `losses/hierarchical_loss.py` | 루트 `losses/` | 그대로 복사 | 없음 |
| `losses/__init__.py` | 루트 `losses/` | 그대로 복사 | 없음 |
| `data/__init__.py` | 루트 `data/` | 그대로 복사 | 없음 |
| `data/cifar100.py` | 루트 `data/` | 복사 + 수정 | persistent_workers, prefetch 추가 |
| `data/cifar-100-python/` | `Pyramidnet110/data/` | 그대로 복사 | 없음 (동일 데이터셋) |
| `setup_server.sh` | `Pyramidnet110/` | 복사 + 수정 | 타이틀, .env 로직, 안내 명령어 |
| `requirements_server.txt` | `Pyramidnet110/` | 복사 + 수정 | `python-dotenv` 제거 |
| `evaluate.py` | 루트 | 복사 + 수정 | checkpoint 로딩 강화 |
| `run_seeds.py` | 루트 | 복사 + 수정 | 호출 대상, CSV 컬럼명 |
| `train_server.py` | — | 새로 작성 | 루트 `train.py` 기반 |
| `command.txt` | — | 새로 작성 | — |

### 업로드 대상 구분

| 대상 | Git | 서버 업로드 (scp) | 비고 |
|------|-----|-------------------|------|
| 소스코드 (`.py`, `.sh`, `.txt`) | O | O | Git push 후 서버에서 clone하거나 scp |
| `data/cifar-100-python/` (178MB) | X | O | Git에는 무거워서 제외, scp로 별도 업로드 |
| `logs/`, `checkpoints/` | X | X | 학습 중 서버에서 자동 생성 |
| `venv/` | X | X | `setup_server.sh`가 서버에서 생성 |

---

## 4. Step별 구현 상세

### Step 1: `train_server.py` (새로 작성)

**기반**: 루트 `train.py`

#### 제거 항목

- `import wandb`
- `_load_env()` 함수 및 `load_dotenv` 호출
- `--env-file` 인자
- `wandb.init()`, `wandb.log()`, `wandb.save()`, `wandb.watch()`, `wandb.finish()`
- `wandb_dir` 관련 로직 전체

#### 추가 항목

- `matplotlib.use("Agg")` (GUI 없는 서버 환경)
- `plot_training_curves()` 함수 내장 (110 참고, SC Density에 맞게 조정)
- 학습 완료 후 결과를 `summary_seed{N}.txt`에 저장 (`#final` CSV 행 대신)

#### CSV 헤더 통일

```
epoch,lr,train_loss,train_acc,val_top1,val_sc_density,is_swa
```

- 내부 변수명도 `val_super` → `val_sc_density` 등으로 통일
- progress bar의 `"super"` → `"sc_dens"` 또는 `"sc"`로 변경

#### 272 고유 인자 유지

```
--lam_coarse     (default: 0.4)
--epsilon        (default: 0.1)
--intra_ratio    (default: 0.5)
--cutmix_prob    (default: 0.5)
--swa_start_ratio (default: 0.85)
--swa_lr         (default: 0.0 → 0이면 lr * 0.1)
```

- `--swa_epochs`는 제거: 루트 `train.py`에서도 실제로 사용되지 않음 (`swa_start_ratio`로 계산)
- `swa_start = int(epochs * swa_start_ratio)` 방식만 유지

#### 추가 권장 인자

```
--eval_interval   (default: 20)
--plot_interval   (default: 100)
```

#### checkpoint 저장 구조 변경

```python
# best checkpoint 저장 시 — SWA 모델이면 module.state_dict()로 prefix 제거
best_state = (
    swa_model.module.state_dict()   # module. prefix, n_averaged 없이 깨끗한 state
    if epoch >= swa_start
    else model.state_dict()
)

torch.save({
    "epoch"         : epoch,
    "best_state"    : best_state,        # pyramidnet272에 바로 로드 가능
    "best_is_swa"   : (epoch >= swa_start),
    "val_top1"      : val_top1,
    "val_sc_density": val_sc_density,
    "seed"          : args.seed,
    "args"          : vars(args),
}, ckpt_path)
```

주의: `AveragedModel.state_dict()`를 그대로 저장하면 `module.` prefix와 `n_averaged`가 포함되어, `pyramidnet272()` 모델에 직접 로드 시 key mismatch로 실패하거나 `strict=False`에서 조용히 잘못 로드됨. 반드시 `.module.state_dict()`로 깨끗한 state를 저장할 것.

기존처럼 `model_state`와 `swa_state`를 동시에 저장하면, SWA 시작 전에 best가 갱신된 경우 `swa_state`가 초기 상태라 로드 시 잘못된 결과가 나옴.

#### AMP 호환성

루트 `train.py`는 새 API(`torch.amp.GradScaler('cuda')`)를 사용하는데, 이는 PyTorch 2.4+ 전용.  
서버 환경에 따라 구 버전이 설치될 수 있으므로 둘 중 하나 선택:

- **방법 A**: `torch.cuda.amp.GradScaler()` (구 API, 모든 PyTorch 2.x에서 동작, deprecation 경고만 뜸)
- **방법 B**: 새 API 유지하되 `requirements_server.txt`에서 `torch>=2.4.0`으로 올림

→ 서버 범용성을 위해 **방법 A 권장**

### Step 2: `evaluate.py` (루트 복사 + 수정)

**원본**: 루트 `evaluate.py`

#### 변경점

- 메트릭 명칭: 출력에서 `SC Density`로 통일
- `--batch_size`, `--num_workers` 인자 추가

#### checkpoint 로딩 우선순위

```
1. best_state가 있으면 → best_state 로드 (가장 안전, module. prefix 이미 제거됨)
2. raw state_dict이면 → 그대로 로드
3. model_state가 있으면 → model_state 로드
4. swa_state가 있으면 → module. prefix + n_averaged 제거 후 로드 (fallback)
```

모든 경로에서 공통으로 `module.` prefix 제거 + `n_averaged` 키 제거를 거치도록 구현.  
기존 `swa_state` 무조건 우선 로드 방식에서 개선.

### Step 3: `run_seeds.py` (루트 복사 + 수정)

**원본**: 루트 `run_seeds.py`

#### 변경점

- `train.py` → `train_server.py` 호출
- `--env-file` 관련 로직 제거
- CSV 읽기: `val_superclass` → `val_sc_density`
- 스크립트 위치 기준 실행: `Path(__file__).resolve().parent`로 cwd 고정
- pass-through 인자 지원: `--ckpt_dir`, `--data_root`, `--num_workers`, `--lr`, `--eval_interval`
- CSV 파싱: 숫자 epoch 행만 사용 (`#`으로 시작하는 행 스킵)

#### 최종 출력 형식

```
Seed 42 → Top-1: xx.xx%  SC Density: xx.xx%
Seed 0  → Top-1: xx.xx%  SC Density: xx.xx%
Seed 1  → Top-1: xx.xx%  SC Density: xx.xx%
Mean Top-1      : xx.xx% ± xx.xx%
Mean SC Density : xx.xx% ± xx.xx%
```

### Step 4: `data/cifar100.py` (루트 복사 + 서버 최적화)

**원본**: 루트 `data/cifar100.py`

#### augmentation 판단

| 선택지 | 장점 | 단점 |
|--------|------|------|
| AutoAugment 유지 (권장) | 기존 272 실험 조건 동일, 재현 가능 | 서버에서 약간 느릴 수 있음 |
| RandAugment로 교체 | 서버 속도 최적화 (110과 동일) | 272 실험 조건이 바뀜, 결과 비교 불가 |

→ **AutoAugment 유지 권장**. 필요시 나중에 `--augment autoaugment|randaugment` 옵션으로 분리.

#### 서버 최적화 추가

```python
use_persistent = num_workers > 0

train_loader = DataLoader(
    ...,
    persistent_workers=use_persistent,
    prefetch_factor=2 if use_persistent else None,
    pin_memory=torch.cuda.is_available(),
)
```

#### 3-seed 재현성 강화 (선택)

`worker_init_fn`과 `generator`를 추가하면 멀티프로세스 DataLoader에서도 seed 고정 가능:

```python
g = torch.Generator()
g.manual_seed(seed)

def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)

train_loader = DataLoader(
    ...,
    worker_init_fn=worker_init_fn,
    generator=g,
)
```

### Step 5: `setup_server.sh` (Pyramidnet110 복사 + 수정)

**원본**: `Pyramidnet110/setup_server.sh`

#### 변경점

- 타이틀: `PyramidNet-272 CIFAR-100`으로 변경
- `python3 -m venv venv`로 패키지 전용 가상환경 생성
- `source venv/bin/activate` 후 `python -m pip ...`로 venv 내부에만 설치
- `.env.server` 복사 로직 **제거**
- 안내 명령어를 272 기준으로 변경:

```bash
source venv/bin/activate

nohup python train_server.py --seed 42 --epochs 1800 > logs/seed42.log 2>&1 &
nohup python train_server.py --seed 0  --epochs 1800 > logs/seed0.log 2>&1 &
nohup python train_server.py --seed 1  --epochs 1800 > logs/seed1.log 2>&1 &

tail -f logs/seed42.log
python evaluate.py --ckpt checkpoints/best_seed42.pth
```

### Step 6: `requirements_server.txt` (Pyramidnet110 기반, python-dotenv 제거)

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
tqdm>=4.65.0
matplotlib>=3.7.0
```

- Pyramidnet110의 `requirements_server.txt`에서 `python-dotenv` 한 줄만 제거
- `_load_env`와 `--env-file`을 제거했으므로 불필요
- `torch.amp.GradScaler` 구 API 사용 시 `torch>=2.0.0` 유지 가능
- 새 API 사용 시 `torch>=2.4.0`으로 올려야 함

### Step 7: `command.txt` (새로 작성)

```bash
# === 환경 세팅 ===
bash setup_server.sh

# === 가상환경 활성화 ===
source venv/bin/activate

# === 1 epoch 시간 측정 (batch_size별) ===
python train_server.py --seed 42 --epochs 1 --batch_size 256 --lr 0.2 --eval_interval 1
python train_server.py --seed 42 --epochs 1 --batch_size 512 --lr 0.4 --eval_interval 1

# === 단일 시드 학습 (epoch·batch_size는 측정 후 결정) ===
nohup python train_server.py --seed 42 --batch_size 256 --lr 0.2 > logs/seed42.log 2>&1 &

# === 멀티 시드 학습 ===
python run_seeds.py --batch_size 256 --lr 0.2

# === 로그 모니터링 ===
tail -f logs/seed42.log

# === 평가 ===
python evaluate.py --ckpt checkpoints/best_seed42.pth
python evaluate.py --ckpt checkpoints/swa_final_seed42.pth

# === 프로세스 관리 ===
ps aux | grep train_server.py
kill [PID]
```

> batch_size와 lr의 linear scaling: 128→0.1, 256→0.2, 512→0.4  
> 512는 RTX A5000 24GB에서 OOM 가능성 있으므로 256 우선 권장

### Step 8: `.gitignore` 수정

`Pyramidnet272/`를 Git에 올리려면 현재 `data/` 규칙 때문에 소스가 누락됨.  
루트 `.gitignore`에 추가:

```gitignore
# 서버 학습 패키지 소스는 유지
!Pyramidnet272/
!Pyramidnet272/data/
!Pyramidnet272/data/*.py

# 서버 생성물은 제외
Pyramidnet272/data/cifar-100-python/
Pyramidnet272/checkpoints/
Pyramidnet272/logs/
Pyramidnet272/venv/
```

---

## 5. 검증 절차

```bash
cd Pyramidnet272
source venv/bin/activate

# 1. import/문법 확인
python -m py_compile train_server.py evaluate.py run_seeds.py

# 2. 스모크 학습 (2 epoch)
python train_server.py --epochs 2 --batch_size 16 --num_workers 0 --eval_interval 1

# 3. 생성물 확인
ls checkpoints/log_seed42.csv
ls checkpoints/best_seed42.pth
ls checkpoints/swa_final_seed42.pth

# 4. 평가 확인
python evaluate.py --ckpt checkpoints/best_seed42.pth

# 5. 멀티시드 dry run
python run_seeds.py --epochs 1 --batch_size 16 --num_workers 0 --eval_interval 1
```

---

## 6. 메인 프로젝트 ↔ 서버 패키지 대조표

| 항목 | 루트 (`train.py`) | 서버 (`train_server.py`) |
|------|-------------------|--------------------------|
| 로깅 | wandb | CSV + summary txt |
| 그래프 | 별도 `plot_log.py` | `train_server.py` 내장 |
| 환경설정 | `.env` + `--env-file` | CLI 인자만 |
| AMP API | `torch.amp` (새) | `torch.cuda.amp` (구, 호환성) |
| checkpoint | `model_state` + `swa_state` | `best_state` + `best_is_swa` |
| augmentation | AutoAugment | AutoAugment (동일) |
| DataLoader | 기본 | persistent_workers + prefetch |
| 의존성 | wandb 포함 43개 | 5개 |
