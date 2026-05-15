---
title: Adam optimizer
layer: 2
date: 2026-05-12
---

# Adam (Adaptive Moment Estimation)

> SGD 의 발전형 optimizer. parameter 별로 learning rate 가 자동 조절됨. layer 2 — 누군가 정한 방식 (Kingma & Ba, 2014).

이 노트는 **SGD → SGD+momentum → RMSProp → Adam** 순으로 차근차근 쌓아 올림.

---

## 0. 학습이 뭘 하는가 (배경)

모델은 parameter $\theta$ (예: cloth_logit) 와 loss $L(\theta)$ 가 있을 때, $L$ 을 줄이는 $\theta$ 를 찾는 게 목표.

$\theta$ 가 1개 scalar 라고 가정. loss landscape 가 산골짜기처럼 생겼다고 상상:

```
L(θ)
 │     ╱╲     ← 산
 │    ╱  ╲╱╲
 │___╱      ╲___ ← 목표 (계곡 바닥)
 │              θ
```

매 step 에서 *어디로 어떻게 이동*할지 결정하는 게 optimizer.

## 1. SGD (Stochastic Gradient Descent) — 기본 형태

$$
\theta_{t+1} = \theta_t - \text{lr} \cdot g_t \qquad \text{where} \quad g_t = \nabla L(\theta_t)
$$

- $g_t$ = 현재 위치의 *기울기* (어느 방향이 위로 향하는지)
- $-g_t$ = 아래로 향하는 방향
- lr (learning rate) = 한 발자국 크기

### 예시: 1D parabola

$L(\theta) = \theta^2$. 최소값은 $\theta = 0$. lr = 0.1.

| step | $\theta_t$ | $g_t = 2\theta_t$ | $\theta_{t+1} = \theta_t - 0.1 g_t$ |
|---|---|---|---|
| 0 | 5.0 | 10.0 | 4.0 |
| 1 | 4.0 | 8.0 | 3.2 |
| 2 | 3.2 | 6.4 | 2.56 |
| 3 | 2.56 | 5.12 | 2.05 |
| ... | ... | ... | ... |
| 50 | ≈ 0.0007 | | |

수렴은 하는데 **lr 가 너무 작으면 느림**, **너무 크면 진동/발산**.

### SGD 의 약점

- **noisy gradient** — mini-batch 마다 $g_t$ 가 흔들림 → 진로가 지그재그
- **고정 lr** — 어떤 parameter 는 큰 step 필요, 어떤 건 작은 step 필요한데 같은 lr 사용 → 일부는 느리고 일부는 발산

## 2. Moment 의 정의 — gradient 의 통계량

확률·통계의 *moment* 와 같은 개념. random variable $X$ 의 k차 moment:
- 1차: $E[X]$ = **평균** (mean)
- 2차 (raw): $E[X^2]$ = **변동성** (uncentered variance)

Adam 에서 $X$ = 매 step 의 gradient $g_t$. 정의:
- 1차 모멘트 $m_t \approx E[g_t]$ — **gradient 의 평균** (즉 일관된 방향)
- 2차 모멘트 $v_t \approx E[g_t^2]$ — **gradient 의 제곱 평균** (얼마나 크게 흔들리는지)

매 step 정확히 평균 내면 비싸니 **지수이동평균 (EMA)** 으로 추정:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
\end{aligned}
$$

- $\beta_1 = 0.9$ → 직전 평균 90% 유지, 새 sample 10% 반영 → 약 10 step 평균
- $\beta_2 = 0.999$ → 약 1000 step 평균 (더 안정적)

### Moment 예시 (gradient 가 지그재그 흔들릴 때)

real gradient $g_t = [+2, -2, +2, -2, +2, -2, ...]$ (완전 지그재그). $\beta_1 = 0.9$, $m_0 = 0$.

| t | $g_t$ | $m_t = 0.9 m_{t-1} + 0.1 g_t$ |
|---|---|---|
| 1 | +2 | 0.2 |
| 2 | -2 | 0.18 − 0.2 = -0.02 |
| 3 | +2 | -0.018 + 0.2 = 0.182 |
| 4 | -2 | 0.164 − 0.2 = -0.036 |
| ... | | |
| ∞ | 진동 | $m \to 0$ (지그재그라 평균 0) |

→ noisy gradient 가 평균으로 cancel out. **흔들림 제거** 효과.

real gradient $g_t = [+2, +2, +2, +2, ...]$ (일관된 방향). $m_0 = 0$.

| t | $g_t$ | $m_t$ |
|---|---|---|
| 1 | 2 | 0.2 |
| 2 | 2 | 0.38 |
| 3 | 2 | 0.542 |
| ... | | |
| ∞ | | $m \to 2$ |

→ 일관 방향이면 **누적 가속**. = momentum.

## 3. SGD+momentum — 1차 moment 만 사용

$$
\theta_{t+1} = \theta_t - \text{lr} \cdot m_t
$$

- 지그재그 gradient → $m_t$ 작아짐 → 천천히 이동
- 일관 gradient → $m_t$ 커짐 → 빠르게 이동 (탄력)
- 비유: 공이 굴러가는 듯한 관성. *모멘텀*

## 4. RMSProp — 2차 moment 만 사용

$$
\theta_{t+1} = \theta_t - \text{lr} \cdot \frac{g_t}{\sqrt{v_t} + \epsilon}
$$

- gradient 가 크게 흔들리는 parameter: $v_t$ 크다 → 분모 크다 → step 작게 (안전)
- gradient 가 작고 안정적: $v_t$ 작다 → 분모 작다 → step 크게 (가속)

= **per-parameter adaptive learning rate**.

## 5. Adam — 둘 다 사용

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t & \text{(1차 모멘트)} \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 & \text{(2차 모멘트)}
\end{aligned}
$$

**Bias correction**: $m_0 = v_0 = 0$ 으로 시작하니 초기 step 에선 $m_t$, $v_t$ 가 작게 편향됨. 이를 보정:

$$
\hat m_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat v_t = \frac{v_t}{1 - \beta_2^t}
$$

t 가 클수록 $\beta_1^t \to 0$ 이므로 $\hat m_t \approx m_t$. t=1, β₁=0.9 면 $1 - 0.9 = 0.1$ → $\hat m_1 = m_1 / 0.1 = g_1$ → 초기 step 에서도 제대로 된 크기.

**최종 update**:

$$
\boxed{\theta_{t+1} = \theta_t - \text{lr} \cdot \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}}
$$

## 6. 종합 예시 — $L(\theta) = \theta^2$ 다시 풀기

lr=0.1, β₁=0.9, β₂=0.999, ε=1e-8. $\theta_0 = 5$.

| t | $\theta_t$ | $g_t = 2\theta_t$ | $m_t$ | $v_t$ | $\hat m_t$ | $\hat v_t$ | update step | $\theta_{t+1}$ |
|---|---|---|---|---|---|---|---|---|
| 1 | 5.000 | 10.000 | 1.000 | 0.100 | 10.000 | 100.000 | -0.100·(10/10) = -0.100 | 4.900 |
| 2 | 4.900 | 9.800 | 1.880 | 0.196 | 9.895 | 98.099 | -0.100·(9.895/9.904) = -0.0999 | 4.800 |
| 3 | 4.800 | 9.600 | 2.652 | 0.288 | 9.788 | 96.213 | -0.0999 | 4.700 |
| ... | | | | | | | | |

SGD 와 비교: SGD 는 step 1 에서 5 → 4 (− 1.0), Adam 은 5 → 4.9 (− 0.1).

**왜 SGD 보다 느려 보이나?** Adam 은 *대부분 1 단위 step* 으로 일정하게 이동. SGD 는 처음엔 빠르지만 끝엔 더 빨리 수렴 못 함 (gradient 작아져서 step 도 작아짐). Adam 은 lr 가 *self-normalizing* 되어 처음부터 끝까지 균일한 속도.

## 7. Adam vs SGD 한 줄 정리

```
SGD          : θ ← θ − lr · g                  (방향만 보고 일정 lr 로 이동)
SGD+momentum : θ ← θ − lr · m_t                (momentum 누적: 흔들림 제거 + 가속)
RMSProp      : θ ← θ − lr · g / √v_t           (per-param adaptive lr)
Adam         : θ ← θ − lr · m̂_t / √v̂_t         (momentum + adaptive lr 결합)
```

비유:
- **SGD** = 매번 발이 닿는 방향으로 똑같은 걸음 폭으로 이동
- **SGD+momentum** = 관성 있는 공처럼 굴러가기 (지그재그 cancel out)
- **RMSProp** = 미끄러운 곳은 조심해서 작게, 안정한 곳은 크게 걸음
- **Adam** = 위 두 가지 다 — 관성 + 지형 적응

## 8. 우리 연구에서

- **4DGS 학습** (`xyz, scale, rot, opacity, SH, deformation MLP`) — Adam, lr 은 parameter group 별 다름 (`feature_lr = 2.5e-3`, `rotation_lr = 1e-3`, ...)
- **[[../results/exp010_per_gaussian_projection|exp010]] cloth_logit fine-tune** — Adam, lr=0.05, 2000 step. 1D scalar per Gaussian 의 단순 문제라 매우 빠르게 수렴.

## 9. 깰 수 있는 부분

- $\beta_1=0.9, \beta_2=0.999$ 고정값 — 작은 모델 / 큰 batch 등에서 다른 값
- Adam 의 weight decay 처리 약점 → **AdamW** (Loshchilov 2019) 가 분리해서 수정
- 메모리 부담 ($m_t, v_t$ 가 parameter 마다 2 배) → **Lion** (Chen 2023) 등은 $m_t$ 만 사용
- adaptive lr 가 generalization 해친다는 비판 → SGD 가 vision task 에선 여전히 경쟁력
