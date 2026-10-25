# -*- coding: utf-8 -*-
"""Part 2: Section 10 ~ end."""

CELLS = []
A = CELLS.append


def md(s):
    A(("md", s))


def code(s):
    A(("code", s))


# ------------------------------------------------------------------ Section 10
md(r"""## Section 10. One BO Iteration

지금 가진 데이터에서 **단 한 번의 BO iteration**을 실행한다. 순서는 정확히 다음과 같다.

```text
1. GP fit            (posterior mean/var 계산)
2. EI 계산
3. EI 최대 위치 선택
4. 실제 objective 평가   <-- 여기서만 비싼 비용이 든다
5. training data에 추가
```

BO의 계산 비용은 1~3단계(싸다)에 있고, 4단계(비싸다)는 오직 한 번만 일어난다.
**싼 계산을 많이 써서 비싼 평가를 아끼는 것**이 BO의 본질이다.""")

code(r'''X_iter = X_train.copy()
y_iter = y_train.copy()

# 1~2. GP fit + EI
mu_i, var_i = gp_posterior(X_iter, y_iter, x_plot, length_scale=1.0, sigma_n=1e-6)
std_i = np.sqrt(var_i)
ei_i = expected_improvement(mu_i, std_i, best_y=float(np.min(y_iter)), xi=0.01)

# 3. next candidate
x_next_i = float(x_plot[int(np.argmax(ei_i))])

# 4. 실제 objective 평가 (유일하게 비싼 단계)
y_next_i = float(objective(x_next_i))

# 5. 데이터 추가
X_iter = np.append(X_iter, x_next_i)
y_iter = np.append(y_iter, y_next_i)

print(f"Current best      = {np.min(y_iter[:-1]):.6f}")
print(f"Selected x_next   = {x_next_i:.6f}")
print(f"EI(x_next)        = {ei_i.max():.6f}")
print(f"Observed f(x_next)= {y_next_i:.6f}")
print(f"New best          = {np.min(y_iter):.6f}")''')

# ------------------------------------------------------------------ Section 11
md(r"""## Section 11. Full Bayesian Optimization Loop

이제 한 iteration을 반복문으로 감싼다. 초기 $n_{\text{initial}}$개 점은 random sampling으로 채우고(어디서 시작할지 모르므로),
그 이후부터는 EI로 다음 점을 고른다.

매 iteration마다 다음을 history에 저장한다. 나중에 진행 과정을 그림으로 되돌려 보기 위해서다.

```python
{
    "iteration": ...,
    "X": ..., "y": ...,          # 이 시점까지의 관측
    "mu": ..., "sigma": ...,     # grid 위 posterior
    "ei": ...,                   # grid 위 EI
    "x_next": ..., "y_next": ...,  # 이번에 고르고 평가한 점
}
```""")

code(r'''def bayesian_optimization(objective, bounds, n_initial=3, n_iterations=12,
                           length_scale=1.0, sigma_f=1.0, sigma_n=1e-6,
                           xi=0.01, grid_size=400, seed=0, verbose=True):
    """1D Bayesian Optimization with a hand-written GP and EI.

    Returns
    -------
    X, y : (n_initial + n_iterations,) arrays, 평가한 모든 점
    history : list of dict, 매 iteration의 상태
    """
    rng = np.random.default_rng(seed)
    low, high = bounds
    grid = np.linspace(low, high, grid_size)

    # 초기 설계: random sampling
    X = rng.uniform(low, high, size=n_initial)
    y = objective(X)

    history = []
    n_fallback = 0
    for it in range(n_iterations):
        mu, var = gp_posterior(X, y, grid, length_scale, sigma_f, sigma_n)
        sigma = np.sqrt(var)
        ei = expected_improvement(mu, sigma, best_y=float(np.min(y)), xi=xi)

        if ei.max() > 1e-12:
            x_next = float(grid[int(np.argmax(ei))])
        else:
            # 모든 후보의 EI가 0이면(=더 개선할 여지가 없다고 믿으면)
            # argmax는 index 0을 돌려주므로 같은 점을 반복하게 된다.
            # 이때는 불확실성이 가장 큰 곳으로 되돌아간다.
            x_next = float(grid[int(np.argmax(sigma))])
            n_fallback += 1

        y_next = float(objective(np.array([x_next]))[0])

        record = {
            "iteration": it,
            "X": X.copy(), "y": y.copy(),
            "mu": mu.copy(), "sigma": sigma.copy(), "ei": ei.copy(),
            "x_next": x_next, "y_next": y_next,
            "best_y": float(np.min(y)),
        }
        history.append(record)

        X = np.append(X, x_next)
        y = np.append(y, y_next)

        if verbose:
            flag = "  (EI=0 -> max sigma fallback)" if ei.max() <= 1e-12 else ""
            print(f"iter {it:2d} | best={record['best_y']:.4f} "
                  f"| x_next={x_next:.3f} | EI={ei.max():.5f} "
                  f"| f(x_next)={y_next:.4f} | new best={np.min(y):.4f}{flag}")

    if verbose and n_fallback:
        print(f"\n[참고] EI가 전 구간에서 0이 되어 max-sigma fallback으로 선택된 횟수: {n_fallback}")

    return X, y, history, grid''')

code(r'''X_bo, y_bo, history, grid_bo = bayesian_optimization(
    objective, BOUNDS, n_initial=3, n_iterations=12, seed=0)

print()
print(f"total evaluations = {len(X_bo)}")
print(f"best x = {X_bo[int(np.argmin(y_bo))]:.4f}, best f = {np.min(y_bo):.4f}")
print(f"regret = {np.min(y_bo) - GLOBAL_F:.4f}")''')

md(r"""출력 로그를 보면 흥미로운 점이 있다. 초반에는 EI가 큰(불확실한) 곳을 넓게 훑고,
어느 순간부터 best 근처를 좁게 파고든다. 이 **자동적인 exploration → exploitation 전환**이 EI의 힘이다.

### 주의: EI가 전 구간에서 0이 되는 경우

초기 관측이 3개뿐인 1D 문제에서는 `xi=0.01`일 때 **모든 후보의 EI가 0**이 되는 순간이 실제로 생긴다.
(`f_best`가 현재 관측 중 최솟값이라, GP가 "어디서도 그보다 좋을 것 같지 않다"고 믿는 상태)

이때 `np.argmax`는 조용히 **index 0**(= domain 왼쪽 끝)을 돌려준다. 즉 아무 정보도 없이 같은 점을 반복해서 평가하게 된다.
로그에서 `EI=0 -> max sigma fallback` 이 붙은 줄이 그런 경우다.
그래서 위 구현에는 예외 처리를 넣었다: EI가 모두 0이면 **불확실성이 가장 큰 곳**으로 돌아간다.

> 교육 포인트: 논문의 수식은 맞지만, 코드로 옮길 때는 이런 **degenerate case**를 반드시 처리해야 한다.

### 학습 확인 질문

1. 각 iteration에서 실제 objective를 몇 번 평가하는가? 그 이유는 무엇인가?
2. `n_initial`을 늘리면 초기 정보는 좋아지지만 무엇을 잃는가?
3. EI가 전 구간 0일 때 `argmax`가 index 0을 돌려주는 것을 그대로 두면 어떤 문제가 생기는가?""")

# ------------------------------------------------------------------ Section 12
md(r"""## Section 12. BO 진행 과정 시각화

iteration 0, 2, 5, 10의 상태를 그림으로 되돌려 본다. 각 그림에서 확인할 것:

- objective 곡선
- GP posterior mean
- uncertainty band (관측점 주변에서 좁아진다)
- 관측점
- 이번 iteration에서 선택된 다음 점

목표는 **BO가 시간이 지나며 유망한 영역에 집중하는 것**을 눈으로 보는 것이다.""")

code(r'''def plot_bo_iteration(record, grid, title_prefix="", fname=None, show=True):
    """history의 한 record를 그림으로 그린다."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(grid, objective(grid), linewidth=1.1, alpha=0.5, label="true objective")
    ax.plot(grid, record["mu"], linewidth=2.0, label="GP posterior mean")
    ax.fill_between(grid, record["mu"] - 1.96 * record["sigma"],
                    record["mu"] + 1.96 * record["sigma"], alpha=0.2,
                    label="95% interval")
    ax.plot(record["X"], record["y"], "o", markersize=7, label="observations")
    if record["x_next"] is not None:
        ax.axvline(record["x_next"], linestyle="--", linewidth=1.4,
                   label=f"selected x = {record['x_next']:.2f} (EI max)")
    ax.set_title(f"{title_prefix}iteration {record['iteration']} "
                 f"- evaluations: {len(record['X'])} "
                 f"- best: {np.min(record['y']):.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(loc="upper left", fontsize=8)
    if fname:
        fig.savefig(os.path.join(FIG_DIR, fname), bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax''')

code(r'''for it in [0, 2, 5, 10]:
    plot_bo_iteration(history[it], grid_bo,
                      fname=f"10_bo_iter_{it:02d}.png")''')

md(r"""그림을 순서대로 보면:

- **iteration 0**: 아직 점이 3개뿐이라 신뢰구간이 넓고, 선택점은 정보가 부족한 영역에 있다.
- **iteration 2**: 관측이 늘면서 mean이 실제 곡선을 따라가기 시작한다.
- **iteration 5**: 좋은 영역(local minimum 근처)에 점이 몰리기 시작한다.
- **iteration 10**: 거의 global minimum 주변에 집중된다.

또 하나 눈여겨볼 것: 관측점이 지나치게 겹치지 않는다. EI가 0이 되는 이미 아는 곳은 선택되지 않기 때문이다.
**같은 점을 두 번 평가하는 것은 낭비**이고, EI는 이를 구조적으로 피한다.""")

# ------------------------------------------------------------------ Section 13
md(r"""## Section 13. Random Search vs Bayesian Optimization

공정한 비교를 위해 **평가 횟수를 맞춘다**. BO는 `n_initial=3` + `n_iterations=12` = 15회, Random Search도 15회.
BO는 최초 포인트 3개를 random으로 뽑으므로, **처음 3회는 두 방법이 사실상 동일**하다. 이후 12회가 차이다.

비교 지표는 `best_so_far` (지금까지 찾은 최소값)이다.

```python
best_so_far = np.minimum.accumulate(y)
```""")

code(r'''X_rs15, y_rs15 = random_search(objective, BOUNDS, n_evals=15, seed=0)

best_rs = np.minimum.accumulate(y_rs15)
best_bo = np.minimum.accumulate(y_bo)
evals = np.arange(1, len(best_rs) + 1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(evals, best_rs, marker="o", linewidth=1.8, label="Random Search")
ax.plot(evals, best_bo, marker="s", linewidth=1.8, label="Bayesian Optimization")
ax.axhline(GLOBAL_F, linestyle="--", linewidth=1.4, label="true global optimum")
ax.set_title("Best objective found so far (equal budget of 15 evaluations)")
ax.set_xlabel("Number of objective evaluations")
ax.set_ylabel("Best objective found so far")
ax.legend()
fig.savefig(os.path.join(FIG_DIR, "11_rs_vs_bo.png"), bbox_inches="tight")
plt.show()

print(f"Random Search      final best = {best_rs[-1]:.4f}  (regret {best_rs[-1] - GLOBAL_F:.4f})")
print(f"Bayesian Optim.    final best = {best_bo[-1]:.4f}  (regret {best_bo[-1] - GLOBAL_F:.4f})")''')

md(r"""여기서 정직하게 짚어야 할 점: **한 번의 seed 결과를 절대적인 우열로 해석하면 안 된다.**
Random Search가 15번 중 우연히 좋은 점을 뽑으면 BO를 이길 수도 있다. 그래서 다음 섹션에서 반복 실험을 한다.""")

# ------------------------------------------------------------------ Section 14
md(r"""## Section 14. 반복 실험 (30 seeds)

운의 영향을 제거하기 위해 seed를 바꿔 30번 반복하고, 각 방법의 **regret**을 비교한다.

$$\text{regret} = f_{\text{best-found}} - f_{\text{global}}$$

regret이 0에 가까울수록 좋다. 최종 regret의 mean / std / median을 보고, boxplot으로 분포를 확인한다.""")

code(r'''N_SEEDS = 30
N_INIT = 3
N_ITER = 12
BUDGET = N_INIT + N_ITER  # 15

regret_rs = np.zeros(N_SEEDS)
regret_bo = np.zeros(N_SEEDS)

for s in range(N_SEEDS):
    _, y_s = random_search(objective, BOUNDS, n_evals=BUDGET, seed=s)
    _, y_b, _, _ = bayesian_optimization(
        objective, BOUNDS, n_initial=N_INIT, n_iterations=N_ITER,
        seed=s, verbose=False)
    regret_rs[s] = np.min(y_s) - GLOBAL_F
    regret_bo[s] = np.min(y_b) - GLOBAL_F

print(f"{'method':<20}{'mean':>10}{'std':>10}{'median':>10}{'max':>10}")
for name, r in [("Random Search", regret_rs), ("Bayesian Optim.", regret_bo)]:
    print(f"{name:<20}{r.mean():>10.4f}{r.std():>10.4f}{np.median(r):>10.4f}{r.max():>10.4f}")''')

code(r'''fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot([regret_rs, regret_bo],
                tick_labels=["Random Search", "Bayesian Optimization"],
                widths=0.5, showmeans=True)
ax.set_title(f"Final regret over {N_SEEDS} random seeds ({BUDGET} evaluations each)")
ax.set_ylabel("regret = f_best_found - f_global  (lower is better)")
ax.set_xlabel("method")
fig.savefig(os.path.join(FIG_DIR, "12_regret_boxplot_rs_vs_bo.png"), bbox_inches="tight")
plt.show()

wins_bo = int(np.sum(regret_bo < regret_rs - 1e-12))
ties = int(np.sum(np.abs(regret_bo - regret_rs) <= 1e-12))
print(f"BO strictly better: {wins_bo}/{N_SEEDS}  | ties: {ties}  "
      f"| RS better: {N_SEEDS - wins_bo - ties}/{N_SEEDS}")''')

md(r"""**해석 포인트**

- mean과 median을 함께 봐야 한다. BO는 대부분의 seed에서 안정적으로 낮은 regret을 내는 경향이 있지만,
  RS가 우연히 아주 좋은 점을 뽑으면 RS의 최솟값이 튈 수 있다.
- std가 크다는 것은 "운에 따라 편차가 크다"는 뜻이다. 실무에서는 편차가 작은 쪽이 예측 가능성이 높다.
- 평가 예산이 15회로 매우 작다는 점을 기억하자. 예산이 커지면 RS도 결국 좋은 해를 찾고, 차이는 줄어든다.

### 학습 확인 질문

1. mean regret과 median regret이 다르게 나오는 이유는 무엇인가?
2. 이 실험에서 BO의 우위가 압도적이지 않다면, 그 원인을 두 가지 이상 들어 보라.""")

# ------------------------------------------------------------------ Section 15
md(r"""## Section 15. CMA-ES 소개

CMA-ES(Covariance Matrix Adaptation Evolution Strategy)는 Evolutionary Algorithm 계열의 대표적인 방법이다.
핵심 아이디어는 **좋은 해가 존재하는 search distribution 자체를 학습**하는 것이다.

$$x \sim \mathcal{N}(m, \sigma^2 C)$$

- $m$: 지금까지 좋은 해들이 모인 **중심**
- $\sigma$: 전체적인 **탐색 스케일**
- $C$: 탐색의 **방향과 상관관계**를 담은 covariance matrix

한 세대(generation)마다 population을 샘플링하고, 성능이 좋은 일부만 선택해서 $m, \sigma, C$를 업데이트한다.
$C$가 학습된다는 점이 단순 hill climbing과의 결정적 차이다.

### BO와의 철학적 차이

| | BO | CMA-ES |
|---|---|---|
| 무엇을 모델링하는가 | objective function $f$의 surrogate | 좋은 해가 있을 search distribution |
| uncertainty | $\sigma(x)$를 명시적으로 사용 | population의 spread로 암묵적으로 존재 |
| 평가당 정보 | 1점을 아주 비싸게 여김 | population 전체를 한 세대로 처리 |
| 병렬화 | 어색함 (다음 점이 이전 점에 의존) | 매우 자연스러움 (population을 동시에 평가) |

**BO**: 적은 수의 아주 비싼 평가, 낮은 차원, 매끄러운 objective에서 강하다.
**CMA-ES**: 중간~높은 차원, 노이즈가 크거나 매끄럽지 않은 objective, 병렬 평가 환경에서 강하다.""")

# ------------------------------------------------------------------ Section 16
md(r"""## Section 16. CMA-ES 실험

`cma` 패키지로 같은 objective를 최적화한다.

주의: CMA-ES는 **population 단위**로 평가하므로 평가 횟수를 정확히 15로 맞출 수 없다.
`cma`의 `maxfevals`는 세대 경계에서만 확인되므로, `popsize=5, maxfevals=15`는 3세대(15회)가 아니라
**4세대(20회)** 로 끝날 수 있다. 이건 버그가 아니라 population 기반 알고리즘의 구조적 성질이다.

여기서는 실제로 사용한 평가 횟수를 그대로 보여주고, 비교할 때 이 차이를 명시한다.
(즉 아래 비교에서 CMA-ES는 약 33% 더 많은 예산을 쓴 셈이다. 결과 해석 시 감안해야 한다.)

공정성을 위해 평가 궤적(best-so-far)을 **실제 평가 횟수** 기준으로 그린다.""")

code(r'''import cma


def run_cmaes(objective, bounds, max_evals=15, popsize=5, sigma0=2.0, seed=0, verbose=False):
    """CMA-ES를 실행하고 평가 이력을 순서대로 기록한다.

    Notes
    -----
    `cma`의 'seed' 옵션은 'randn'이 np.random.randn일 때만 동작한다.
    bounds를 주면 내부 sampler가 다른 randn을 쓰므로 seed가 조용히 무시되어
    결과가 전역 RNG 상태에 의존하게 된다(재현 불가). 그래서 여기서는
    명시적인 RandomState를 randn으로 주입한다.

    Returns
    -------
    X : 평가한 x들
    y : 대응하는 f(x)들
    """
    low, high = bounds
    X_log, y_log = [], []

    rng = np.random.RandomState(seed)

    def randn(*size):
        """cma가 요구하는 randn(size) / randn(n, m) 시그니처를 맞춘다."""
        return rng.standard_normal(size[0] if len(size) == 1 else size)

    def wrapped(x):
        xi = float(np.clip(np.asarray(x).ravel()[0], low, high))
        yi = float(objective(np.array([xi]))[0])
        X_log.append(xi)
        y_log.append(yi)
        return yi

    x0 = 0.5 * (low + high)
    es = cma.CMAEvolutionStrategy(
        [x0], sigma0,
        {"maxfevals": max_evals, "popsize": popsize, "randn": randn,
         "verbose": -9, "bounds": [low, high]})
    while not es.stop():
        candidates = es.ask()
        values = [wrapped(c) for c in candidates]
        es.tell(candidates, values)

    return np.array(X_log), np.array(y_log)


X_cma, y_cma = run_cmaes(objective, BOUNDS, max_evals=15, popsize=5, seed=0)

print(f"CMA-ES evaluations used = {len(y_cma)}")
print(f"best x = {X_cma[int(np.argmin(y_cma))]:.4f}")
print(f"best f = {np.min(y_cma):.4f}")
print(f"regret = {np.min(y_cma) - GLOBAL_F:.4f}")''')

code(r'''fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_plot, objective(x_plot), linewidth=1.2, alpha=0.5, label="objective f(x)")
ax.plot(X_cma, y_cma, "o", markersize=7, alpha=0.85,
        label=f"CMA-ES evaluations (n={len(y_cma)})")
ax.plot(X_cma[int(np.argmin(y_cma))], np.min(y_cma), "*", markersize=16,
        label=f"best found: f={np.min(y_cma):.3f}")
ax.set_title("CMA-ES samples: population moves toward good regions")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend()
fig.savefig(os.path.join(FIG_DIR, "13_cmaes_samples.png"), bbox_inches="tight")
plt.show()''')

md(r"""### 구현 함정: `cma`의 seed는 bounds와 함께 쓰면 무시된다

`cma` 소스에서 seed는 다음 조건에서만 적용된다.

```python
if self.opts['randn'] is np.random.randn:
    np.random.seed(opts['seed'])
else:
    print_warning("seed will never be used")
```

`bounds`(또는 `fixed_variables`, `transformation`)를 지정하면 내부 sampler가 **다른 randn**을 쓰기 때문에
`'seed': 0`을 줘도 조용히 무시된다. 그러면 결과가 전역 `np.random` 상태에 의존해서,
앞 셀에서 난수를 얼마나 뽑았는지에 따라 결과가 달라진다. (재현 불가 + 간헐적 수치 오류까지 발생한다.)

해결책은 위 코드처럼 재현 가능한 `RandomState`를 `randn` 옵션으로 **직접 주입**하는 것이다.

> 교육 포인트: 라이브러리의 seed 옵션이 항상 모든 randomness를 지배한다고 가정하지 말 것.
> 재현성을 확인하는 가장 빠른 방법은 **같은 스크립트를 두 번 돌려 출력이 같은지 보는 것**이다.

CMA-ES의 점들은 BO와 다르게 **여러 개가 동시에** 나타난다. 이것이 population 기반 방법의 특징이다.
population이 좋은 영역으로 이동하면서 $\sigma$가 줄어들고, 후보들이 한 곳에 수렴한다.

### 학습 확인 질문

1. CMA-ES의 randomness는 Random Search의 randomness와 무엇이 다른가?
2. covariance matrix $C$는 무엇을 학습하는가?
3. 평가를 병렬로 해야 하는 상황에서 CMA-ES가 BO보다 유리한 이유는 무엇인가?""")

# ------------------------------------------------------------------ Section 17
md(r"""## Section 17. BO vs CMA-ES 정리

| 특성 | BO | CMA-ES |
|---|---|---|
| 핵심 아이디어 | surrogate model | search distribution |
| uncertainty | 명시적으로 사용 ($\sigma(x)$) | 보통 없음 (population spread) |
| sample efficiency | 높음 (수십 회 이내) | 상대적으로 낮음 |
| 많은 평가 가능 | 보통 | 강함 |
| parallel evaluation | 어색함 | 매우 자연스러움 |
| 저차원 + 비싼 objective | **강함** | 가능 |
| 고차원 / 복잡한 landscape | 어려워질 수 있음 | 상대적으로 강함 |
| 하이퍼파라미터 | kernel, $\xi$, 노이즈 | $\sigma_0$, popsize |

핵심 문장: **BO는 "평가 한 번이 비싸다"는 상황에 최적화되어 있고, CMA-ES는 "평가를 많이, 병렬로 할 수 있다"는 상황에 최적화되어 있다.**""")

# ------------------------------------------------------------------ Section 18
md(r"""## Section 18. Random Search vs BO vs CMA-ES

세 알고리즘을 같은 objective, 같은 예산(≈15회)에서 비교한다.
x축은 **Number of objective evaluations**로 통일한다.

주의: 하나의 random seed 결과를 절대적인 우열로 설명하지 않는다. 아래 그래프는 "한 번의 예시"이고,
뒤에서 반복 실험으로 분포를 확인한다.""")

code(r'''best_rs_s0 = np.minimum.accumulate(y_rs15)
best_bo_s0 = np.minimum.accumulate(y_bo)
best_cma_s0 = np.minimum.accumulate(y_cma)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(np.arange(1, len(best_rs_s0) + 1), best_rs_s0, marker="o", linewidth=1.7,
        label="Random Search")
ax.plot(np.arange(1, len(best_bo_s0) + 1), best_bo_s0, marker="s", linewidth=1.7,
        label="Bayesian Optimization")
ax.plot(np.arange(1, len(best_cma_s0) + 1), best_cma_s0, marker="^", linewidth=1.7,
        label=f"CMA-ES (n={len(best_cma_s0)})")
ax.axhline(GLOBAL_F, linestyle="--", linewidth=1.4, label="true global optimum")
ax.set_title("Random Search vs BO vs CMA-ES (one seed, ~15 evaluations)")
ax.set_xlabel("Number of objective evaluations")
ax.set_ylabel("Best objective found so far")
ax.legend()
fig.savefig(os.path.join(FIG_DIR, "14_three_way.png"), bbox_inches="tight")
plt.show()

print(f"RS     final {best_rs_s0[-1]:.4f}")
print(f"BO     final {best_bo_s0[-1]:.4f}")
print(f"CMA-ES final {best_cma_s0[-1]:.4f}")''')

md(r"""### 반복 실험으로 확인

seed만 바꿔 15회 반복해서 세 방법의 최종 regret 분포를 본다. (CMA-ES는 실행 시간이 더 걸리므로 seed 수를 줄였다.)""")

code(r'''N_SEEDS_3 = 15
reg_rs3 = np.zeros(N_SEEDS_3)
reg_bo3 = np.zeros(N_SEEDS_3)
reg_cma3 = np.zeros(N_SEEDS_3)

for s in range(N_SEEDS_3):
    _, y_s = random_search(objective, BOUNDS, n_evals=BUDGET, seed=s)
    _, y_b, _, _ = bayesian_optimization(objective, BOUNDS, n_initial=N_INIT,
                                        n_iterations=N_ITER, seed=s, verbose=False)
    _, y_c = run_cmaes(objective, BOUNDS, max_evals=BUDGET, popsize=5, seed=s)
    reg_rs3[s] = np.min(y_s) - GLOBAL_F
    reg_bo3[s] = np.min(y_b) - GLOBAL_F
    reg_cma3[s] = np.min(y_c) - GLOBAL_F

print(f"{'method':<20}{'mean':>10}{'std':>10}{'median':>10}")
for name, r in [("Random Search", reg_rs3), ("Bayesian Optim.", reg_bo3), ("CMA-ES", reg_cma3)]:
    print(f"{name:<20}{r.mean():>10.4f}{r.std():>10.4f}{np.median(r):>10.4f}")''')

code(r'''fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot([reg_rs3, reg_bo3, reg_cma3],
           tick_labels=["Random Search", "Bayesian Optimization", "CMA-ES"],
           widths=0.5, showmeans=True)
ax.set_title(f"Final regret over {N_SEEDS_3} seeds: RS vs BO vs CMA-ES")
ax.set_ylabel("regret (lower is better)")
ax.set_xlabel("method")
fig.savefig(os.path.join(FIG_DIR, "15_regret_boxplot_three_way.png"), bbox_inches="tight")
plt.show()''')

md(r"""**결과 해석 가이드**

- 이 toy problem은 1차원이고 objective가 매끄러우며, 예산이 15회로 극히 작다.
  이 조건은 **BO에게 유리한 조건**이다. 그래도 CMA-ES가 나쁘지 않다면, 그것은 1D 문제에서 population이 곧 넓은 탐색이 되기 때문이다.
- CMA-ES는 한 세대에 5개를 평가한다. 즉 15회 예산이면 **3~4세대**뿐이다. 세대 수가 적으면 $C$를 학습할 시간이 없다.
  CMA-ES가 진짜 강해지는 것은 예산이 수백 회 이상일 때다. 참고로 이 비교에서 CMA-ES는 20회를 평가했으므로 예산상 약간 유리했다.
- 반대로 BO는 15회 안에서도 "다음 한 점"을 매번 신중하게 고른다. **평가가 1회당 몇 시간 걸리는 상황**이라면 BO가 압도적이다.

### 학습 확인 질문

1. 1D 문제에서 CMA-ES의 population 크기가 5라는 것은 탐색 관점에서 어떤 의미인가?
2. 평가 예산이 15회에서 500회로 늘어난다면, 세 방법의 순위는 어떻게 바뀔 것으로 예상하는가?""")

# ------------------------------------------------------------------ Section 19
md(r"""## Section 19. 최종 개념 정리

### concept map

```text
Random Search
    |
    | previous trials를 활용하지 않음
    v
무작위 next candidate


Bayesian Optimization
    |
    | GP surrogate
    | mu(x), sigma(x)
    | acquisition
    v
next experiment


Evolutionary Algorithm (CMA-ES)
    |
    | population
    | selection
    | mutation / crossover
    v
next generation
```

### 세 문장으로 정리

**Random Search**

```text
Search space 전체에서 무작위로 후보를 뽑는다.
```

**Bayesian Optimization**

```text
과거 관측으로 objective를 모델링하고,
가장 가치 있는 다음 실험점을 선택한다.
```

**CMA-ES**

```text
좋은 후보들이 존재하는 search distribution을 반복적으로 개선한다.
```

### 수식 → 코드 대응표

| 개념 | 수식 | 코드 |
|---|---|---|
| RBF kernel | $k(x,x')=\sigma_f^2 e^{-(x-x')^2/2\ell^2}$ | `rbf_kernel` |
| GP posterior mean | $K_*^{\top}(K+\sigma_n^2I)^{-1}y$ | `gp_posterior` |
| GP posterior variance | $K_{**}-K_*^{\top}(K+\sigma_n^2I)^{-1}K_*$ | `gp_posterior` |
| EI | $(f_{\text{best}}-\mu-\xi)\Phi(Z)+\sigma\phi(Z)$ | `expected_improvement` |
| next point | $\arg\max_x EI(x)$ | `bayesian_optimization` |
| best-so-far | $\min_{i\le t} y_i$ | `np.minimum.accumulate` |
| regret | $f_{\text{best-found}}-f_{\text{global}}$ | Section 14 |

### 한 문장 요약

> GP는 지금까지 관측한 데이터로 black-box objective의 posterior mean과 uncertainty를 주고,
> BO는 이 surrogate와 acquisition function으로 **다음에 평가할 가장 가치 있는 위치**를 고른다.
> Random Search는 이전 결과를 버리지만 BO는 모든 관측을 다음 실험 설계에 활용한다.
> CMA-ES는 objective를 모델링하지 않고 **좋은 해가 있을 search distribution**을 학습한다.""")

# ------------------------------------------------------------------ Experiments
md(r"""---
# 추가 실험

여기서부터는 hyperparameter가 결과에 어떤 영향을 주는지 확인한다.
각 실험은 "왜 이런 일이 일어나는가"를 먼저 예측한 뒤 실행해 보기 바란다.

## Experiment A — length scale 변화

$\ell \in \{0.2, 0.5, 1.0, 2.0\}$로 바꾸며 GP posterior가 어떻게 달라지는지 본다.

예상:

- 너무 작으면 ($\ell=0.2$) 관측점 사이에서 mean이 0으로 급격히 돌아가고 uncertainty가 커진다 → 과소평활
- 너무 크면 ($\ell=2.0$) mean이 실제 곡선의 굴곡을 따라가지 못한다 → 과대평활""")

code(r'''for ell in [0.2, 0.5, 1.0, 2.0]:
    mu_l, var_l = gp_posterior(X_train, y_train, x_plot, length_scale=ell, sigma_n=1e-6)
    std_l = np.sqrt(var_l)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_plot, objective(x_plot), linewidth=1.1, alpha=0.5, label="true objective")
    ax.plot(x_plot, mu_l, linewidth=2.0, label="posterior mean")
    ax.fill_between(x_plot, mu_l - 1.96 * std_l, mu_l + 1.96 * std_l, alpha=0.22,
                    label="95% interval")
    ax.plot(X_train, y_train, "o", markersize=7, label="observations")
    ax.set_title(f"GP posterior with length_scale = {ell}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(os.path.join(FIG_DIR, f"20_len_scale_{ell}.png"), bbox_inches="tight")
    plt.show()

    print(f"length_scale={ell:<4} | mean |error| on grid = "
          f"{np.mean(np.abs(objective(x_plot) - mu_l)):.4f}")''')

md(r"""출력의 평균 절대오차를 비교해 보자. 이 4개 관측점(간격이 넓고 불규칙하다)에 대해서는 $\ell=0.5$가 가장 낮은 오차를 준다.
$\ell=0.2$는 관측점 사이에서 mean이 0으로 급격히 돌아가고, $\ell=1.0, 2.0$은 실제 굴곡을 놓쳐 **과대평활**된다.
즉 최적의 $\ell$은 **관측점의 밀도와 함수의 실제 굴곡**에 달려 있다. 관측점이 많아지면 최적 $\ell$은 작아지는 경향이 있다.

실무에서는 이 $\ell$을 **marginal likelihood를 최대화**해서 자동으로 정한다 (이 Notebook에서는 다루지 않는다).
hyperparameter를 사람이 고정한다는 것이 BO를 쓸 때 가장 번거로운 부분 중 하나다.""")

md(r"""## Experiment B — observation noise 변화

$\sigma_n \in \{0, 0.1, 0.5\}$로 바꾼다. 관측에 노이즈가 있다고 가정하면 GP는 관측값을 **정확히 지나가지 않으려 한다**.
그래서 uncertainty가 0으로 완전히 줄지 않는다.""")

code(r'''for sigma_n in [0.0, 0.1, 0.5]:
    sn = max(sigma_n, 1e-8)   # sigma_n=0 -> 수치적 안정을 위해 아주 작은 값
    mu_n, var_n = gp_posterior(X_train, y_train, x_plot, length_scale=1.0, sigma_n=sn)
    std_n = np.sqrt(var_n)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_plot, objective(x_plot), linewidth=1.1, alpha=0.5, label="true objective")
    ax.plot(x_plot, mu_n, linewidth=2.0, label="posterior mean")
    ax.fill_between(x_plot, mu_n - 1.96 * std_n, mu_n + 1.96 * std_n, alpha=0.22,
                    label="95% interval")
    ax.plot(X_train, y_train, "o", markersize=7, label="observations")
    ax.set_title(f"GP posterior with sigma_n = {sigma_n}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(os.path.join(FIG_DIR, f"21_noise_{sigma_n}.png"), bbox_inches="tight")
    plt.show()

    at_train = np.sqrt(gp_posterior(X_train, y_train, X_train, sigma_n=sn)[1])
    print(f"sigma_n={sigma_n:<4} | std at observed points = {np.round(at_train, 4)}")''')

md(r"""$\sigma_n$이 커질수록 관측점에서도 std가 0이 되지 않는다. 이는 "측정값을 완전히 믿지 않는다"는 모델의 태도다.
실제 실험(센서 노이즈, stochastic simulation)에서는 이 설정이 필수적이다. $\sigma_n$을 0으로 두면 GP가 노이즈를 함수의 굴곡으로 착각한다.""")

md(r"""## Experiment C — initial sample 수 변화

`n_initial` $\in \{2, 3, 5, 10\}$로 바꾸고, **총 예산은 15회로 고정**한 채 BO의 최종 regret을 비교한다.

trade-off:

- 초기점이 적으면 → BO가 일찍부터 EI로 유용한 곳을 고를 수 있다 (단, 초반 GP가 매우 부정확)
- 초기점이 많으면 → GP는 정확해지지만, EI를 쓸 기회(iteration 수)가 줄어든다""")

code(r'''init_list = [2, 3, 5, 10]
N_SEEDS_INIT = 20
results_init = {}

for n_init in init_list:
    n_iter = BUDGET - n_init
    if n_iter <= 0:
        continue
    regs = []
    for s in range(N_SEEDS_INIT):
        _, y_b, _, _ = bayesian_optimization(
            objective, BOUNDS, n_initial=n_init, n_iterations=n_iter,
            seed=s, verbose=False)
        regs.append(np.min(y_b) - GLOBAL_F)
    results_init[n_init] = np.array(regs)
    r = results_init[n_init]
    print(f"n_initial={n_init:<3d} (n_iter={n_iter:<2d}) | mean={r.mean():.4f} "
          f"std={r.std():.4f} median={np.median(r):.4f}")''')

code(r'''fig, ax = plt.subplots(figsize=(8, 5))
labels = [str(k) for k in results_init]
ax.boxplot([results_init[k] for k in results_init], tick_labels=labels, showmeans=True)
ax.set_title(f"Effect of initial design size on BO regret ({N_SEEDS_INIT} seeds, "
             f"{BUDGET} evals total)")
ax.set_xlabel("n_initial (random design points)")
ax.set_ylabel("final regret (lower is better)")
fig.savefig(os.path.join(FIG_DIR, "22_init_size.png"), bbox_inches="tight")
plt.show()''')

md(r"""초기점이 너무 적으면 분산이 커지고, 너무 많으면 개선 기회가 줄어든다.
실무에서는 보통 $d+1$ (차원 + 1) 근처에서 시작하는 것이 관례다. 여기서는 $d=1$이므로 2~3이 합리적이다.""")

md(r"""## Experiment D — EI의 exploration parameter $\xi$

$\xi \in \{0, 0.01, 0.1, 0.5\}$로 바꾼다.

$$\text{EI}(x) = (f_{\text{best}}-\mu(x)-\xi)\Phi(Z) + \sigma(x)\phi(Z)$$

$\xi$가 커지면 "$\mu$가 $f_{\text{best}}$보다 $\xi$만큼 더 좋아야 improvement로 인정"되므로,
mean이 좋은 곳의 매력이 줄고 상대적으로 $\sigma$가 큰 곳이 선택된다 → **exploration 증가**.""")

code(r'''xi_list = [0.0, 0.01, 0.1, 0.5]
N_SEEDS_XI = 20
results_xi = {}

for xi in xi_list:
    regs = []
    for s in range(N_SEEDS_XI):
        _, y_b, _, _ = bayesian_optimization(
            objective, BOUNDS, n_initial=N_INIT, n_iterations=N_ITER,
            xi=xi, seed=s, verbose=False)
        regs.append(np.min(y_b) - GLOBAL_F)
    results_xi[xi] = np.array(regs)
    r = results_xi[xi]
    print(f"xi={xi:<5} | mean={r.mean():.4f} std={r.std():.4f} median={np.median(r):.4f}")''')

code(r'''# xi가 클수록 선택점이 넓게 퍼지는지 한 눈에 확인
fig, ax = plt.subplots(figsize=(10, 5))
for xi in xi_list:
    _, _, hist_xi, grid_xi = bayesian_optimization(
        objective, BOUNDS, n_initial=N_INIT, n_iterations=N_ITER,
        xi=xi, seed=0, verbose=False)
    chosen = [h["x_next"] for h in hist_xi]
    ax.plot(np.arange(1, len(chosen) + 1), chosen, marker="o", linewidth=1.5,
            label=f"xi = {xi}")
ax.set_title("Sequence of selected points for different xi (seed 0)")
ax.set_xlabel("BO iteration")
ax.set_ylabel("selected x_next")
ax.legend()
fig.savefig(os.path.join(FIG_DIR, "23_xi_sequences.png"), bbox_inches="tight")
plt.show()''')

md(r"""$\xi$가 큰 경우 선택점이 더 넓게 흩어지고(exploration), $\xi=0$인 경우 빠르게 한 곳에 붙는 경향이 있다.
단, 이 1D toy problem은 작아서 $\xi$의 효과가 크지 않을 수 있다. 효과가 고차원/복잡한 문제에서 더 분명해진다.""")

# ------------------------------------------------------------------ Optional
md(r"""---
# 선택적 확장: 다른 Acquisition Function

EI 외에 자주 쓰이는 두 가지를 구현하고, **같은 posterior에서 어디를 선택하는지** 비교한다.

### Probability of Improvement (PI)

$$PI(x) = \Phi\left(\frac{f_{\text{best}}-\mu(x)-\xi}{\sigma(x)}\right)$$

"개선이 일어날 확률"만 보고, **개선의 크기는 무시**한다. 그래서 아주 작은 개선이라도 확률이 높으면 선택된다 → 지나치게 exploitation 편향.

### Lower Confidence Bound (LCB)

최소화 문제에서

$$LCB(x) = \mu(x) - \kappa\sigma(x)$$

"낙관적 하한"을 최소화한다. $\kappa$가 크면 exploration이 강해진다 (대표적으로 $\kappa=2$).""")

code(r'''def probability_of_improvement(mu, sigma, best_y, xi=0.01):
    """Minimization 기준 PI."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    Z = (best_y - mu - xi) / sigma
    return norm.cdf(Z)


def lower_confidence_bound(mu, sigma, kappa=2.0):
    """Minimization 기준 LCB (작을수록 좋은 점수)."""
    return np.asarray(mu, dtype=float) - kappa * np.asarray(sigma, dtype=float)''')

code(r'''best_y_demo = float(np.min(y_train))
mu_d, var_d = gp_posterior(X_train, y_train, x_plot, length_scale=1.0, sigma_n=1e-6)
std_d = np.sqrt(var_d)

ei_d = expected_improvement(mu_d, std_d, best_y_demo, xi=0.01)
pi_d = probability_of_improvement(mu_d, std_d, best_y_demo, xi=0.01)
lcb_d = lower_confidence_bound(mu_d, std_d, kappa=2.0)

x_ei = x_plot[int(np.argmax(ei_d))]
x_pi = x_plot[int(np.argmax(pi_d))]
x_lcb = x_plot[int(np.argmin(lcb_d))]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_plot, ei_d, linewidth=2.0, label="EI (max)")
ax.axvline(x_ei, linestyle="--", linewidth=1.4, label=f"EI choice x={x_ei:.2f}")
ax.set_title("Acquisition 1/3: Expected Improvement")
ax.set_xlabel("x")
ax.set_ylabel("EI(x)")
ax.legend()
fig.savefig(os.path.join(FIG_DIR, "30_acq_ei.png"), bbox_inches="tight")
plt.show()''')

code(r'''fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_plot, pi_d, linewidth=2.0, label="PI (max)")
ax.axvline(x_pi, linestyle="--", linewidth=1.4, label=f"PI choice x={x_pi:.2f}")
ax.set_title("Acquisition 2/3: Probability of Improvement")
ax.set_xlabel("x")
ax.set_ylabel("PI(x)")
ax.legend()
fig.savefig(os.path.join(FIG_DIR, "31_acq_pi.png"), bbox_inches="tight")
plt.show()''')

code(r'''fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_plot, lcb_d, linewidth=2.0, label="LCB (min), kappa=2")
ax.axvline(x_lcb, linestyle="--", linewidth=1.4, label=f"LCB choice x={x_lcb:.2f}")
ax.set_title("Acquisition 3/3: Lower Confidence Bound")
ax.set_xlabel("x")
ax.set_ylabel("LCB(x)")
ax.legend()
fig.savefig(os.path.join(FIG_DIR, "32_acq_lcb.png"), bbox_inches="tight")
plt.show()

print(f"EI  selects x = {x_ei:.3f}")
print(f"PI  selects x = {x_pi:.3f}")
print(f"LCB selects x = {x_lcb:.3f}  (kappa=2)")''')

md(r"""세 acquisition function이 서로 다른 점을 선택할 수 있다는 것을 확인했으면 한다.

- **PI**는 확률만 보므로 이미 좋아 보이는 곳에 붙는 경향 (exploitation)
- **LCB**는 $\kappa$로 exploration을 조절할 수 있어 단순하고 강력하다
- **EI**는 개선의 **크기**까지 기대값에 반영하므로 균형이 좋아 가장 널리 쓰인다

### 학습 확인 질문

1. PI가 EI보다 exploitation에 치우치는 이유를 수식으로 설명해 보라.
2. LCB의 $\kappa$를 0으로 두면 어떤 전략이 되는가?""")

md(r"""---
# 마무리: 스스로 확인할 것

다음 문장을 **자기 말로** 설명할 수 있으면 이 Notebook의 목표는 달성된 것이다.

1. Gaussian Process는 지금까지 관측한 데이터를 이용해 black-box objective의 posterior mean과 uncertainty를 제공한다.
2. Bayesian Optimization은 이 surrogate model과 acquisition function을 이용해 다음으로 평가할 가장 가치 있는 위치를 선택한다.
3. Random Search는 이전 결과를 활용하지 않지만, BO는 모든 관측 결과를 다음 실험 설계에 활용한다.
4. CMA-ES는 objective 자체를 surrogate로 모델링하지 않고, 좋은 해가 존재할 가능성이 높은 search distribution을 반복적으로 업데이트한다.
5. 따라서 BO와 EA/CMA-ES는 어느 한쪽이 항상 우월한 것이 아니라, evaluation budget과 search space 특성에 따라 적합성이 달라진다.

## 직접 해볼 것 (연습)

1. `objective`를 2D로 확장해 보라. (예: Rosenbrock 또는 Branin) GP posterior는 grid 대신 meshgrid에서 계산하고,
   EI는 argmax를 2D에서 찾으면 된다. 몇 차원부터 BO가 힘들어지는가?
2. `bayesian_optimization`에 **observation noise**를 추가하라. `y_next = objective(x_next) + noise`로 두고,
   GP의 `sigma_n`을 실제 노이즈에 맞춰 보라. `sigma_n`을 0으로 두면 무슨 일이 생기는가?
3. 더 이상 개선이 없을 때 BO를 멈추는 조건(예: EI 최댓값이 threshold 미만)을 구현해 보라.
4. `bayesian_optimization`에 **batch** 버전을 만들어 보라. EI를 하나 고르고, 그 점을 관측했다고 가정한 뒤
   다시 EI를 계산해 다음 점을 고르는 식("kriging believer")으로 병렬 평가를 흉내낼 수 있다.""")

md(r"""---
## 참고문헌

- Rasmussen & Williams, *Gaussian Processes for Machine Learning*, MIT Press, 2006. (2장: regression, 5장: BO)
- Brochu, Cora & de Freitas, *A Tutorial on Bayesian Optimization of Expensive Cost Functions*, arXiv:1012.2599, 2010.
- Shahriari et al., *Taking the Human Out of the Loop: A Review of Bayesian Optimization*, Proc. IEEE, 2016.
- Hansen, *The CMA Evolution Strategy: A Tutorial*, arXiv:1604.00772, 2016.
- Jones, Schonlau & Welch, *Efficient Global Optimization of Expensive Black-Box Functions*, J. Global Optimization, 1998. (EI의 원 논문)""")
