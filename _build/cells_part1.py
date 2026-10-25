# -*- coding: utf-8 -*-
"""Part 1: Section 0 ~ Section 10."""

CELLS = []
A = CELLS.append


def md(s):
    A(("md", s))


def code(s):
    A(("code", s))


# ------------------------------------------------------------------ Section 0
md(r"""# Bayesian Optimization: Random Search에서 CMA-ES까지

이 Notebook은 **Bayesian Optimization(BO)을 직접 구현하면서 이해하는 것**을 목표로 한다.
라이브러리 사용법을 외우는 것이 아니라, 다음 흐름을 손으로 따라간다.

$$\text{Random Search}\ \rightarrow\ \text{RBF Kernel}\ \rightarrow\ \text{GP Prior}\ \rightarrow\ \text{GP Posterior}\ \rightarrow\ \text{Surrogate Model}\ \rightarrow\ \text{Acquisition Function}\ \rightarrow\ \text{Bayesian Optimization}$$

마지막에는 다음 비교까지 한다.

$$\text{Random Search} \quad vs \quad \text{Bayesian Optimization} \quad vs \quad \text{CMA-ES}$$

## Section 0. 학습 목표

이 Notebook을 끝내면 다음을 설명할 수 있어야 한다.

1. 왜 Random Search가 baseline인지
2. RBF kernel이 무엇을 표현하는지
3. GP prior가 무엇인지
4. 데이터를 관측하면 posterior가 어떻게 바뀌는지
5. posterior mean과 uncertainty의 의미
6. GP가 surrogate model로 왜 적합한지
7. acquisition function이 왜 필요한지
8. exploration과 exploitation의 차이
9. Expected Improvement가 무엇인지
10. BO가 iteration마다 어떻게 동작하는지
11. Random Search와 BO의 sample efficiency 차이
12. CMA-ES와 BO의 철학적 차이

## 읽는 순서

모든 핵심 개념은 **직관 → 수식 → 코드 → 그림 → 실험** 순서로 진행된다.
수식을 완전히 외울 필요는 없지만, 각 수식이 코드의 어느 줄에 대응하는지는 확인하기 바란다.

## 필요한 패키지

```python
# pip install numpy scipy matplotlib scikit-learn cma
```

`cma`는 마지막 비교 실험에서만 사용한다. 나머지는 모두 직접 구현한다.""")

# ------------------------------------------------------------------ Section 1
md(r"""## Section 1. Black-box Objective Function

BO가 풀고자 하는 문제는 다음과 같은 **black-box 최적화**이다.

$$\min_{x \in \mathcal{X}} f(x)$$

여기서 black-box란 다음을 의미한다.

- $f$의 해석적 형태를 모르거나, 알더라도 gradient를 쓰지 않는다.
- $f(x)$ 한 번의 평가가 **비싸다**. (실험이나 시뮬레이션 한 번에 수 분 ~ 수 시간)
- 따라서 **가능한 한 적은 평가 횟수**로 좋은 해를 찾아야 한다.

교육용으로는 1D toy function을 쓴다. local minimum이 여러 개 있고 global minimum이 한 곳에 있는 함수다.""")

code(r'''import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

np.set_printoptions(precision=4, suppress=True)
print("numpy", np.__version__)''')

code(r'''def objective(x):
    """비싼 black-box objective. 우리는 이 함수의 gradient를 쓰지 않는다."""
    x = np.asarray(x, dtype=float)
    return 0.15 * (x - 2.0) ** 2 + np.sin(3.0 * x) + 0.3 * np.sin(7.0 * x)


BOUNDS = (-2.0, 6.0)  # search domain

# dense grid로 global optimum을 근사한다 (실제 문제에서는 알 수 없음)
grid_dense = np.linspace(*BOUNDS, 200001)
y_dense = objective(grid_dense)
idx = int(np.argmin(y_dense))

GLOBAL_X = float(grid_dense[idx])
GLOBAL_F = float(y_dense[idx])

print("Approximate global optimum:")
print(f"  x    = {GLOBAL_X:.6f}")
print(f"  f(x) = {GLOBAL_F:.6f}")''')

code(r'''x_plot = np.linspace(*BOUNDS, 800)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_plot, objective(x_plot), linewidth=1.6, label="objective f(x)")
ax.plot(GLOBAL_X, GLOBAL_F, "o", markersize=8, label="global optimum (approx.)")
ax.set_title("Black-box objective function (1D, non-convex)")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend()
fig.savefig(os.path.join(FIG_DIR, "01_objective.png"), bbox_inches="tight")
plt.show()

print("local minima 개수 확인용: 기울기 부호가 바뀌는 지점")
sign = np.sign(np.gradient(y_dense))
turn = np.where(np.diff(sign) != 0)[0]
print(f"  대략 {len(turn)} 개의 stationary point 근처 지점이 있다")''')

md(r"""**관찰**: 이 함수는 local minimum이 여러 개다. Random Search나 gradient descent는 이 local minimum 중 어디에 갇힐지 알 수 없다.
BO는 "지금까지 관측한 값"과 "아직 관측하지 않아 불확실한 곳"을 함께 고려해서 다음 실험점을 고른다.

### 학습 확인 질문

1. 이 함수가 "black-box"라는 말은 구체적으로 무엇을 못 한다는 뜻인가?
2. dense grid로 global optimum을 구했는데, 실제 BO 문제에서는 이 값을 알 수 없는 이유는 무엇인가?""")

# ------------------------------------------------------------------ Section 2
md(r"""## Section 2. Random Search

가장 단순한 baseline이다. 다음 후보를 이전 결과와 **무관하게** 뽑는다.

$$x_{t+1} \sim p(x)$$

여기서 $p(x)$는 search domain 위의 uniform 분포다. 이전 평가 결과는 전혀 사용되지 않는다.
그럼에도 baseline으로 중요한 이유는, 평가 횟수가 늘어날수록 **최적해에 수렴하는 속도가 이론적으로 보장**되기 때문이다.
BO가 이 baseline보다 나은지 판단하는 기준이 된다.""")

code(r'''def random_search(objective, bounds, n_evals, seed=0):
    """Uniform random sampling으로 objective를 평가한다.

    Returns
    -------
    X : (n_evals,) array, 평가한 x
    y : (n_evals,) array, 평가한 f(x)
    """
    rng = np.random.default_rng(seed)
    low, high = bounds
    X = rng.uniform(low, high, size=n_evals)
    y = objective(X)
    return X, y


N_EVALS_RS = 15
X_rs, y_rs = random_search(objective, BOUNDS, n_evals=N_EVALS_RS, seed=0)

best_idx = int(np.argmin(y_rs))
print(f"Random Search ({N_EVALS_RS} evaluations)")
print(f"  best x = {X_rs[best_idx]:.4f}")
print(f"  best f = {y_rs[best_idx]:.4f}")
print(f"  regret = {y_rs[best_idx] - GLOBAL_F:.4f}")''')

code(r'''fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_plot, objective(x_plot), linewidth=1.4, label="objective f(x)")
ax.plot(X_rs, y_rs, "o", markersize=7, alpha=0.85, label=f"random samples (n={N_EVALS_RS})")
ax.plot(X_rs[best_idx], y_rs[best_idx], "*", markersize=16,
        label=f"best found: f={y_rs[best_idx]:.3f}")
ax.set_title("Random Search: samples are independent of previous results")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend()
fig.savefig(os.path.join(FIG_DIR, "02_random_search.png"), bbox_inches="tight")
plt.show()''')

md(r"""그림에서 점들이 domain 전체에 고르게 흩어져 있는 것을 볼 수 있다. 15번 중 좋은 영역에 2~3번 들어가면 운이 좋은 것이고,
그 영역에 다시 집중해서 파고들지는 않는다. **이전 결과를 활용하지 않기 때문**이다.

### 학습 확인 질문

1. Random Search는 왜 "baseline"이라고 부르는가?
2. 평가 예산이 15회로 제한되어 있을 때, Random Search의 가장 큰 비효율은 무엇인가?""")

# ------------------------------------------------------------------ Section 3
md(r"""## Section 3. RBF Kernel

GP는 "함수들에 대한 분포"다. 그 분포의 성격은 **covariance function(kernel)** 이 결정한다.
가장 널리 쓰이는 것이 RBF(radial basis function) kernel이다.

$$k(x,x') = \sigma_f^2 \exp\left(-\frac{(x-x')^2}{2\ell^2}\right)$$

- $\sigma_f^2$: 함수 값의 진폭(amplitude). 함수가 평균에서 얼마나 멀리 출렁일 수 있는가.
- $\ell$: length scale. **두 점의 거리가 어느 정도일 때까지 서로 비슷한 값**을 가질 수 있는가.

직관: $k(x,x')$는 "$x$에서 $f$를 관측했을 때 $x'$에서의 $f$에 대해 얼마나 알 수 있는가"를 나타내는 유사도다.
$\ell$이 작으면 $k$가 빠르게 0으로 떨어져 이웃 정보가 거의 전달되지 않고, $\ell$이 크면 멀리까지 전달된다.""")

code(r'''def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    """RBF kernel matrix K[i, j] = k(X1[i], X2[j]).

    Parameters
    ----------
    X1, X2 : (n,) or (n, d) array
    length_scale : float, l
    sigma_f : float, signal variance

    Returns
    -------
    K : (len(X1), len(X2)) array
    """
    X1 = np.atleast_2d(np.asarray(X1, dtype=float)).reshape(-1, 1)
    X2 = np.atleast_2d(np.asarray(X2, dtype=float)).reshape(-1, 1)
    sq_dist = (X1 - X2.T) ** 2
    return sigma_f ** 2 * np.exp(-0.5 * sq_dist / length_scale ** 2)


# 손으로 확인: 같은 점이면 k = sigma_f^2, 멀어지면 0
print("k(0, 0)   =", rbf_kernel([0.0], [0.0])[0, 0])
print("k(0, 1)   =", rbf_kernel([0.0], [1.0], length_scale=1.0)[0, 0])
print("k(0, 3)   =", rbf_kernel([0.0], [3.0], length_scale=1.0)[0, 0])''')

code(r'''x_k = np.linspace(-4, 4, 400)
anchor = np.array([0.0])
length_scales = [0.3, 1.0, 2.0]

fig, ax = plt.subplots(figsize=(9, 5))
for ell in length_scales:
    k_vals = rbf_kernel(anchor, x_k, length_scale=ell, sigma_f=1.0)[0]
    ax.plot(x_k, k_vals, linewidth=1.8, label=f"length_scale = {ell}")
ax.axvline(0.0, linewidth=1.0, alpha=0.5)
ax.set_title("RBF kernel k(0, x): similarity decays with distance")
ax.set_xlabel("x")
ax.set_ylabel("k(0, x)")
ax.legend()
fig.savefig(os.path.join(FIG_DIR, "03_rbf_kernel.png"), bbox_inches="tight")
plt.show()''')

md(r"""- 작은 $\ell$: kernel이 좁게 퍼진다 → **빠르게 변하는 함수**를 허용 (유연하지만 과적합 위험)
- 큰 $\ell$: kernel이 넓게 퍼진다 → **매끄러운 함수**를 가정 (관측 사이를 길게 보간)

### 학습 확인 질문

1. $\ell \to 0$에 가까워지면 GP prior에서 샘플링한 함수는 어떤 모양이 되는가?
2. kernel matrix $K$의 대각 성분은 항상 $\sigma_f^2$인 이유는 무엇인가?""")

# ------------------------------------------------------------------ Section 4
md(r"""## Section 4. Gaussian Process Prior

관측 데이터가 하나도 없을 때, GP는 함수에 대한 **사전 분포(prior)** 를 정의한다.

$$f(X) \sim \mathcal{N}(0, K)$$

즉 하나의 함수를 정하는 것이 아니라, **함수들의 분포**를 정한다. 평균은 0이라 가정하고(데이터가 없으니 특별히 선호할 곳이 없다),
함수들 사이의 상관관계는 $K$가 결정한다.

절차는 정확히 세 단계다.

1. grid $X$ 생성
2. covariance matrix $K = k(X, X)$ 계산
3. multivariate Gaussian에서 샘플 추출

$K$가 positive semi-definite이어야 샘플링이 가능하다. RBF kernel은 항상 그렇다.""")

code(r'''X_grid = np.linspace(*BOUNDS, 200)

K_prior = rbf_kernel(X_grid, X_grid, length_scale=1.0, sigma_f=1.0)
print("K shape      :", K_prior.shape)
print("symmetric    :", np.allclose(K_prior, K_prior.T))
print("eigenvalues>0:", np.all(np.linalg.eigvalsh(K_prior) > -1e-8))''')

code(r'''rng = np.random.default_rng(1)
n_samples = 5
prior_samples = rng.multivariate_normal(mean=np.zeros(len(X_grid)), cov=K_prior, size=n_samples)

fig, ax = plt.subplots(figsize=(10, 5))
for i in range(n_samples):
    ax.plot(X_grid, prior_samples[i], linewidth=1.5, alpha=0.9, label=f"sample {i + 1}")
ax.axhline(0.0, linewidth=1.0, alpha=0.5)
ax.set_title("GP prior samples: functions drawn before seeing any data")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend(ncol=2)
fig.savefig(os.path.join(FIG_DIR, "04_gp_prior_samples.png"), bbox_inches="tight")
plt.show()''')

md(r"""샘플들이 **매끄럽게 이어지고**, 서로 다른 함수지만 같은 "성격"(부드러움, 진폭)을 공유하는 것을 볼 수 있다.
이 성격을 정하는 것이 kernel과 hyperparameter($\ell, \sigma_f$)다.

### 학습 확인 질문

1. GP prior에서 함수를 여러 개 샘플링했을 때, 이 함수들의 공통점은 무엇인가?
2. $K$의 비대각 성분이 크면(즉 $\ell$이 크면) 샘플 함수는 어떤 성질을 갖는가?""")

# ------------------------------------------------------------------ Section 5
md(r"""## Section 5. Gaussian Process Posterior

이제 몇 개의 점을 관측했다고 하자. 관측값 $y$가 노이즈를 포함한다고 하면

$$y = f(x) + \varepsilon,\qquad \varepsilon \sim \mathcal{N}(0, \sigma_n^2)$$

관측 이후의 함수에 대한 분포(posterior)는 **multivariate Gaussian의 conditional distribution** 공식으로 주어진다.

$$\mu_* = K_*^{\top}(K + \sigma_n^2 I)^{-1} y$$

$$\Sigma_* = K_{**} - K_*^{\top}(K + \sigma_n^2 I)^{-1} K_*$$

여기서

- $K = k(X, X)$: 학습점끼리의 covariance $(n \times n)$
- $K_* = k(X, X_*)$: 학습점과 예측점 사이 $(n \times m)$
- $K_{**} = k(X_*, X_*)$: 예측점끼리 $(m \times m)$

이 공식의 의미는 하나다. **관측한 점 근처에서는 $K_*$가 커지므로 posterior variance가 줄어들고, posterior mean이 관측값을 따라간다.**

구현 시 주의: `np.linalg.inv`를 쓰지 말고 `np.linalg.solve`를 쓴다 (수치적으로 더 안정적이고 빠르다).""")

code(r'''def gp_posterior(X_train, y_train, X_test, length_scale=1.0, sigma_f=1.0, sigma_n=1e-6):
    """GP posterior mean and variance at X_test given noisy observations.

    Parameters
    ----------
    X_train, y_train : (n,) arrays
    X_test : (m,) array
    length_scale, sigma_f : kernel hyperparameters
    sigma_n : observation noise standard deviation

    Returns
    -------
    mu : (m,) posterior mean
    var : (m,) posterior variance (diagonal of Sigma_*)
    """
    X_train = np.asarray(X_train, dtype=float).ravel()
    y_train = np.asarray(y_train, dtype=float).ravel()
    X_test = np.asarray(X_test, dtype=float).ravel()

    K = rbf_kernel(X_train, X_train, length_scale, sigma_f)
    K_star = rbf_kernel(X_train, X_test, length_scale, sigma_f)      # (n, m)
    K_starstar = rbf_kernel(X_test, X_test, length_scale, sigma_f)   # (m, m)

    A = K + sigma_n ** 2 * np.eye(len(X_train))

    # A^{-1} y  and  A^{-1} K_*  를 solve로 한 번에 구한다
    alpha = np.linalg.solve(A, y_train)          # (n,)
    mu = K_star.T @ alpha                        # (m,)

    V = np.linalg.solve(A, K_star)               # (n, m)
    cov = K_starstar - K_star.T @ V              # (m, m)
    var = np.diag(cov).copy()
    var = np.clip(var, 1e-12, None)
    return mu, var''')

code(r'''X_train = np.array([-1.5, 0.0, 2.2, 4.8])
y_train = objective(X_train)

mu, var = gp_posterior(X_train, y_train, x_plot, length_scale=1.0, sigma_n=1e-6)

print("X_train :", X_train)
print("y_train :", np.round(y_train, 4))
print("posterior mean at train points:", np.round(
    gp_posterior(X_train, y_train, X_train)[0], 4))
print("posterior std  at train points:", np.round(
    np.sqrt(gp_posterior(X_train, y_train, X_train)[1]), 6))''')

code(r'''std = np.sqrt(var)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_plot, objective(x_plot), linewidth=1.2, alpha=0.6, label="true objective")
ax.plot(x_plot, mu, linewidth=2.0, label="GP posterior mean")
ax.fill_between(x_plot, mu - 1.96 * std, mu + 1.96 * std, alpha=0.25,
                label="95% credible interval")
ax.plot(X_train, y_train, "o", markersize=8, label="observations")
ax.set_title("GP posterior: mean tracks data, uncertainty shrinks near observations")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend(loc="upper left")
fig.savefig(os.path.join(FIG_DIR, "05_gp_posterior.png"), bbox_inches="tight")
plt.show()''')

md(r"""그림에서 세 가지를 확인하자.

1. 관측점에서 posterior mean이 관측값과 거의 일치한다 ($\sigma_n$이 작기 때문).
2. 관측점에서 credible interval 폭이 거의 0으로 줄어든다 → **uncertainty 감소**.
3. 관측점 사이, 특히 양 끝($x < -1.5$, $x > 4.8$)에서는 prior 수준으로 다시 벌어진다 → **데이터가 없는 곳은 모른다**.

수식으로 보면, 관측점과 멀어지면 $K_* \to 0$이므로 $\Sigma_* \to K_{**}$, 즉 prior variance로 돌아간다.

### 학습 확인 질문

1. 왜 관측점 근처에서 uncertainty가 감소하는가?
2. $K_*$가 거의 0이면 posterior variance는 어떻게 되는가?
3. $\sigma_n$을 크게 하면 posterior mean과 variance는 각각 어떻게 변하는가?""")

# ------------------------------------------------------------------ Section 6
md(r"""## Section 6. Surrogate Model로서의 GP

지금까지 GP를 regression 도구로 봤다. BO에서는 **surrogate model**(대리 모델)로 쓴다.

> 실제 objective $f(x)$는 매우 비싸다. 그래서 이미 평가한 데이터만으로 값싼 근사 모델 $\hat f(x)$를 만든다.

핵심은 GP가 **두 가지를 동시에** 준다는 점이다.

$$\mu(x) \qquad \text{and} \qquad \sigma(x)$$

즉 `prediction + uncertainty`다.

일반적인 회귀 모델(선형회귀, 신경망)은 예측값만 준다. 예측이 얼마나 믿을 만한지 알려면 별도의 작업이 필요하다.
GP는 posterior distribution 자체가 uncertainty를 포함하므로, **"어디를 더 실험해야 하는가"를 결정하는 데 직접 쓸 수 있다.**
이것이 BO가 GP를 surrogate로 쓰는 이유다.""")

code(r'''# surrogate의 품질을 한 눈에: 예측 오차와 불확실성의 관계
mu_t, var_t = gp_posterior(X_train, y_train, x_plot, length_scale=1.0, sigma_n=1e-6)
err = np.abs(objective(x_plot) - mu_t)
std_t = np.sqrt(var_t)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x_plot, std_t, linewidth=2.0, label="posterior std (model uncertainty)")
ax.plot(x_plot, err, linewidth=1.4, alpha=0.85, label="|true - mean| (actual error)")
ax.set_title("Surrogate uncertainty vs actual prediction error")
ax.set_xlabel("x")
ax.set_ylabel("magnitude")
ax.legend()
fig.savefig(os.path.join(FIG_DIR, "06_surrogate_uncertainty.png"), bbox_inches="tight")
plt.show()

print("correlation(std, |error|) =", round(float(np.corrcoef(std_t, err)[0, 1]), 3))''')

md(r"""상관계수는 0.3 정도로 아주 높지는 않다. 그 이유는 오차 $|f-\mu|$에 **bias**가 섞여 있기 때문이다:
관측점 근처에서도 $\mu$는 함수의 굴곡을 완전히 맞히지 못하므로 오차가 0이 아니다.
반면 $\sigma(x)$는 순수하게 "데이터가 없는 정도"만 반영한다.

그럼에도 중요한 성질은 유지된다: **데이터가 없는 구간에서는 $\sigma$가 크고, 관측점 근처에서는 작다.**
BO가 실제로 쓰는 것은 $\sigma$의 절댓값이 아니라 **어디가 상대적으로 더 불확실한가**라는 순서다.
그 순서를 이용해 다음 실험 위치를 정한다. 이것이 다음 두 섹션의 주제다.""")

# ------------------------------------------------------------------ Section 7
md(r"""## Section 7. Exploration vs Exploitation

acquisition function을 이해하려면 먼저 이 trade-off를 이해해야 한다.

### Exploitation (이용)

posterior mean $\mu(x)$가 좋은(작은) 위치를 고른다.

$$x_{\text{next}} = \arg\min_x \mu(x)$$

지금까지의 정보만 믿고 **가장 좋아 보이는 곳**을 파고든다.
문제: GP mean은 관측점 근처에서만 믿을 만하고, 나머지는 prior(0)로 되돌아간다. mean만 보면 **아직 안 가본 곳의 진짜 좋은 값**을 영원히 놓칠 수 있다.

### Exploration (탐색)

uncertainty $\sigma(x)$가 큰 위치를 고른다.

$$x_{\text{next}} = \arg\max_x \sigma(x)$$

**모르는 곳**을 확인해서 정보를 얻는다.
문제: 정보만 얻고 성능은 전혀 개선되지 않을 수 있다.

### BO의 선택

BO는 두 신호를 **하나의 점수 함수**로 결합하고, 그 점수의 최대점을 다음 실험으로 삼는다.

$$\text{acquisition}(x) = g\big(\mu(x), \sigma(x)\big)$$

이렇게 하면 "좋아 보이면서 + 아직 불확실한" 곳이 자동으로 선택된다. 다음 섹션에서 $g$의 가장 대표적인 형태인 EI를 구현한다.""")

code(r'''# 같은 posterior에서 exploitation 후보와 exploration 후보를 각각 계산
mu_e, var_e = gp_posterior(X_train, y_train, x_plot, length_scale=1.0, sigma_n=1e-6)
std_e = np.sqrt(var_e)

x_exploit = x_plot[int(np.argmin(mu_e))]
x_explore = x_plot[int(np.argmax(std_e))]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_plot, mu_e, linewidth=2.0, label="posterior mean (exploitation signal)")
ax.fill_between(x_plot, mu_e - 1.96 * std_e, mu_e + 1.96 * std_e, alpha=0.2,
                label="95% interval")
ax.plot(X_train, y_train, "o", markersize=8, label="observations")
ax.axvline(x_exploit, linestyle="--", linewidth=1.5,
           label=f"min mean  x={x_exploit:.2f}")
ax.axvline(x_explore, linestyle=":", linewidth=1.8,
           label=f"max std   x={x_explore:.2f}")
ax.set_title("Exploitation vs Exploration: two pure strategies disagree")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend(loc="upper left", fontsize=9)
fig.savefig(os.path.join(FIG_DIR, "07_explore_exploit.png"), bbox_inches="tight")
plt.show()

print(f"pure exploitation choice: x = {x_exploit:.3f}")
print(f"pure exploration  choice: x = {x_explore:.3f}")''')

md(r"""두 전략이 서로 다른 곳을 가리키는 것을 볼 수 있다. 어느 한쪽만 쓰면 실패한다.

### 학습 확인 질문

1. 왜 GP mean만으로 next point를 고르면 안 되는가?
2. Exploration과 Exploitation이 왜 동시에 필요한가?
3. objective evaluation이 매우 비쌀수록 BO가 유리한 이유는 무엇인가?""")

# ------------------------------------------------------------------ Section 8
md(r"""## Section 8. Expected Improvement

acquisition function 중 가장 널리 쓰이는 **Expected Improvement (EI)** 를 직접 구현한다. 최소화 문제 기준이다.

현재까지의 best 값을 $f_{\text{best}} = \min(y)$라 하자. 어떤 점 $x$를 평가했을 때 얻게 될 improvement는

$$I(x) = \max\big(f_{\text{best}} - f(x),\ 0\big)$$

$f(x)$는 아직 모르지만, GP posterior에 따르면 $f(x) \sim \mathcal{N}(\mu(x), \sigma^2(x))$다.
따라서 $I(x)$도 확률변수이고, 그 **기댓값**을 최대화하면 된다.

$$EI(x) = \mathbb{E}[I(x)]$$

이 기댓값은 closed form으로 계산된다. $Z = \dfrac{f_{\text{best}} - \mu(x) - \xi}{\sigma(x)}$라 두면

$$EI(x) = \big(f_{\text{best}} - \mu(x) - \xi\big)\Phi(Z) + \sigma(x)\phi(Z)$$

여기서 $\Phi$는 standard normal CDF, $\phi$는 PDF다. $\xi \geq 0$는 exploration을 조금 더 밀어주는 파라미터다.

**직관적으로 읽는 법**: 첫 항은 "mean이 좋을수록 커지고"(exploitation), 두 항 모두 "$\sigma$가 클수록 커진다"(exploration).
$\sigma = 0$이면 EI도 0이 되어 이미 아는 곳은 절대 선택되지 않는다.""")

code(r'''def expected_improvement(mu, sigma, best_y, xi=0.01):
    """Minimization 기준 Expected Improvement.

    Parameters
    ----------
    mu, sigma : (m,) posterior mean and std
    best_y : float, 지금까지 관측한 최소값
    xi : float, exploration 여유 파라미터

    Returns
    -------
    ei : (m,) array
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.maximum(sigma, 1e-12)          # 0으로 나누기 방지

    improvement = best_y - mu - xi
    Z = improvement / sigma
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-11] = 0.0
    return np.maximum(ei, 0.0)''')

code(r'''# 간단한 sanity check: sigma -> 0 이면 EI -> 0, mean이 좋을수록 EI가 크다
mu_test = np.array([0.0, 0.0, 0.0])
sig_test = np.array([1.0, 0.1, 1e-12])
print("EI (sigma 큼/작음/0):", np.round(expected_improvement(mu_test, sig_test, best_y=0.0), 6))
print("EI (mean 낮을수록 큼) :", np.round(
    expected_improvement(np.array([-1.0, 0.0, 1.0]), np.ones(3) * 0.5, best_y=0.0), 4))''')

md(r"""출력에서 확인할 것:

- $\sigma \to 0$이면 EI $\to 0$ (이미 확실히 아는 곳은 갈 이유가 없다)
- $\mu$가 작을수록 EI가 크다 (mean이 좋은 곳을 선호)
- 하지만 $\sigma$가 충분히 크면 mean이 나빠도 EI가 남는다 (불확실한 곳도 후보)

### 학습 확인 질문

1. EI에서 $\xi$를 크게 하면 선택이 exploitation 쪽으로 가는가, exploration 쪽으로 가는가?
2. $\sigma(x) = 0$인 점의 EI가 정확히 0인 이유를 수식으로 설명해 보라.""")

# ------------------------------------------------------------------ Section 9
md(r"""## Section 9. Acquisition Function 시각화

이제 같은 posterior에서 EI를 계산하고, **EI를 최대화하는 위치**를 다음 실험점으로 고른다.

$$x_{\text{next}} = \arg\max_x EI(x)$$

아래 두 그림은 같은 x grid, 같은 GP posterior를 사용하지만 서로 다른 것을 보여준다. (spec 규칙에 따라 subplot을 쓰지 않고 각각 별도 figure로 만든다.)""")

code(r'''best_y = float(np.min(y_train))
ei_vals = expected_improvement(mu_e, std_e, best_y=best_y, xi=0.01)
x_next = x_plot[int(np.argmax(ei_vals))]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_plot, mu_e, linewidth=2.0, label="GP posterior mean")
ax.fill_between(x_plot, mu_e - 1.96 * std_e, mu_e + 1.96 * std_e, alpha=0.22,
                label="95% interval")
ax.plot(X_train, y_train, "o", markersize=8, label="observations")
ax.axhline(best_y, linestyle="--", linewidth=1.2, label=f"current best = {best_y:.3f}")
ax.set_title("1) GP posterior belief (what we know)")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend(loc="upper left")
fig.savefig(os.path.join(FIG_DIR, "08_gp_belief_for_ei.png"), bbox_inches="tight")
plt.show()''')

code(r'''fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_plot, ei_vals, linewidth=2.0, label="Expected Improvement EI(x)")
ax.axvline(x_next, linestyle="--", linewidth=1.6,
           label=f"argmax EI: x={x_next:.3f}")
ax.plot(x_next, ei_vals.max(), "o", markersize=9, label=f"EI={ei_vals.max():.5f}")
ax.set_title("2) Acquisition function: where is the next experiment worth doing?")
ax.set_xlabel("x")
ax.set_ylabel("EI(x)")
ax.legend()
fig.savefig(os.path.join(FIG_DIR, "09_expected_improvement.png"), bbox_inches="tight")
plt.show()

print(f"current best   = {best_y:.4f}")
print(f"selected x_next= {x_next:.4f}")
print(f"EI(x_next)     = {ei_vals.max():.6f}")
print(f"posterior mu   = {np.interp(x_next, x_plot, mu_e):.4f}")
print(f"posterior std  = {np.interp(x_next, x_plot, std_e):.4f}")''')

md(r"""EI 곡선이 봉우리를 갖는 위치는 대개 **"mean이 괜찮으면서 uncertainty가 아직 큰"** 영역이다.
순수 exploitation 위치(mean 최소)와 순수 exploration 위치(max std)의 중간 어딘가가 되는 경우가 많다. 이것이 trade-off의 실제 모습이다.""")
