# Bayesian Optimization Educational Notebook — Development Specification

## 1. 목적

이 프로젝트의 목표는 **Bayesian Optimization(BO)을 처음부터 끝까지 직접 체험하며 이해할 수 있는 교육용 Jupyter Notebook**을 만드는 것이다.

단순히 라이브러리 사용법을 보여주는 것이 아니라, 다음 흐름을 학습자가 직접 따라갈 수 있어야 한다.

\[
\text{Random Search}
\rightarrow
\text{RBF Kernel}
\rightarrow
\text{Gaussian Process Prior}
\rightarrow
\text{Gaussian Process Posterior}
\rightarrow
\text{Surrogate Model}
\rightarrow
\text{Acquisition Function}
\rightarrow
\text{Bayesian Optimization}
\]

추가로 최종 단계에서는 다음 비교까지 포함한다.

\[
\text{Random Search}
\quad vs \quad
\text{Bayesian Optimization}
\quad vs \quad
\text{CMA-ES}
\]

학습자는 이미 다음 개념을 대략 알고 있다고 가정한다.

- Random Search의 기본 개념
- Gaussian distribution
- RBF kernel의 의미
- Gaussian Process의 prior/posterior 큰 그림
- posterior mean과 variance의 의미
- BO가 black-box optimization이라는 점

하지만 수식을 완전히 외우거나 GP를 깊게 구현해본 상태는 아니므로, **직관 → 수식 → 코드 → 그림 → 실험** 순서로 설명해야 한다.

---

# 2. 산출물

최종적으로 아래 파일을 생성한다.

```text
bayesian_optimization_tutorial.ipynb
```

가능하면 다음 보조 폴더도 사용한다.

```text
figures/
```

단, notebook 자체만 실행해도 모든 결과가 재현 가능해야 한다.

---

# 3. Notebook의 교육 철학

## 3.1 라이브러리 호출보다 직접 구현 우선

초반 핵심 개념은 가능한 한 직접 구현한다.

- RBF kernel 직접 구현
- GP posterior mean/variance 직접 계산
- Expected Improvement 직접 구현
- BO loop 직접 구현

비교 실험용으로는 `scikit-learn`, `scipy`, `cma` 등의 라이브러리를 사용해도 된다.

## 3.2 모든 핵심 개념은 시각화

학습자가 다음을 그림으로 볼 수 있어야 한다.

- RBF kernel shape
- GP prior sample
- 관측 전 uncertainty
- GP posterior mean
- GP posterior uncertainty
- 관측점 주변 uncertainty 감소
- Expected Improvement
- next candidate 선택
- BO iteration progression
- Random Search vs BO 성능
- CMA-ES vs BO 성능

## 3.3 함수 평가 횟수를 핵심 metric으로 사용

BO의 핵심은 단순 runtime 감소가 아니다.

중요한 것은:

\[
\text{objective evaluation budget}
\]

이다.

따라서 모든 알고리즘 비교에서 x축은 가능하면 다음으로 통일한다.

```text
Number of objective evaluations
```

---

# 4. 필요한 패키지

Notebook 시작 부분에 다음 설치 명령을 제공한다.

```python
# pip install numpy scipy matplotlib scikit-learn cma
```

필수 패키지:

```text
numpy
scipy
matplotlib
scikit-learn
cma
```

가능하면 추가 라이브러리 의존성은 최소화한다.

---

# 5. 전체 Notebook 구성

## Section 0. 학습 목표

Markdown cell에서 아래 내용을 설명한다.

이 Notebook을 끝내면 다음을 이해할 수 있어야 한다.

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

---

## Section 1. Black-box Objective Function

먼저 1D toy function을 만든다.

조건:

- local minimum이 여러 개 존재
- global minimum이 한 곳에 존재
- 그래프로 보기 쉬움
- 미분 없이 black-box처럼 취급

추천 예시:

```python
def objective(x):
    return (
        0.15 * (x - 2.0) ** 2
        + np.sin(3 * x)
        + 0.3 * np.sin(7 * x)
    )
```

search domain:

```text
x ∈ [-2, 6]
```

시각화:

```text
x축: x
y축: f(x)
```

global optimum은 dense grid를 이용해 근사값을 구한다.

출력:

```text
Approximate global optimum:
x = ...
f(x) = ...
```

---

## Section 2. Random Search

Random Search를 직접 구현한다.

```python
def random_search(objective, bounds, n_evals, seed):
    ...
```

15회 정도 평가한다.

시각화:

- true objective curve
- random samples
- best point

설명:

Random Search는

\[
x_{t+1} \sim p(x)
\]

이며 이전 평가 결과를 다음 candidate 생성에 사용하지 않는다.

---

## Section 3. RBF Kernel

RBF kernel을 직접 구현한다.

\[
k(x,x')
=
\sigma_f^2
\exp\left(
-\frac{(x-x')^2}{2\ell^2}
\right)
\]

코드:

```python
def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    ...
```

시각화할 length scale:

```text
0.3
1.0
2.0
```

설명:

- 작은 length scale → 빠르게 변하는 함수 허용
- 큰 length scale → smoother function 가정

---

## Section 4. Gaussian Process Prior

관측 데이터 없이 GP prior에서 함수를 샘플링한다.

절차:

1. grid 생성
2. covariance matrix 계산
3. multivariate Gaussian sample 생성

예:

```python
K = rbf_kernel(X, X)
samples = np.random.multivariate_normal(...)
```

5개 정도 함수 sample 시각화.

설명:

\[
f(X)
\sim
\mathcal N(0,K)
\]

즉 GP는 하나의 함수를 정의하는 것이 아니라 함수들의 분포를 정의한다.

---

## Section 5. Gaussian Process Posterior

관측점을 3~5개 선택한다.

예:

```python
X_train = [-1.5, 0.0, 2.2, 4.8]
```

각 위치의 objective를 평가한다.

posterior mean:

\[
\mu_*
=
K_*^T
(K+\sigma_n^2I)^{-1}
y
\]

posterior covariance:

\[
\Sigma_*
=
K_{**}
-
K_*^T
(K+\sigma_n^2I)^{-1}
K_*
\]

이를 직접 구현한다.

```python
def gp_posterior(
    X_train,
    y_train,
    X_test,
    length_scale=1.0,
    sigma_f=1.0,
    sigma_n=1e-6
):
    ...
```

주의:

실제 구현에서는 matrix inverse를 직접 계산하지 말고

```python
np.linalg.solve
```

를 사용한다.

---

## Section 6. Posterior 시각화

다음 요소를 한 그래프에 표시한다.

- true objective
- GP posterior mean
- observed points
- ±1.96 sigma uncertainty

설명:

\[
f(x)|D
\sim
\mathcal N(
\mu(x),
\sigma^2(x)
)
\]

관측점 주변에서는 uncertainty가 줄어드는 것을 강조한다.

---

## Section 7. Surrogate Model

이제 GP를 단순 regression model이 아니라 **surrogate model** 관점으로 설명한다.

핵심 문장:

> 실제 objective \(f(x)\)가 매우 비싸기 때문에, 이미 평가한 데이터만 이용해 값싼 근사 모델 \(\hat f(x)\)을 만든다.

GP의 장점은 다음 두 값을 동시에 제공한다는 것이다.

\[
\mu(x)
\]

\[
\sigma(x)
\]

즉,

```text
prediction + uncertainty
```

이다.

---

## Section 8. Exploration vs Exploitation

Markdown 설명과 간단한 그림을 만든다.

### Exploitation

현재 mean이 좋아 보이는 위치.

\[
\mu(x)
\]

가 좋은 곳.

### Exploration

uncertainty가 큰 위치.

\[
\sigma(x)
\]

가 큰 곳.

BO는 둘의 trade-off를 다룬다.

---

## Section 9. Expected Improvement

최소화 문제 기준으로 Expected Improvement를 직접 구현한다.

현재 best:

\[
f_{best}
=
\min(y)
\]

improvement:

\[
I(x)
=
\max(
f_{best}-f(x),
0
)
\]

Expected Improvement:

\[
EI(x)
=
\mathbb E[I(x)]
\]

closed form:

\[
Z
=
\frac{
f_{best}
-
\mu(x)
-
\xi
}{
\sigma(x)
}
\]

\[
EI(x)
=
(f_{best}-\mu(x)-\xi)\Phi(Z)
+
\sigma(x)\phi(Z)
\]

구현:

```python
def expected_improvement(mu, sigma, best_y, xi=0.01):
    ...
```

`scipy.stats.norm` 사용 가능.

---

## Section 10. Acquisition Function 시각화

같은 x grid에서 다음 두 그래프를 만든다.

단, subplot을 사용하지 말고 각각 별도 figure로 만든다.

첫 번째:

```text
GP posterior mean + uncertainty
```

두 번째:

```text
Expected Improvement
```

EI 최대 위치에 vertical line 표시.

\[
x_{next}
=
\arg\max_x EI(x)
\]

---

## Section 11. One BO Iteration

현재 데이터에서 실제 한 번의 BO iteration을 실행한다.

과정:

```text
1. GP fit
2. posterior 계산
3. EI 계산
4. EI 최대 위치 선택
5. 실제 objective 평가
6. training data에 추가
```

각 단계의 값을 출력한다.

```text
Current best = ...
Selected x_next = ...
EI(x_next) = ...
Observed f(x_next) = ...
```

---

## Section 12. Full Bayesian Optimization Loop

직접 BO loop를 구현한다.

```python
def bayesian_optimization(
    objective,
    bounds,
    n_initial=3,
    n_iterations=12,
    ...
):
    ...
```

초기 3개 점은 random sampling.

그 이후는 EI로 선택.

history에는 매 iteration의 다음 내용을 저장한다.

```python
{
    "iteration": ...,
    "X": ...,
    "y": ...,
    "mu": ...,
    "sigma": ...,
    "ei": ...,
    "x_next": ...,
}
```

---

## Section 13. BO 진행 과정 시각화

다음 iteration을 그림으로 보여준다.

```text
iteration 0
iteration 2
iteration 5
iteration 10
```

각 그림에 다음을 표시한다.

- objective
- GP mean
- uncertainty
- observations
- selected next point

목표는 BO가 시간이 지나며 유망한 영역에 집중하는 것을 보여주는 것이다.

---

## Section 14. Random Search vs Bayesian Optimization

동일한 evaluation budget을 사용한다.

예:

```text
15 evaluations
```

각 알고리즘에서

```python
best_so_far = np.minimum.accumulate(...)
```

를 계산한다.

그래프:

x축:

```text
Number of objective evaluations
```

y축:

```text
Best objective found so far
```

다음 세 선:

```text
Random Search
Bayesian Optimization
True global optimum
```

---

## Section 15. 반복 실험

한 번의 seed만 비교하면 운이 개입할 수 있으므로 반복 실험을 한다.

예:

```text
30 random seeds
```

각 방법에 대해 최종 regret 계산.

\[
regret
=
f_{best-found}
-
f_{global}
\]

결과:

```text
mean regret
std regret
median regret
```

bar plot 또는 boxplot으로 비교한다.

---

## Section 16. CMA-ES 소개

Evolutionary Algorithm 계열의 대표 예로 CMA-ES를 간단하게 소개한다.

핵심 수식:

\[
x
\sim
\mathcal N(
m,
\sigma^2C
)
\]

설명:

- \(m\): 좋은 해들이 모인 중심
- \(\sigma\): 전체 search scale
- \(C\): 탐색 방향과 correlation

핵심 차이:

BO:

```text
objective function을 surrogate로 모델링
```

CMA-ES:

```text
좋은 candidate가 존재할 search distribution을 학습
```

---

## Section 17. CMA-ES 실험

`cma` 패키지를 사용한다.

같은 objective function을 최적화한다.

가능한 한 BO와 비슷한 evaluation budget으로 비교한다.

주의:

CMA-ES는 population 단위로 평가하기 때문에 budget이 정확히 같지 않을 수 있다.

가능하면 가장 가까운 evaluation 횟수로 비교하고 이 차이를 설명한다.

---

## Section 18. BO vs CMA-ES

표로 비교한다.

| 특성 | BO | CMA-ES |
|---|---|---|
| 핵심 아이디어 | surrogate model | search distribution |
| uncertainty | 명시적으로 사용 | 보통 없음 |
| sample efficiency | 높음 | 상대적으로 낮음 |
| 많은 평가 가능 | 보통 | 강함 |
| parallel evaluation | 가능 | 매우 자연스러움 |
| low-dimensional expensive objective | 강함 | 가능 |
| complex/high-dimensional search | 어려워질 수 있음 | 상대적으로 강함 |

---

## Section 19. Random Search vs BO vs CMA-ES

세 알고리즘을 동일 objective에서 비교한다.

그래프:

```text
Best objective vs Number of Evaluations
```

세 알고리즘:

```text
Random Search
Bayesian Optimization
CMA-ES
```

주의:

하나의 random seed 결과를 절대적인 우열처럼 설명하지 않는다.

---

## Section 20. 최종 개념 정리

Notebook 마지막에 다음 concept map을 Markdown으로 작성한다.

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


Evolutionary Algorithm
    |
    | population
    | selection
    | mutation / crossover
    v
next generation
```

그리고 아래 세 문장을 강조한다.

### Random Search

```text
Search space 전체에서 무작위로 후보를 뽑는다.
```

### Bayesian Optimization

```text
과거 관측으로 objective를 모델링하고,
가장 가치 있는 다음 실험점을 선택한다.
```

### CMA-ES

```text
좋은 후보들이 존재하는 search distribution을 반복적으로 개선한다.
```

---

# 6. 반드시 포함할 학습 확인 질문

각 주요 section 끝에 학습 확인 질문을 1~3개 넣는다.

### GP posterior

1. 왜 관측점 근처에서 uncertainty가 감소하는가?
2. \(K_*\)가 거의 0이면 posterior variance는 어떻게 되는가?

### BO

1. 왜 GP mean만으로 next point를 고르면 안 되는가?
2. Exploration과 Exploitation이 왜 동시에 필요한가?
3. objective evaluation이 매우 비쌀수록 BO가 유리한 이유는 무엇인가?

### CMA-ES

1. CMA-ES의 randomness는 Random Search와 무엇이 다른가?
2. covariance matrix \(C\)는 무엇을 학습하는가?

---

# 7. 코드 스타일

다음 원칙을 따른다.

- 함수 이름은 직관적으로 작성
- magic number 최소화
- random seed 명시
- 각 함수에 짧은 docstring
- 너무 복잡한 class 설계 금지
- 교육 목적이므로 procedural code 우선
- 각 핵심 수식 바로 아래에 해당 코드 배치
- plot title, axis label, legend 반드시 작성

---

# 8. 시각화 규칙

Matplotlib을 사용한다.

- subplot 사용 금지
- 각 개념마다 별도 figure 생성
- 특정 색상 강제 지정 금지
- figsize는 대략 `(8, 5)` 또는 `(10, 5)`
- grid는 가볍게 사용 가능
- 모든 그래프에 제목과 축 이름 포함

---

# 9. 수식 표기

Notebook의 Markdown cell에서는 LaTeX를 사용한다.

inline:

```text
\( ... \)
```

display:

```text
\[
...
\]
```

긴 설명 문장을 `\text{}` 안에 넣지 않는다.

수식은 수식만 쓰고 설명은 Markdown 본문으로 분리한다.

---

# 10. 난이도 조절

이 Notebook은 학부 수준의 확률론/선형대수 지식을 가정하지만, 다음은 자세히 유도하지 않아도 된다.

- multivariate Gaussian conditional distribution의 완전한 증명
- Cholesky decomposition의 세부 수학
- GP hyperparameter marginal likelihood 최적화의 상세 유도

대신 다음은 반드시 직관적으로 설명한다.

- covariance matrix의 의미
- posterior mean
- posterior uncertainty
- acquisition function
- sample efficiency

---

# 11. 추가 실험

가능하면 마지막에 다음 실험도 넣는다.

## Experiment A — length scale 변화

```text
0.2
0.5
1.0
2.0
```

GP posterior가 어떻게 달라지는지 비교.

## Experiment B — observation noise 변화

```text
sigma_n = 0
sigma_n = 0.1
sigma_n = 0.5
```

posterior uncertainty 변화 관찰.

## Experiment C — initial sample 수

```text
2
3
5
10
```

BO 성능 변화 관찰.

## Experiment D — EI exploration parameter

```text
xi = 0
0.01
0.1
0.5
```

exploration 정도 변화 관찰.

---

# 12. 선택적 확장

기본 Notebook 완성 이후 optional section으로 아래를 추가할 수 있다.

## Probability of Improvement

\[
PI(x)
=
\Phi
\left(
\frac{
f_{best}-\mu(x)-\xi
}{
\sigma(x)
}
\right)
\]

## Lower Confidence Bound

최소화 문제에서:

\[
LCB(x)
=
\mu(x)
-
\kappa\sigma(x)
\]

## Acquisition Function 비교

같은 GP posterior에서 다음을 각각 plot하여 어떤 위치를 next candidate로 선택하는지 비교한다.

```text
EI
PI
LCB
```

---

# 13. 완료 조건

Notebook은 아래 조건을 모두 만족해야 완료된 것으로 본다.

- [ ] 처음부터 끝까지 restart + run all 가능
- [ ] Random Search 구현
- [ ] RBF kernel 직접 구현
- [ ] GP prior 시각화
- [ ] GP posterior 직접 구현
- [ ] posterior uncertainty 시각화
- [ ] GP를 surrogate model로 설명
- [ ] EI 직접 구현
- [ ] BO loop 직접 구현
- [ ] BO iteration 시각화
- [ ] Random Search vs BO 비교
- [ ] 반복 seed 실험
- [ ] CMA-ES 소개
- [ ] CMA-ES 실험
- [ ] Random Search vs BO vs CMA-ES 비교
- [ ] 각 section 학습 질문 포함
- [ ] 결과에 대한 해석 포함
- [ ] 과도한 library abstraction 없이 핵심 알고리즘이 보임

---

# 14. 최종 목표

Notebook을 다 보고 나면 학습자가 아래 문장을 스스로 설명할 수 있어야 한다.

> Gaussian Process는 지금까지 관측한 데이터를 이용해 black-box objective의 posterior mean과 uncertainty를 제공한다.

> Bayesian Optimization은 이 surrogate model과 acquisition function을 이용해 다음으로 평가할 가장 가치 있는 위치를 선택한다.

> Random Search는 이전 결과를 활용하지 않지만, BO는 모든 관측 결과를 다음 실험 설계에 활용한다.

> CMA-ES는 objective 자체를 surrogate로 모델링하지 않고, 좋은 해가 존재할 가능성이 높은 search distribution을 반복적으로 업데이트한다.

> 따라서 BO와 EA/CMA-ES는 어느 한쪽이 항상 우월한 것이 아니라, evaluation budget과 search space 특성에 따라 적합성이 달라진다.

이 Notebook의 최우선 목표는 **BO의 수식을 외우는 것보다, BO가 왜 필요한지 그리고 실제로 어떻게 동작하는지를 체험하게 만드는 것**이다.
