# Bayesian Optimization 실습

Bayesian Optimization의 핵심 아이디어를 1차원 예제로 살펴보는 한국어 튜토리얼입니다. Gaussian Process를 직접 구현하고, 획득 함수와 탐색 전략이 최적화 과정에 미치는 영향을 노트북에서 시각화합니다.

## 다루는 내용

- Random Search와 Gaussian Process 회귀
- Expected Improvement (EI), Probability of Improvement (PI), Lower Confidence Bound (LCB)
- Bayesian Optimization, Random Search, CMA-ES 비교
- 길이 척도, 관측 노이즈, 초기 샘플 수, 탐색 파라미터의 영향

## 실행 환경

프로젝트는 Python 3.12와 `llm` conda 환경에서 확인했습니다.

```bash
conda activate llm
cd /Users/gunhee/myspace/01_study/06_bayesian_optim
python -m pip install -r requirements.txt
```

`bayesian_optimization_tutorial.ipynb`를 Jupyter 또는 VS Code에서 열고 커널로 **Python (llm)**을 선택하세요. 노트북은 실행 중 `figures/`에 시각화 파일을 생성합니다.

## 파일 구성

- `bayesian_optimization_tutorial.ipynb`: 설명, 코드, 실험 결과를 담은 튜토리얼 노트북
- `bayesian_optimization_notebook_spec.md`: 노트북의 상세 설계 문서
- `_build/`: 노트북을 조립하는 소스 코드
- `figures/`: 노트북에서 사용하는 시각화 결과
- `requirements.txt`: Python 패키지 목록

노트북을 `_build/`의 소스로 다시 생성하려면 프로젝트 루트에서 실행하세요.

```bash
python _build/build.py
```
