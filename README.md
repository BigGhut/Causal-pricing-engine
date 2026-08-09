# 🎯 Causal Pricing & Uplift Experimentation Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: CausalML / EconML](https://img.shields.io/badge/ML-CausalML%20%7C%20EconML-orange.svg)](https://github.com/uber/causalml)

Сервис причинно-следственного анализа (**Causal ML**), **Uplift-моделирования** и продвинутого A/B-тестирования для решения задач динамического ценообразования, расчета персональных скидок и оптимизации Surge-надбавок.

---

## 🎯 Цели и Задачи Проекта

В отличие от классического ML, который предсказывает абсолютную конверсию $P(\text{Conversion} | \text{Features}, \text{Treatment})$, данный движок фокусируется на **приращенном эффекте (Uplift)**:
$$\tau(x) = \mathbb{E}[Y^{(1)} - Y^{(0)} \mid X = x]$$

Где:
- $Y^{(1)}$ — результат при назначении воздействия (например, предоставление скидки 10% или Surge +150₽).
- $Y^{(0)}$ — результат без воздействия (базовый тариф).
- $\tau(x)$ — чистый прирост конверсии/прибыли (Individual Treatment Effect, ITE).

---

## 📐 Архитектура Проекта

```
causal-pricing-engine/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   └── config.yaml
├── data/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── causal/
│   │   ├── __init__.py
│   │   ├── uplift_models.py      # S-Learner, T-Learner, X-Learner, Uplift Random Forest
│   │   └── dml_engine.py         # Double Machine Learning (EconML) & Synthetic Control
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── switchback_splitter.py # Пространственно-временной сплиттер для A/B тестов
│   │   └── power_analysis.py     # Оценка стат. мощности и бутстрап
│   └── api/
│       ├── __init__.py
│       └── main.py               # FastAPI сервис выдачи оптимального Treatment
└── tests/
    ├── __init__.py
    └── test_uplift.py
```

---

## 🔬 Ключевой Функционал (Roadmap)

### 1. Uplift-моделирование (Causal Tree / Meta-Learners)
- **T-Learner & X-Learner:** Оценка чистого приращенного дохода от воздействия.
- **Сегментация пользователей по квадрантам:**
  - *Persuadables (Колеблющиеся):* Купят только с воздействием ($\tau(x) > 0$) — **Целевая группа!**
  - *Sure Things (Уверенные):* Купят в любом случае ($\tau(x) \approx 0$) — **Не тратим бюджет.**
  - *Lost Causes (Безнадежные):* Не купят в любом случае ($\tau(x) \approx 0$) — **Не тратим бюджет.**
  - *Sleeping Dogs (Раздражаемые):* Воздействие ухудшает конверсию ($\tau(x) < 0$) — **Исключаем.**

### 2. Double Machine Learning (DML)
- Избавление от смещения при нерандомизированных данных (Confounding Bias).
- Использование моделей CatBoost / LightGBM для ортогонализации признаков и воздействия.

### 3. Advanced Experimentation Framework
- **Switchback A/B Splitter:** Пространственно-временное разделение для маркетплейсов и сервисов такси.
- **Synthetic Control:** Оценка причинно-следственного эффекта ценовых изменений на исторических данных без проведения честных A/B тестов.
- **Variance Reduction (CUPED / CUPAC):** Снижение дисперсии метрик для ускорения экспериментов.

---

## 🛠️ Запуск и Разработка

```bash
# 1. Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Установка зависимостей
pip install -r requirements.txt

# 3. Запуск тестов
pytest tests/
```
