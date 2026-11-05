# F1 Neural Predictor

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Streamlit app that estimates each driver's chance of winning the next Formula 1 race. It downloads race results, builds recent-form features and trains a small neural network with scikit-learn.

> Educational project. The probabilities are not betting advice.

![F1 Neural Predictor](docs/screenshot.png)

## What the app does

- Shows the win probability for every active driver.
- Checks the model by race: did the top pick win?
- Compares the model with a simple average-points baseline.
- Lets you download the table as CSV.

## Running

Requires Python 3.12+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

On Linux or macOS, use `source .venv/bin/activate`.

Open `http://localhost:8501`, choose the period in the sidebar and click **Treinar e prever**. The first run downloads the data and may take a few minutes.

## How it works

1. **Download:** completed race results from OpenF1, with Jolpica as a fallback.
2. **Recent form:** for each driver, average the last races (position, wins, podiums, top 5, retirements and points).
3. **Train:** a small neural network learns which drivers won, using only earlier races.
4. **Predict:** the model scores each driver and converts the scores to percentages that add up to 100%.

The model trains on 80% of races and tests on the last 20%. Races are kept whole, so the same race never appears in both sets.

## Structure

```text
├── app.py                    Streamlit app
├── predictor.py              Data, model and prediction
├── docs/screenshot.png
├── .streamlit/config.toml    Dark theme
├── requirements.txt
└── LICENSE
```

## Limitations

- Free OpenF1 data starts mainly in 2023.
- The model does not use qualifying, circuit, weather or car updates.
- The app interface and code comments are in Portuguese.

## Stack

Python, pandas, NumPy, scikit-learn, Streamlit, Plotly, OpenF1 and Jolpica.
