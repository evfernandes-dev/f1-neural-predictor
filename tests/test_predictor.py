import numpy as np
import pandas as pd
import pytest

from predictor import (
    FEATURES,
    _clean_name,
    _next_race_features,
    _winner_hit_rate,
    clean_and_engineer,
    predict_next_race,
    train_model,
)


def make_raw_data(races: int = 8) -> pd.DataFrame:
    """Cria resultados fictícios: em corridas ímpares Alex vence, nas pares Bruno."""
    rows = []
    drivers = [("Alex Albon", "Williams"), ("Bruno Costa", "Sauber")]
    for race in range(1, races + 1):
        for number, (name, team) in enumerate(drivers, start=1):
            alex_wins = race % 2 == 1
            winner = 1 if alex_wins else 2
            position = 1 if number == winner else 2
            rows.append({
                "ano": 2026,
                "corrida": race,
                "data": f"2026-03-{race:02d}",
                "piloto_id": number,
                "piloto": name,
                "equipe": team,
                "posicao": position,
                "abandono_api": False,
            })
    return pd.DataFrame(rows)


def test_clean_and_engineer_uses_only_previous_races():
    df = clean_and_engineer(make_raw_data(), recent_races=3)

    assert set(df["corrida"]) == {4, 5, 6, 7, 8}
    assert "prova" in df.columns

    alex = df[(df["piloto"] == "Alex Albon") & (df["corrida"] == 4)].iloc[0]
    # Posições de Alex nas corridas 1, 2 e 3: 1, 2 e 1.
    assert alex["media_posicao_5"] == pytest.approx((1 + 2 + 1) / 3)
    assert alex["vitorias_5"] == pytest.approx(2 / 3)


def test_next_race_features_include_the_latest_race():
    df = clean_and_engineer(make_raw_data(), recent_races=3)
    features = _next_race_features(df, recent_races=3).set_index("piloto")
    alex = features.loc["Alex Albon"]

    # As três últimas corridas (6, 7 e 8) têm posições 2, 1 e 2.
    assert alex["media_posicao_5"] == pytest.approx((2 + 1 + 2) / 3)
    assert alex["vitorias_5"] == pytest.approx(1 / 3)
    assert alex["equipe"] == "Williams"
    assert set(FEATURES).issubset(features.columns)


def test_predict_next_race_normalizes_probabilities():
    class FakeModel:
        def predict_proba(self, features):
            return np.array([[0.5, 0.5]] * len(features))

    df = clean_and_engineer(make_raw_data(), recent_races=3)
    result = predict_next_race(df, FakeModel(), recent_races=3)

    assert len(result) == 2
    assert result["probabilidade"].sum() == pytest.approx(100.0)


def test_winner_hit_rate_counts_one_pick_per_race():
    test = pd.DataFrame({
        "prova": ["2026-1", "2026-1", "2026-2", "2026-2"],
        "venceu": [1, 0, 0, 1],
        "prob": [0.7, 0.3, 0.9, 0.1],
    })

    assert _winner_hit_rate(test, "prob") == pytest.approx(0.5)


def test_train_model_returns_race_level_metrics():
    rows = []
    drivers = [(1, "A", "Time A"), (2, "B", "Time B"), (3, "C", "Time C"), (4, "D", "Time D")]
    for race in range(1, 16):
        winner = race % 4
        for number, name, team in drivers:
            rows.append({
                "ano": 2026,
                "corrida": race,
                "data": f"2026-04-{race:02d}",
                "piloto_id": number,
                "piloto": name,
                "equipe": team,
                "posicao": 1 if number == winner else 2 + number,
                "abandono_api": False,
            })

    df = clean_and_engineer(pd.DataFrame(rows), recent_races=3)
    _, metrics = train_model(df)

    assert 0 <= metrics["top1"] <= 1
    assert 0 <= metrics["baseline"] <= 1
    assert metrics["corridas_teste"] > 0
    assert metrics["treino"] > 0
    assert metrics["teste"] > 0


def test_clean_name_normalizes_spaces_and_case():
    assert _clean_name("  ALEX   ALBON ") == "Alex Albon"
