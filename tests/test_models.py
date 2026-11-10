"""The three saved models: sensible probabilities, and they respond to the match the right way."""
import joblib
import numpy as np
import pytest

from leverage import concentration, win_probability

MODELS = {name: joblib.load(f"models/{name}.joblib") for name in ["logistic", "lightgbm", "mlp"]}


def wp(model, runs_needed, balls_remaining, wickets):
    n = np.broadcast(runs_needed, balls_remaining, wickets).size

    def full(v):
        return np.broadcast_to(v, n).astype(float)

    return win_probability(model, full(runs_needed), full(balls_remaining), full(wickets),
                           full(180), full(170), full(175))


@pytest.mark.parametrize("name", MODELS)
def test_probabilities_are_probabilities(name):
    p = wp(MODELS[name], np.arange(1, 181, 7), 60, 6)
    assert ((p >= 0) & (p <= 1)).all()


@pytest.mark.parametrize("name", MODELS)
def test_needing_more_runs_never_helps(name):
    p = wp(MODELS[name], np.arange(10, 150, 10), 60, 6)
    assert (np.diff(p) <= 1e-9).all()


@pytest.mark.parametrize("name", MODELS)
def test_more_wickets_in_hand_never_hurts(name):
    p = wp(MODELS[name], 60, 48, np.arange(1, 11))
    assert (np.diff(p) >= -1e-9).all()


def test_finished_chases_have_known_results():
    model = MODELS["logistic"]
    assert wp(model, 0, 10, 5)[0] == 1.0      # target reached
    assert wp(model, 5, 0, 5)[0] == 0.0       # out of balls
    assert wp(model, 5, 10, 0)[0] == 0.0      # all out
    assert wp(model, 1, 0, 5)[0] == 0.5       # scores level: a tie


def test_concentration():
    assert concentration(np.array([5.0, 3.0, 1.0, 1.0]), 0.25) == pytest.approx(0.5)
    assert concentration(np.ones(10), 0.10) == pytest.approx(0.1)
