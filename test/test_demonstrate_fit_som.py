import pytest

from examples.demonstrate_fit_som import demonstrate_second_order_model_fitting
import matplotlib.pyplot as plt


@pytest.mark.slow
def test_demonstrate_second_order_model_fitting(monkeypatch):
    # With this expression we override the matplotlib.pyplot.show method with a
    # lambda expression returning None but only for this one test.
    monkeypatch.setattr(plt, "show", lambda: None, raising=True)
    demonstrate_second_order_model_fitting()
