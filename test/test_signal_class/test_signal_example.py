import pytest
from matplotlib import pyplot as plt

from PyDynamic.examples.working_with_signals import demonstrate_signal


@pytest.mark.slow
def test_signal_example(monkeypatch):
    # Test executability of the demonstrate_signal example.
    # With this expression we override the matplotlib.pyplot.show method with a
    # lambda expression returning None but only for this one test.
    monkeypatch.setattr(plt, "show", lambda: None, raising=True)
    demonstrate_signal()
