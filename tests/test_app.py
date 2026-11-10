"""Both pages of the Streamlit app render from the committed data without errors."""
from streamlit.testing.v1 import AppTest

from conftest import ROOT


def test_win_probability_page():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "IPL win probability model"


def test_leverage_page():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.switch_page("views/leverage.py").run()
    assert not at.exception
    assert at.title[0].value == "Leverage"
