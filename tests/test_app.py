from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_starts_without_errors():
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.run()

    assert not app.exception
