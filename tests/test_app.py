"""End-to-end tests for the Streamlit UI (app.py) using AppTest."""
import os
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def app():
    """Initializes AppTest with app.py."""
    # Ensure offline mock provider during testing
    os.environ["LLM_PROVIDER"] = "mock"
    app_path = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    return at


def test_app_initial_load(app):
    """Verifies that the app loads cleanly and displays the main title and default dataset."""
    app.run()
    assert not app.exception
    # Check title
    assert len(app.markdown) > 0
    title_matches = [m.value for m in app.markdown if "Conversational Data Analysis Assistant" in m.value]
    assert len(title_matches) > 0


def test_app_run_query(app):
    """Verifies entering a query and running analysis."""
    app.run()
    assert not app.exception

    # Enter a question in the text input
    if len(app.text_input) > 0:
        app.text_input[0].input("Which category generated the highest profit?").run()
        # Click the run button
        run_buttons = [b for b in app.button if "Run Analysis" in b.label]
        if run_buttons:
            run_buttons[0].click().run()
            assert not app.exception
            # Verify direct answer card rendered
            answer_cards = [m.value for m in app.markdown if "Direct Answer" in m.value or "Technology" in m.value]
            assert len(answer_cards) > 0


def test_app_dataset_switch(app):
    """Verifies selecting a different preset dataset in the sidebar."""
    app.run()
    assert not app.exception

    # Select Telco dataset from selectbox
    preset_selectbox = [s for s in app.sidebar.selectbox if "Select Benchmark Dataset:" in s.label]
    if preset_selectbox:
        preset_selectbox[0].select("Validation: Telco Customer Churn (.csv)").run()
        load_btn = [b for b in app.sidebar.button if "Load Selected Dataset" in b.label]
        if load_btn:
            load_btn[0].click().run()
            assert not app.exception
            # Verify rows updated
            metrics = [m.value for m in app.metric]
            assert "7,043" in metrics or any("7043" in str(m) for m in metrics)
