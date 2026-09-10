import pandas as pd

from ai_data_analyst.llm.validator import validate_route
from ai_data_analyst.llm.router import RouteDecision


def create_test_dataframe():
    return pd.DataFrame({
        "order_date": pd.to_datetime([
            "2025-01-01",
            "2025-02-01",
        ]),
        "region": ["West", "East"],
        "sales": [1000, 2000],
        "profit": [300, 500],
        "discount": [0.10, 0.20],
    })


def test_valid_python_route():
    df = create_test_dataframe()

    decision = RouteDecision(
        tool="python",
        operation="mean",
        value_column="profit",
    )

    validated = validate_route(decision, df)

    assert validated is True


def test_invalid_tool():
    df = create_test_dataframe()

    decision = RouteDecision(
        tool="database",
        operation="mean",
        value_column="profit",
    )

    try:
        validate_route(decision, df)
        assert False
    except ValueError:
        assert True


def test_invalid_column():
    df = create_test_dataframe()

    decision = RouteDecision(
        tool="python",
        operation="mean",
        value_column="revenue",
    )

    try:
        validate_route(decision, df)
        assert False
    except ValueError:
        assert True


def test_valid_visualization_route():
    df = create_test_dataframe()

    decision = RouteDecision(
        tool="visualization",
        chart_type="bar",
        operation="sum",
        group_column="region",
        value_column="sales",
    )

    validated = validate_route(decision, df)

    assert validated is True


def test_invalid_chart_type():
    df = create_test_dataframe()

    decision = RouteDecision(
        tool="visualization",
        chart_type="histogram",
        operation="sum",
        group_column="region",
        value_column="sales",
    )

    try:
        validate_route(decision, df)
        assert False
    except ValueError:
        assert True


def test_valid_line_chart():
    df = create_test_dataframe()

    decision = RouteDecision(
        tool="visualization",
        chart_type="line",
        operation="sum",
        date_column="order_date",
        value_column="sales",
        time_granularity="month",
    )

    validated = validate_route(decision, df)

    assert validated is True

def test_valid_rag_route():
    df = create_test_dataframe()

    decision = RouteDecision(
        tool="rag"
    )

    validated = validate_route(decision, df)

    assert validated is True