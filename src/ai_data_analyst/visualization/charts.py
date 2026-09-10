import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def create_bar_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str = "Bar Chart"
):
    """
    Create a bar chart from two columns.
    """

    if x_column not in df.columns:
        raise ValueError(f"Column not found: {x_column}")

    if y_column not in df.columns:
        raise ValueError(f"Column not found: {y_column}")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        df[x_column].astype(str),
        df[y_column]
    )

    ax.set_title(title)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)

    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig

def create_line_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str = "Line Chart"
):
    """
    Create a line chart from two columns.
    """

    if x_column not in df.columns:
        raise ValueError(f"Column not found: {x_column}")

    if y_column not in df.columns:
        raise ValueError(f"Column not found: {y_column}")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        df[x_column],
        df[y_column],
        marker="o"
    )

    ax.set_title(title)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)

    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def create_scatter_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str = "Scatter Plot"
):
    """
    Create a scatter plot showing the relationship
    between two numerical columns.
    """

    if x_column not in df.columns:
        raise ValueError(f"Column not found: {x_column}")

    if y_column not in df.columns:
        raise ValueError(f"Column not found: {y_column}")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        df[x_column],
        df[y_column],
        alpha=0.6
    )

    ax.set_title(title)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)

    plt.tight_layout()

    return fig

def create_pie_chart(
    df: pd.DataFrame,
    label_column: str,
    value_column: str,
    title: str = "Pie Chart"
):
    """
    Create a pie chart showing the proportion
    of a numerical value across categories.
    """

    if label_column not in df.columns:
        raise ValueError(f"Column not found: {label_column}")

    if value_column not in df.columns:
        raise ValueError(f"Column not found: {value_column}")

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.pie(
        df[value_column],
        labels=df[label_column],
        autopct="%1.1f%%"
    )

    ax.set_title(title)

    return fig