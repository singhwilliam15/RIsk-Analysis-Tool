"""Formatting helpers and colours shared across the UI."""

MODEL_COLORS = ["#3182CE", "#DD6B20", "#38A169", "#D53F8C", "#805AD5", "#4FD1C5", "#F56565", "#ECC94B"]


def pct_label(cl: float) -> str:
    return f"{cl * 100:g}%"
