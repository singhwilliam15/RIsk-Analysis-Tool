"""The context object that carries inputs and results from one UI section to the next."""

from types import SimpleNamespace


def new_context() -> SimpleNamespace:
    return SimpleNamespace()


def export(ctx: SimpleNamespace, values: dict) -> None:
    """Store every value a section computed on the context, so later sections can use it."""
    for name, value in values.items():
        if name != "ctx":
            setattr(ctx, name, value)
