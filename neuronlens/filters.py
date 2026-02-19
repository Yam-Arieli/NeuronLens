"""Filter builder helpers for NeuronLens.

Instead of writing raw filter dicts, use these functions:

    from neuronlens.filters import eq, gt, and_

    precomputed_filters = [
        eq("label", "cat"),
        eq("label", "dog"),
        and_(gt("age", 18), lt("age", 65)),
        in_("source", ["train", "val"]),
    ]

Every function returns a plain dict compatible with NeuronLens's filter spec,
so the two styles are fully interchangeable.
"""

from typing import Any, List


def eq(column: str, value: Any) -> dict:
    """column == value"""
    return {"column": column, "op": "eq", "value": value}


def ne(column: str, value: Any) -> dict:
    """column != value"""
    return {"column": column, "op": "ne", "value": value}


def lt(column: str, value: Any) -> dict:
    """column < value"""
    return {"column": column, "op": "lt", "value": value}


def le(column: str, value: Any) -> dict:
    """column <= value"""
    return {"column": column, "op": "le", "value": value}


def gt(column: str, value: Any) -> dict:
    """column > value"""
    return {"column": column, "op": "gt", "value": value}


def ge(column: str, value: Any) -> dict:
    """column >= value"""
    return {"column": column, "op": "ge", "value": value}


def in_(column: str, values: List[Any]) -> dict:
    """column value is in the given list"""
    return {"column": column, "op": "in", "value": list(values)}


def not_in(column: str, values: List[Any]) -> dict:
    """column value is NOT in the given list"""
    return {"column": column, "op": "not_in", "value": list(values)}


def and_(*conditions: dict) -> dict:
    """All of the given conditions must hold (logical AND)."""
    return {"and": list(conditions)}
