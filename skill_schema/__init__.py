"""Versioned JSON Schema for shareable skills (workflow, node, episode, reward).

Stdlib-only validator so both this repo and a knowledge-base repo can
check the same shapes without extra dependencies.
"""

from .validate import (
    SCHEMA_VERSION,
    SchemaError,
    is_full_episode,
    load_schema,
    schema_dir,
    validate,
)

__all__ = [
    "SCHEMA_VERSION",
    "SchemaError",
    "is_full_episode",
    "load_schema",
    "schema_dir",
    "validate",
]
