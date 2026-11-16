"""
env_setup.py — Bootstrap environment for agent-learning store backend.

Sets AGENT_LEARNING_STORE_BACKEND and AGENT_LEARNING_LOCAL_STORE_DIR
to their defaults if not already set, so get_default_store() returns
a LocalFileStore instead of InMemoryStore.

Import at the TOP of any script that uses get_default_store():

    import env_setup  # noqa: F401
    from agent_learning.storage.cosmos import get_default_store
"""

import os

_STORE_BACKEND = os.environ.get("AGENT_LEARNING_STORE_BACKEND")
if _STORE_BACKEND is None:
    os.environ["AGENT_LEARNING_STORE_BACKEND"] = "local"

_STORE_DIR = os.environ.get("AGENT_LEARNING_LOCAL_STORE_DIR")
if _STORE_DIR is None:
    _default_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "store"
    )
    os.environ["AGENT_LEARNING_LOCAL_STORE_DIR"] = _default_dir