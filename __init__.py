"""Repo-root Hermes plugin entry for `hermes plugins install owner/repo`."""

from hermes_plugin import PLUGIN_NAME, on_post_tool_call, on_pre_tool_call, register

__all__ = ["PLUGIN_NAME", "on_pre_tool_call", "on_post_tool_call", "register"]
