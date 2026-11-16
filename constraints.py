"""
constraints.py — Harde constraints op acties, onafhankelijk van reward.

Een learner die op completion optimaliseert kan leren om site-voorwaarden
te schenden (bv. Jina sturen op Cloudflare-sites). Constraints blokkeren
deze acties voordat de policy ze overweegt — geen penalty, gewoon weg.

Usage:
    from constraints import apply_scrape_constraints
    allowed = apply_scrape_constraints(context_features, all_actions)
    # policy.choose() krijgt alleen 'allowed' te zien
"""

from __future__ import annotations

from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Scrape constraints
# ---------------------------------------------------------------------------

# Actions that require antibot capability
_ANTIBOT_ACTIONS = {"use_webclaw_scrape", "use_browser"}

# Actions that CANNOT handle Cloudflare / antibot
_NO_ANTIBOT_ACTIONS = {"use_jina_reader"}

# Actions that CAN handle heavy JavaScript
_JS_CAPABLE = {"use_browser", "use_webclaw_scrape"}

# Actions that are too expensive for simple sites
_HEAVY_ACTIONS = {"use_browser"}


def apply_scrape_constraints(
    context_features: Dict[str, float],
    available_actions: List[str],
) -> List[str]:
    """Filter actions based on site characteristics.

    Args:
        context_features: Dict with keys like cloudflare, js_heavy,
            requires_antibot, content_len_norm, etc.
        available_actions: Full list of candidate action IDs.

    Returns:
        Filtered list with only allowed actions.
    """
    allowed = list(available_actions)

    cloudflare = context_features.get("cloudflare", 0.0)
    js_heavy = context_features.get("js_heavy", 0.0)
    requires_antibot = context_features.get("requires_antibot", 0.0)
    content_len = context_features.get("content_len_norm", 0.0)
    site_known = context_features.get("site_known", 0.0)

    # --- Hard rule 1: Antibot/Cloudflare sites --------------------------------
    # Jina faalt op Cloudflare-sites. Blokkeer Jina.
    if cloudflare >= 0.5 or requires_antibot >= 0.5:
        for bad in _NO_ANTIBOT_ACTIONS:
            if bad in allowed:
                allowed.remove(bad)

    # --- Hard rule 2: Heavy JS sites ------------------------------------------
    # Alleen Browser of WebClaw kunnen JS-heavy sites aan.
    if js_heavy >= 0.8:
        for a in list(allowed):
            if a not in _JS_CAPABLE:
                allowed.remove(a)

    # --- Hard rule 3: Simple HTML / API ---------------------------------------
    # Browser is overkill voor simpele sites. Blokkeer Browser.
    if cloudflare < 0.5 and js_heavy < 0.5 and content_len < 0.2:
        if "use_browser" in allowed and len(allowed) > 1:
            allowed.remove("use_browser")

    # --- Hard rule 4: Bekende, simpele WP-sites -------------------------------
    # WebClaw is onnodig duur voor simpele bekende WP-sites.
    # Jina werkt hier prima. Blokkeer Browser.
    if site_known >= 0.5 and cloudflare < 0.5 and js_heavy < 0.5:
        if "use_browser" in allowed and len(allowed) > 1:
            allowed.remove("use_browser")

    # Fallback: never return empty list
    if not allowed:
        allowed = list(available_actions)

    return allowed


# ---------------------------------------------------------------------------
# Podcast workflow constraints
# ---------------------------------------------------------------------------

# Publish actions that actually write to live systems
_LIVE_PUBLISH_ACTIONS = {"wordpress", "youtube", "spotify", "all"}

# Draft/safe actions
_SAFE_PUBLISH_ACTIONS = {"draft"}


def apply_podcast_constraints(
    node_id: str,
    action_ids: List[str],
    context: Dict,
) -> List[str]:
    """Filter podcast actions based on workflow state.

    Args:
        node_id: The workflow node (e.g., 'publish', 'audio').
        action_ids: Full list of candidate action IDs for this node.
        context: Workflow context with flags like _is_dry_run, _is_approved.

    Returns:
        Filtered list with only allowed actions.
    """
    allowed = list(action_ids)

    # --- Publish node: only draft unless explicitly approved ------------------
    if node_id == "publish":
        is_approved = context.get("_is_approved", False)
        is_dry_run = context.get("_is_dry_run", False)

        if not is_approved or is_dry_run:
            # Remove all live publish actions
            for bad in _LIVE_PUBLISH_ACTIONS:
                if bad in allowed:
                    allowed.remove(bad)

    # --- Audio node: skip only if explicitly allowed --------------------------
    if node_id == "audio":
        # Keep skip as a valid option (it's a deliberate choice)
        pass

    # Fallback
    if not allowed:
        allowed = list(action_ids)

    return allowed


__all__ = [
    "apply_scrape_constraints",
    "apply_podcast_constraints",
]