"""
scrape_features.py — Context feature schema voor scrape-strategy TaskPolicy.

Feature vector (8 dimensies):
  [cloudflare, js_heavy, is_wordpress, is_elementor,
   content_len_norm, requires_antibot, site_known, has_structured_data]

Elke feature is genormaliseerd naar ruwweg [0, 1] voor stabiele learning.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# Feature index mapping — sorteert alfabetisch voor determinisme
FEATURE_NAMES: List[str] = sorted([
    "cloudflare",        # 1.0 = site achter Cloudflare
    "js_heavy",          # 1.0 = veel JavaScript (Elementor, React, etc.)
    "is_wordpress",      # 1.0 = WordPress site
    "is_elementor",      # 1.0 = Elementor (bovenop WordPress)
    "content_len_norm",  # geschatte lengte in KB, gecapt op 100, gedeeld door 100
    "requires_antibot",  # 1.0 = antibot/anti-scrape nodig
    "site_known",        # 1.0 = deze site hebben we vaker gescraped
    "has_structured_data", # 1.0 = JSON-LD / microdata aanwezig
])

FEATURE_DIM = len(FEATURE_NAMES)

_INDEX = {name: idx for idx, name in enumerate(FEATURE_NAMES)}


def make_phi(**kwargs: float) -> Dict[str, float]:
    """Bouw context feature dict uit keyword args.

    Voorbeeld:
        phi = make_phi(cloudflare=1.0, js_heavy=1.0, is_wordpress=0.0, ...)
    """
    phi: Dict[str, float] = {}
    for name in FEATURE_NAMES:
        phi[name] = float(kwargs.get(name, 0.0))
    return phi


def phi_to_vector(phi: Dict[str, float]) -> List[float]:
    """Zet context feature dict om naar numpy-vriendelijke lijst (feature order)."""
    return [phi.get(name, 0.0) for name in FEATURE_NAMES]


def describe_site(url: str, html: str = "", headers: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    """Analyseer een URL/html en produceer context features.

    Dit is een heuristische classifier — later te vervangen door
    een aparte TaskPolicy of LLM-call voor feature extractie.
    """
    import re
    url_lower = url.lower()
    html_lower = html.lower()

    cloudflare = 1.0 if any(
        s in headers.get("server", "") if headers else False
        for s in ["cloudflare", "cloudflare"]
    ) else 0.0

    # Detecteer CF in body als headers niet beschikbaar zijn
    if not headers:
        cloudflare = 1.0 if "cloudflare" in html_lower[:5000] else 0.0

    is_wp = 1.0 if any(
        s in html_lower for s in [
            "wp-content", "wp-includes", "wordpress", "astra", "elementor"
        ]
    ) else 0.0

    is_elem = 1.0 if "elementor" in html_lower else 0.0

    js_heavy = 1.0 if (
        html_lower.count("<script") > 10
        or "react" in html_lower
        or "vue" in html_lower
    ) else (0.5 if html_lower.count("<script") > 3 else 0.0)

    content_len_norm = min(len(html) / 100000, 1.0)

    requires_antibot = cloudflare  # CF = antibot nodig

    site_known = 0.0  # wordt later gezet op basis van eerdere episodes

    has_structured_data = 1.0 if any(
        s in html_lower for s in [
            "application/ld+json", "microdata", "itemprop",
            "og:title", "twitter:card"
        ]
    ) else 0.0

    return make_phi(
        cloudflare=cloudflare,
        js_heavy=js_heavy,
        is_wordpress=is_wp,
        is_elementor=is_elem,
        content_len_norm=content_len_norm,
        requires_antibot=requires_antibot,
        site_known=site_known,
        has_structured_data=has_structured_data,
    )


# === Korte aliassen voor veel voorkomende sitetypes ===

SITE_PROFILES = {
    "wordpress_astra": make_phi(
        cloudflare=0.0, js_heavy=0.3, is_wordpress=1.0,
        is_elementor=0.0, content_len_norm=0.3, requires_antibot=0.0,
        site_known=1.0, has_structured_data=0.5,
    ),
    "wordpress_elementor": make_phi(
        cloudflare=0.0, js_heavy=0.7, is_wordpress=1.0,
        is_elementor=1.0, content_len_norm=0.5, requires_antibot=0.0,
        site_known=1.0, has_structured_data=0.5,
    ),
    "cloudflare_wp": make_phi(
        cloudflare=1.0, js_heavy=0.5, is_wordpress=1.0,
        is_elementor=0.0, content_len_norm=0.3, requires_antibot=1.0,
        site_known=0.5, has_structured_data=0.3,
    ),
    "cloudflare_js_heavy": make_phi(
        cloudflare=1.0, js_heavy=1.0, is_wordpress=0.0,
        is_elementor=0.0, content_len_norm=0.5, requires_antibot=1.0,
        site_known=0.0, has_structured_data=0.0,
    ),
    "simple_html": make_phi(
        cloudflare=0.0, js_heavy=0.0, is_wordpress=0.0,
        is_elementor=0.0, content_len_norm=0.1, requires_antibot=0.0,
        site_known=0.5, has_structured_data=0.0,
    ),
    "api_endpoint": make_phi(
        cloudflare=0.0, js_heavy=0.0, is_wordpress=0.0,
        is_elementor=0.0, content_len_norm=0.01, requires_antibot=0.0,
        site_known=0.0, has_structured_data=1.0,
    ),
}


__all__ = [
    "FEATURE_NAMES", "FEATURE_DIM",
    "make_phi", "phi_to_vector", "describe_site", "SITE_PROFILES",
]