"""Isolated, cache-only CodexSwing v0.5 research lane.

This package deliberately has no dependency on the live ORATS or Schwab
adapters.  It can describe and plan research from already cached records, but
it cannot spend API quota or place an order.
"""

from codexswing.v5.spec import V5ResearchSpec

__all__ = ["V5ResearchSpec"]

