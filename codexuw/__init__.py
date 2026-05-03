"""Clean UW options income pipeline.

This package is intentionally separate from ``uwos``.  It reads the raw
Unusual Whales exports directly, builds a bounded candidate set, then validates
candidate spreads against live Schwab chains when credentials are available.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
