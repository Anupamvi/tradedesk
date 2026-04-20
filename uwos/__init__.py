"""UWOS package."""


def report_main(*args, **kwargs):
    from .report import main

    return main(*args, **kwargs)


__all__ = ["report_main"]
