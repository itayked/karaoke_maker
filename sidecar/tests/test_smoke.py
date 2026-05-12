"""Smoke test — ensures the package imports cleanly."""

import karaoke
import karaoke.models
import karaoke.pipeline


def test_imports():
    assert karaoke is not None
    assert karaoke.models is not None
    assert karaoke.pipeline is not None
