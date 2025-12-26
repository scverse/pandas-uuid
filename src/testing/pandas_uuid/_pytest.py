# SPDX-License-Identifier: MPL-2.0
"""Private pytest configuration that has to run early."""

from __future__ import annotations

import doctest
import os
import random
import re
from importlib.util import find_spec
from pathlib import Path

import pytest

INCLUDE_RE = re.compile(r"""\
\.\.\s+include::\s+(?P<path>.*)
\s{3,}:start-after:\s+(?P<start_after>.*)
\s{3,}:end-before:\s+(?P<end_before>.*)
""")

HAS_PYARROW = bool(find_spec("pyarrow"))


@pytest.fixture(autouse=True)
def _env(request: pytest.FixtureRequest) -> None:
    if isinstance(request.node, pytest.DoctestItem):
        if not HAS_PYARROW and any(
            "pyarrow" in example.source for example in request.node.dtest.examples
        ):
            pytest.skip("pyarrow not installed")
        request.getfixturevalue("doctest_env")


@pytest.fixture
def doctest_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set “seed” for UUID generation."""
    rng = random.Random(0)  # noqa: S311
    monkeypatch.setattr(os, "urandom", rng.randbytes)


def _allow_doctest_includes() -> None:
    gd = doctest.DocTestParser.get_doctest

    def get_doctest(  # noqa: PLR0917
        self: doctest.DocTestParser,
        string: str,
        globs: dict[str, str],
        name: str,
        filename: str | None,
        lineno: int | None,
    ) -> doctest.DocTest:
        if filename is not None:
            path = Path(filename)
            if path.name == "index.rst":  # sanity check
                assert INCLUDE_RE.search(string)  # noqa: S101
            for match in INCLUDE_RE.finditer(string):
                found = (path.parent / str(match["path"])).read_text()
                start, end = (
                    found.index(match[g]) for g in ("start_after", "end_before")
                )
                string = (
                    string[: match.start()] + found[start:end] + string[match.end() :]
                )

        return gd(self, string, globs, name, filename, lineno)

    doctest.DocTestParser.get_doctest = get_doctest


_allow_doctest_includes()
