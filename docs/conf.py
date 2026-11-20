# SPDX-License-Identifier: MPL-2.0
"""Configuration file for the Sphinx documentation builder.

See <https://www.sphinx-doc.org/en/master/usage/configuration.html>.
"""

# -- Path setup --------------------------------------------------------------
from __future__ import annotations

import sys
from datetime import UTC, datetime
from importlib.metadata import metadata
from pathlib import Path

from scverse_doc.registry import intersphinx

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))


# -- Project information -----------------------------------------------------

project = "pandas-uuid"  # matches GitHub repo name and distribution name
info = metadata(project)
author = info["Author-Email"].split('"')[1]
copyright = f"{datetime.now(tz=UTC):%Y}, {author}."  # noqa: A001
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
source_repository = urls["Source"]

# The full version, including alpha/beta/rc tags
release = info["Version"]

# -- General configuration ---------------------------------------------------

extensions = [
    "scverse_doc",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

# Deviations from `scverse_doc.config.DEFAULTS`
autosummary_generate = False
autodoc_member_order = "bysource"
autodoc_default_options = {
    "inherited-members": False,
    "show-inheritance": True,
}
default_role = "py:obj"

intersphinx_mapping = intersphinx() | {
    "pyarrow": ("https://arrow.apache.org/docs", None),
}

xref_overrides = {
    "NAType": ("py:data", "pandas.NA"),
    "pandas.api.typing.NAType": ("py:data", "pandas.NA"),
    **{
        f"pa.{cls}": f"pyarrow.{cls}"
        for cls in ["ChunkedArray", "UuidArray", "UuidScalar"]
    },
}
