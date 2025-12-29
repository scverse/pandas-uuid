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

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))


# -- Project information -----------------------------------------------------

project = "pandas-uuid"  # matches GitHub repo name and distribution name
info = metadata(project)
author = info["Author-Email"].split('"')[1]
copyright = f"{datetime.now(tz=UTC):%Y}, {author}."  # noqa: A001
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
repository_url = urls["Source"]

# The full version, including alpha/beta/rc tags
release = info["Version"]

templates_path = ["_templates"]
nitpicky = True  # Warn about broken links

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "scverse",
    "github_repo": project,
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_tabs.tabs",
    "sphinxext.opengraph",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

autosummary_generate = False
autodoc_member_order = "bysource"
autodoc_default_options = {
    "inherited-members": False,
    "show-inheritance": True,
}
default_role = "py:obj"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "pyarrow": ("https://arrow.apache.org/docs", None),
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]


# Fix up undocumented types using the `resolve_from_map` extension
type_map = {
    ("py", "class", "NAType"): ("attr", "pandas.NA"),
    ("py", "class", "np.random.Generator"): ("class", "numpy.random.Generator"),
    **{
        ("py", "class", f"pa.{cls}"): ("class", f"pyarrow.{cls}")
        for cls in [
            "Array",
            "ChunkedArray",
            "UuidArray",
            "UuidScalar",
            "UuidType",
            "Scalar",
        ]
    },
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
# html_css_files = ["css/custom.css"]  # noqa: ERA001

html_title = project

html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,
    "path_to_docs": "docs/",
    "navigation_with_keys": False,
}

pygments_style = "default"
