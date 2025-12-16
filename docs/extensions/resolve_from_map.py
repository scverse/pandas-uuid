# SPDX-License-Identifier: MPL-2.0
"""Resolve failed type lookup from `type_map`."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from sphinx.util import logging

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Final

    from docutils.nodes import TextElement, reference
    from sphinx.addnodes import pending_xref
    from sphinx.application import Sphinx
    from sphinx.environment import BuildEnvironment


LOGGER: Final = logging.getLogger("fast_array_utils")


def resolve_type_aliases(  # noqa: PLR0917
    app: Sphinx, env: BuildEnvironment, node: pending_xref, contnode: TextElement
) -> reference | None:
    """Fixup undocumented types."""
    type_map = cast(
        "Mapping[tuple[str, str, str], tuple[str, str]]", app.config.type_map
    )

    k = node["refdomain"], node["reftype"], node["reftarget"]
    if k not in type_map:
        return None

    typ, target = type_map[k]

    from sphinx.ext.intersphinx import resolve_reference_any_inventory

    node["reftype"] = typ
    node["reftarget"] = target
    ref = resolve_reference_any_inventory(
        env=env, honor_disabled_refs=False, node=node, contnode=contnode
    )
    if ref is None:
        msg = f"Could not resolve {typ} {target} (from {k})"
        LOGGER.warning(msg, type="ref")
        return ref
    return ref


def setup(app: Sphinx) -> None:
    """Register custom hooks."""
    app.add_config_value("type_map", {}, "env")
    app.connect("missing-reference", resolve_type_aliases, priority=800)
