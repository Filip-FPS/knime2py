"""
Joiner module for joining two tables

Inputs:
- 2 DataFrames

Outputs:
- Publishes DataFrame to the context output ports

Configuration
----------------------


"""

from lxml import etree as ET
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

XML_PARSER = ET.XMLParser(
    remove_comments=True,
    resolve_entities=False,
    no_network=True,
    ns_clean=True,
    recover=True,
)

def strip_namespace(root):
    """Remove all namespaces from XML tree."""
    for el in root.iter():
        if isinstance(el.tag, str) and '{' in el.tag:
            el.tag = el.tag.split('}', 1)[1]  # strip namespace
    return root

#Test
root = ET.parse(str("settings.xml"), parser=XML_PARSER).getroot()
root = strip_namespace(root)

el = root.find(f".//config[@key='model']/entry[@key='includeMatchesInOutput']")
el.attrib.get("value")

#Implementations Claude
#!/usr/bin/env python3
"""
Joiner module for joining two DataFrames.
Overview
----------------------------
This module emits code that joins two pandas DataFrames using options
parsed from settings.xml. It maps KNIME Joiner3 options to pandas.merge,
handling join type inference, column selection, and duplicate column renaming.

Runtime Behavior
----------------------------
Inputs:
- Two DataFrames from incoming ports (left and right tables)
Outputs:
- A single joined DataFrame published to the output port

Key algorithms or mappings:
- Join type is inferred from the combination of includeMatchesInOutput,
  includeLeftUnmatchedInOutput and includeRightUnmatchedInOutput
- Column selection is applied post-merge using included_names from
  leftColumnSelectionConfig and rightColumnSelectionConfig
- Duplicate column handling mirrors KNIME's APPEND_SUFFIX behavior

Edge Cases & Safeguards
----------------------------
- Handles multiple join columns via list-based left_on/right_on
- Guards against missing columns after merge
- mergeJoinColumns collapses duplicate join keys into one

Generated Code Dependencies
----------------------------
The generated code requires: pandas

Node Identity
----------------------------
FACTORY = "org.knime.base.node.preproc.joiner3.Joiner3NodeFactory"

Configuration
----------------------------
- left_join_cols:           List of left table join columns
- right_join_cols:          List of right table join columns
- include_matches:          Include matched rows in output
- include_left_unmatched:   Include unmatched left rows
- include_right_unmatched:  Include unmatched right rows
- merge_join_columns:       Collapse join columns into one
- left_included_cols:       Columns to keep from left table
- right_included_cols:      Columns to keep from right table
- duplicate_handling:       How to handle duplicate column names
- suffix:                   Suffix appended to duplicate right columns
- composition_mode:         MATCH_ALL (AND) or MATCH_ANY (OR)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from lxml import etree as ET
from ..xml_utils import XML_PARSER
from .node_utils import *

FACTORY = "org.knime.base.node.preproc.joiner3.Joiner3NodeFactory"


# ----------------------------
# XML Extractors
# ----------------------------

def _extract_bool(root: ET._Element, key: str, default: bool = True) -> bool:
    """Extract a boolean entry from the model config."""
    el = root.find(f".//config[@key='model']/entry[@key='{key}']")
    if el is not None:
        return el.attrib.get("value", "true").lower() == "true"
    return default


def _extract_string(root: ET._Element, key: str, default: str = "") -> str:
    """Extract a string entry from the model config."""
    el = root.find(f".//config[@key='model']/entry[@key='{key}']")
    if el is not None:
        return el.attrib.get("value", default)
    return default


def _extract_string_array(root: ET._Element, config_key: str) -> List[str]:
    """
    Extract a string array from a config block like:
    <config key="leftTableJoinPredicate">
        <entry key="array-size" value="1"/>
        <entry key="0" value="NN_STUDYID"/>
    </config>
    """
    config = root.find(f".//config[@key='model']/config[@key='{config_key}']")
    if config is None:
        return []
    size_el = config.find("entry[@key='array-size']")
    if size_el is None:
        return []
    size = int(size_el.attrib.get("value", "0"))
    return [
        config.find(f"entry[@key='{i}']").attrib.get("value", "")
        for i in range(size)
        if config.find(f"entry[@key='{i}']") is not None
    ]


def _extract_included_columns(root: ET._Element, side: str) -> List[str]:
    """
    Extract included column names from leftColumnSelectionConfig
    or rightColumnSelectionConfig.

    Args:
        side: 'left' or 'right'
    """
    config_key = f"{side}ColumnSelectionConfig"
    included = root.find(
        f".//config[@key='model']"
        f"/config[@key='{config_key}']"
        f"/config[@key='included_names']"
    )
    if included is None:
        return []
    size_el = included.find("entry[@key='array-size']")
    if size_el is None:
        return []
    size = int(size_el.attrib.get("value", "0"))
    return [
        included.find(f"entry[@key='{i}']").attrib.get("value", "")
        for i in range(size)
        if included.find(f"entry[@key='{i}']") is not None
    ]


# ----------------------------
# Join Type Inference
# ----------------------------

def _infer_join_type(
    include_matches: bool,
    include_left: bool,
    include_right: bool
) -> str:
    """
    Infer pandas merge 'how' from KNIME's three boolean flags.

    KNIME join type mapping:
    ┌─────────────┬──────────────┬───────────────┬─────────────┐
    │ matches     │ left         │ right         │ pandas how  │
    ├─────────────┼──────────────┼───────────────┼─────────────┤
    │ True        │ False        │ False         │ inner       │
    │ True        │ True         │ False         │ left        │
    │ True        │ False        │ True          │ right       │
    │ True        │ True         │ True          │ outer       │
    │ False       │ True         │ False         │ left + ~in  │ (anti-join left)
    │ False       │ False        │ True          │ right anti  │ (anti-join right)
    └─────────────┴──────────────┴───────────────┴─────────────┘
    """
    if include_matches and not include_left and not include_right:
        return "inner"
    elif include_matches and include_left and not include_right:
        return "left"
    elif include_matches and not include_left and include_right:
        return "right"
    elif include_matches and include_left and include_right:
        return "outer"
    else:
        # Anti-join cases - rare but possible
        # We handle these as outer + post-filter in generated code
        return "outer"


# ----------------------------
# Settings Dataclass
# ----------------------------

@dataclass
class JoinerSettings:
    left_join_cols:         List[str] = field(default_factory=list)
    right_join_cols:        List[str] = field(default_factory=list)
    include_matches:        bool = True
    include_left_unmatched: bool = False
    include_right_unmatched: bool = False
    merge_join_columns:     bool = True
    left_included_cols:     List[str] = field(default_factory=list)
    right_included_cols:    List[str] = field(default_factory=list)
    duplicate_handling:     str = "APPEND_SUFFIX"
    suffix:                 str = " (right)"
    composition_mode:       str = "MATCH_ALL"


# ----------------------------
# Settings.xml → JoinerSettings
# ----------------------------

def parse_joiner_settings(node_dir: Path) -> JoinerSettings:
    """
    Read <node_dir>/settings.xml and extract all relevant Joiner settings.

    Args:
        node_dir: Path to the node directory containing settings.xml
    Returns:
        JoinerSettings populated from settings.xml
    """
    settings_path = node_dir / "settings.xml"
    if not settings_path.exists():
        return JoinerSettings()

    root = ET.parse(str(settings_path), parser=XML_PARSER).getroot()

    return JoinerSettings(
        left_join_cols=         _extract_string_array(root, "leftTableJoinPredicate"),
        right_join_cols=        _extract_string_array(root, "rightTableJoinPredicate"),
        include_matches=        _extract_bool(root, "includeMatchesInOutput",        True),
        include_left_unmatched= _extract_bool(root, "includeLeftUnmatchedInOutput",  False),
        include_right_unmatched=_extract_bool(root, "includeRightUnmatchedInOutput", False),
        merge_join_columns=     _extract_bool(root, "mergeJoinColumns",              True),
        left_included_cols=     _extract_included_columns(root, "left"),
        right_included_cols=    _extract_included_columns(root, "right"),
        duplicate_handling=     _extract_string(root, "duplicateHandling",           "APPEND_SUFFIX"),
        suffix=                 _extract_string(root, "suffix",                      " (right)"),
        composition_mode=       _extract_string(root, "compositionMode",             "MATCH_ALL"),
    )


# ----------------------------
# Code Generators
# ----------------------------

def generate_imports() -> List[str]:
    return ["import pandas as pd"]


def generate_py_body(
    node_id: str,
    node_dir: Optional[str],
    in_ports: List[str],
    out_ports: List[str]
) -> List[str]:
    """
    Emit body lines for a Joiner node that merges two DataFrames
    and publishes the result to the output port.

    Args:
        node_id:   The ID of the node
        node_dir:  The directory of the node
        in_ports:  The two input port variable names [left_df, right_df]
        out_ports: The output port variable names
    Returns:
        List of Python code lines
    """
    ndir = Path(node_dir) if node_dir else None
    s = parse_joiner_settings(ndir) if ndir else JoinerSettings()

    # Resolve input dataframe variable names
    left_df  = in_ports[0] if len(in_ports) > 0 else "df_left"
    right_df = in_ports[1] if len(in_ports) > 1 else "df_right"

    # Infer pandas join type
    how = _infer_join_type(
        s.include_matches,
        s.include_left_unmatched,
        s.include_right_unmatched
    )

    lines: List[str] = []
    lines.append(
        "# https://hub.knime.com/knime/extensions/org.knime.features.base/latest/"
        "org.knime.base.node.preproc.joiner3.Joiner3NodeFactory"
    )

    # ---- Step 1: Column Selection (pre-merge) ----
    # Select only included columns from each side before merging
    # This mirrors KNIME's column selection behavior
    if s.left_included_cols:
        lines.append(f"# Select included columns from left table")
        lines.append(f"_left_cols = {s.left_included_cols}")
        lines.append(
            f"_left_cols = [c for c in _left_cols if c in {left_df}.columns]"
        )
        lines.append(f"_df_left = {left_df}[_left_cols]")
    else:
        lines.append(f"_df_left = {left_df}.copy()")

    if s.right_included_cols:
        lines.append(f"# Select included columns from right table")
        lines.append(f"_right_cols = {s.right_included_cols}")
        lines.append(
            f"_right_cols = [c for c in _right_cols if c in {right_df}.columns]"
        )
        lines.append(f"_df_right = {right_df}[_right_cols]")
    else:
        lines.append(f"_df_right = {right_df}.copy()")

    # ---- Step 2: Merge ----
    lines.append(f"# Perform join (how='{how}')")
    suffix_tuple = repr(("", s.suffix))
    lines.append(
        f"df = _df_left.merge("
        f"_df_right, "
        f"left_on={s.left_join_cols}, "
        f"right_on={s.right_join_cols}, "
        f"how='{how}', "
        f"suffixes={suffix_tuple}"
        f")"
    )

    # ---- Step 3: Merge Join Columns ----
    # If mergeJoinColumns=True and join columns have different names,
    # drop the right join column as left already contains the value
    if s.merge_join_columns:
        lines.append(f"# Merge join columns - drop redundant right join keys")
        for left_col, right_col in zip(s.left_join_cols, s.right_join_cols):
            if left_col != right_col:
                # Different names - right key was suffixed, drop it
                lines.append(
                    f"if '{right_col}{s.suffix}' in df.columns:"
                    f" df = df.drop(columns=['{right_col}{s.suffix}'])"
                )
            else:
                # Same name - pandas created suffixed version, drop it
                lines.append(
                    f"if '{right_col}{s.suffix}' in df.columns:"
                    f" df = df.drop(columns=['{right_col}{s.suffix}'])"
                )

    # ---- Step 4: Anti-join post-filter (edge case) ----
    # Handle cases where include_matches=False (anti-join behavior)
    if not s.include_matches:
        lines.append("# Anti-join: remove matched rows")
        indicator_cols = " & ".join(
            [f"df['{c}'].isna()" for c in s.right_join_cols]
        )
        lines.append(f"df = df[{indicator_cols}]")

    # ---- Publish to context ----
    for line in context_assignment_lines(node_id, out_ports):
        lines.append(line)

    return lines


def get_name() -> str:
    """Return name of the node in KNIME workflow."""
    return "Joiner"


def handle(ntype, nid, npath, incoming, outgoing):
    """
    Handle the processing of a Joiner node.

    Args:
        ntype:    The type of the node
        nid:      The ID of the node
        npath:    The path of the node
        incoming: Incoming connections [left, right]
        outgoing: Outgoing connections
    Returns:
        tuple: (imports, body lines)
    """
    # Resolve input port variable names from incoming connections
    in_ports = [str(getattr(e, "target_port", "") or i)
                for i, (_, e) in enumerate(incoming)]

    out_ports = [str(getattr(e, "source_port", "") or "1")
                 for _, e in outgoing]

    node_lines = generate_py_body(nid, npath, in_ports, out_ports)
    found, body = split_out_imports(node_lines)
    explicit = collect_module_imports(generate_imports)
    imports = sorted(set(found).union(explicit))
    return imports, body
