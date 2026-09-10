"""
inkml_parser.py

Parses CROHME .inkml files into:
  - a list of strokes, where each stroke is a list of (x, y) float points
  - the ground-truth LaTeX string (from the <annotation type="truth"> tag)

InkML is XML. A typical file looks like:

<ink xmlns="http://www.w3.org/2003/InkML">
  <annotation type="truth"> \\frac { a } { b } </annotation>
  <trace id="0">
    7240 6562, 7257 6565, 7275 6561, ...
  </trace>
  <trace id="1"> ... </trace>
  ...
</ink>

Some files have extra fields in each coordinate group (x y time, or x y f),
separated by spaces after the comma-separated points, e.g. "7240 6562 0".
We only ever take the first two numbers of each point.
"""

import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import xml.etree.ElementTree as ET

# InkML files declare this namespace; ElementTree needs it to find tags.
NS = {"ink": "http://www.w3.org/2003/InkML"}

Point = Tuple[float, float]
Stroke = List[Point]


@dataclass
class InkExpression:
    file_path: str
    strokes: List[Stroke]
    latex: str


def normalize_truth(raw: str) -> str:
    """
    Clean a raw <annotation type="truth"> string.

    CROHME_full_v2 combines several sub-corpora with inconsistent
    conventions, so we canonicalize:
      - strip surrounding math delimiters ($ ... $ or $$ ... $$)
      - drop \\left / \\right size hints
      - collapse whitespace
    Brace canonicalization happens later, in tokenize_latex.
    """
    s = raw.strip()
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2]
    elif s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    s = s.replace(r"\left", " ").replace(r"\right", " ")
    s = s.replace(r"\!", " ").replace(r"\,", " ").replace(r"\;", " ").replace(r"\ ", " ")
    return " ".join(s.split())


def _parse_trace_text(trace_text: str) -> Stroke:
    """Turn a <trace> element's text content into a list of (x, y) points."""
    points: Stroke = []
    # Points are comma-separated; each point is space-separated numbers.
    for raw_point in trace_text.strip().split(","):
        raw_point = raw_point.strip()
        if not raw_point:
            continue
        parts = raw_point.split()
        # Some CROHME files append a time or pressure value; ignore anything past x, y.
        x, y = float(parts[0]), float(parts[1])
        points.append((x, y))
    return points


def parse_inkml_file(path: str) -> InkExpression:
    """Parse a single .inkml file into an InkExpression."""
    tree = ET.parse(path)
    root = tree.getroot()

    latex = ""
    # Most releases namespace-qualify <annotation>; a few files in
    # CROHME_full_v2 don't, so try both.
    for annotation in list(root.findall("ink:annotation", NS)) + list(root.findall("annotation")):
        if annotation.attrib.get("type") == "truth":
            latex = normalize_truth(annotation.text or "")
            break

    strokes: List[Stroke] = []
    for trace_el in root.findall("ink:trace", NS):
        if trace_el.text and trace_el.text.strip():
            strokes.append(_parse_trace_text(trace_el.text))

    return InkExpression(file_path=path, strokes=strokes, latex=latex)


def find_inkml_files(root_dir: str) -> List[str]:
    """Recursively find every .inkml file under root_dir."""
    matches = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".inkml"):
                matches.append(os.path.join(dirpath, fname))
    return matches


_TOKEN_RE = re.compile(r"\\[a-zA-Z]+|\\.|[^\s]")


def _reduce_single_atom_braces(tokens: List[str]) -> List[str]:
    """
    Canonical form: a brace group wrapping exactly one atom loses its braces.

      x ^ { 2 }         -> x ^ 2
      a _ { n }         -> a _ n
      \\frac { 1 } { 2 } -> \\frac 1 2
      x ^ { n - 1 }     -> unchanged (multi-atom group keeps its braces)

    This collapses the biggest source of spurious label variance across the
    CROHME_full_v2 sub-corpora (some write `x^2`, some `x^{2}`, some `x ^ 2`).
    Applied repeatedly until stable so nested cases settle.
    """
    changed = True
    while changed:
        changed = False
        out: List[str] = []
        i = 0
        while i < len(tokens):
            if (tokens[i] == "{" and i + 2 < len(tokens)
                    and tokens[i + 2] == "}"
                    and tokens[i + 1] not in ("{", "}")):
                out.append(tokens[i + 1])
                i += 3
                changed = True
            else:
                out.append(tokens[i])
                i += 1
        tokens = out
    return tokens


def tokenize_latex(latex: str) -> List[str]:
    """
    Split a ground-truth LaTeX string into a canonical token list.

    Handles both the whitespace-separated CROHME style ("\\frac { a } { b }")
    and the glued style ("\\frac{a}{b}", "x^2"), then applies
    single-atom-brace reduction so the two styles map to the same tokens.
    """
    latex = latex.strip()
    if not latex:
        return []
    tokens = _TOKEN_RE.findall(latex)
    return _reduce_single_atom_braces(tokens)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python inkml_parser.py <path_to_inkml_or_dir>")
        sys.exit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        files = find_inkml_files(target)
        print(f"Found {len(files)} .inkml files under {target}")
        if files:
            sample = parse_inkml_file(files[0])
            print(f"Sample file: {sample.file_path}")
            print(f"LaTeX: {sample.latex}")
            print(f"Tokens: {tokenize_latex(sample.latex)}")
            print(f"Num strokes: {len(sample.strokes)}")
    else:
        sample = parse_inkml_file(target)
        print(f"LaTeX: {sample.latex}")
        print(f"Tokens: {tokenize_latex(sample.latex)}")
        print(f"Num strokes: {len(sample.strokes)}")
