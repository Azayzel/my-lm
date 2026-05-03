"""vsdxgen — YAML diagram-as-code -> Visio .vsdx.

Implements the v1 spec at .github/skills/VISIO.md:

    python scripts/vsdxgen.py diagram.yaml \
        --template scripts/templates/vsdxgen_template.vstx \
        --out output.vsdx

The template is a Visio-authored .vsdx or .vstx that contains at
least one rectangle shape and one dynamic-connector shape on its
first page. We harvest their masters/styles, then regenerate the
page contents with all nodes + edges from the YAML.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape, quoteattr

import yaml

VISIO_NS = "http://schemas.microsoft.com/office/visio/2012/main"
ET.register_namespace("", VISIO_NS)
NS = {"v": VISIO_NS}


# ---------------------------------------------------------------------------
# YAML model + validation
# ---------------------------------------------------------------------------


@dataclass
class Node:
    id: str
    text: str
    kind: str = "rect"
    layer: int = 0
    layer_index: int = 0
    x: float = 0.0
    y: float = 0.0


@dataclass
class Edge:
    src: str
    dst: str
    label: str = ""
    kind: str = "dynamic"


@dataclass
class Page:
    name: str
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)


@dataclass
class Layout:
    direction: str = "LR"
    w: float = 2.2
    h: float = 1.0
    sx: float = 1.0
    sy: float = 0.8
    margin: float = 0.5


@dataclass
class Diagram:
    title: str
    layout: Layout
    pages: list[Page]


def load_diagram(path: Path) -> Diagram:
    with path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)

    if raw.get("version") != 1:
        raise ValueError("version must be 1")

    layout_raw = raw.get("layout") or {}
    if layout_raw.get("engine", "layered") != "layered":
        raise ValueError("layout.engine must be 'layered' (v1)")
    ns = layout_raw.get("node_size") or {}
    sp = layout_raw.get("spacing") or {}
    layout = Layout(
        direction=layout_raw.get("direction", "LR"),
        w=float(ns.get("w", 2.2)),
        h=float(ns.get("h", 1.0)),
        sx=float(sp.get("x", 1.0)),
        sy=float(sp.get("y", 0.8)),
        margin=float(layout_raw.get("page_margin", 0.5)),
    )
    if layout.direction not in ("LR", "TB"):
        raise ValueError("layout.direction must be LR or TB")

    pages: list[Page] = []
    for page_raw in raw.get("pages", []):
        if not page_raw.get("name"):
            raise ValueError("each page requires a name")
        nodes: list[Node] = []
        ids: set[str] = set()
        for n in page_raw.get("nodes", []):
            if "id" not in n:
                raise ValueError("each node requires an id")
            if n["id"] in ids:
                raise ValueError(f"duplicate node id: {n['id']}")
            ids.add(n["id"])
            nodes.append(
                Node(
                    id=n["id"],
                    text=n.get("text", n["id"]),
                    kind=n.get("kind", "rect"),
                )
            )
        edges: list[Edge] = []
        for e in page_raw.get("edges", []):
            if e["from"] not in ids or e["to"] not in ids:
                raise ValueError(
                    f"edge {e['from']}->{e['to']} references unknown node"
                )
            edges.append(
                Edge(
                    src=e["from"],
                    dst=e["to"],
                    label=e.get("label", ""),
                    kind=e.get("kind", "dynamic"),
                )
            )
        pages.append(Page(name=page_raw["name"], nodes=nodes, edges=edges))

    doc = raw.get("document") or {}
    return Diagram(title=doc.get("title", ""), layout=layout, pages=pages)


# ---------------------------------------------------------------------------
# Layered auto-layout (deterministic, per spec)
# ---------------------------------------------------------------------------


def assign_layers(page: Page) -> None:
    by_id = {n.id: n for n in page.nodes}
    adj: dict[str, list[str]] = defaultdict(list)
    for e in page.edges:
        adj[e.src].append(e.dst)

    for n in page.nodes:
        n.layer = 0

    n_passes = max(1, len(page.nodes))
    for _ in range(n_passes):
        changed = False
        for e in page.edges:
            u, v = by_id[e.src], by_id[e.dst]
            new = u.layer + 1
            if new > v.layer:
                v.layer = new
                changed = True
        if not changed:
            break

    by_layer: dict[int, list[Node]] = defaultdict(list)
    for n in page.nodes:
        by_layer[n.layer].append(n)
    for nodes in by_layer.values():
        for i, n in enumerate(nodes):
            n.layer_index = i


def place(page: Page, layout: Layout) -> tuple[float, float]:
    """Place nodes and return (page_width, page_height) in inches.

    Visio's Y axis is positive-up. To make YAML node order read top-to-bottom
    visually, we anchor idx=0 to the page's top edge.
    """
    assign_layers(page)
    if not page.nodes:
        return (8.5, 11.0)

    counts: dict[int, int] = defaultdict(int)
    for n in page.nodes:
        counts[n.layer] += 1
    max_layer = max(counts)
    max_per_layer = max(counts.values())

    w, h = layout.w, layout.h
    sx, sy = layout.sx, layout.sy
    m = layout.margin

    if layout.direction == "LR":
        page_w = (max_layer + 1) * (w + sx) - sx + 2 * m
        page_h = max_per_layer * (h + sy) - sy + 2 * m
    else:
        page_h = (max_layer + 1) * (h + sy) - sy + 2 * m
        page_w = max_per_layer * (w + sx) - sx + 2 * m

    for n in page.nodes:
        L, i = n.layer, n.layer_index
        if layout.direction == "LR":
            n.x = m + L * (w + sx) + w / 2
            # Y up: idx 0 at top of page, descending by index.
            n.y = page_h - m - i * (h + sy) - h / 2
        else:
            n.y = page_h - m - L * (h + sy) - h / 2
            n.x = m + i * (w + sx) + w / 2

    return page_w, page_h


# ---------------------------------------------------------------------------
# VSDX I/O
# ---------------------------------------------------------------------------


PAGE_PATH = "visio/pages/page1.xml"
PAGES_INDEX_PATH = "visio/pages/pages.xml"
MASTERS_PATH = "visio/masters/masters.xml"
CONTENT_TYPES_PATH = "[Content_Types].xml"
ROOT_RELS_PATH = "_rels/.rels"

# Visio OOXML content types — template <-> drawing.
TEMPLATE_TO_DRAWING_CT = {
    "application/vnd.ms-visio.template.main+xml":
        "application/vnd.ms-visio.drawing.main+xml",
    "application/vnd.ms-visio.template.macroEnabled.main+xml":
        "application/vnd.ms-visio.drawing.macroEnabled.main+xml",
}

RECT_KEYWORDS = ("rect", "square", "box")
CONN_KEYWORDS = ("dynamic connector", "connector", "line")


def find_template_masters(tmp_dir: Path) -> tuple[str, str]:
    """Return (rect_master_id, connector_master_id) from the template's
    masters.xml. Visio shape instances are often named 'Sheet.N' so we
    resolve types via the master index, not via page-level shape names.
    """
    masters_xml = tmp_dir / MASTERS_PATH
    if not masters_xml.exists():
        raise ValueError(
            f"template missing {MASTERS_PATH} — did you save from Visio? "
            "(an empty/blank file won't have masters until shapes are dropped)"
        )

    tree = ET.parse(masters_xml)
    available: list[tuple[str, str]] = []  # (id, name)
    rect_id = conn_id = None

    for master in tree.getroot().findall(".//v:Master", NS):
        mid = master.get("ID")
        name = (master.get("NameU") or master.get("Name") or "").strip()
        if not mid:
            continue
        available.append((mid, name or "<unnamed>"))
        low = name.lower()
        if rect_id is None and any(k in low for k in RECT_KEYWORDS):
            rect_id = mid
        if conn_id is None and any(k in low for k in CONN_KEYWORDS):
            conn_id = mid

    if not rect_id or not conn_id:
        listing = "\n  ".join(f"ID={i} NameU={n!r}" for i, n in available) or "(none)"
        missing = []
        if not rect_id:
            missing.append("rectangle (NameU containing 'rect'/'square'/'box')")
        if not conn_id:
            missing.append("connector (NameU containing 'connector'/'line')")
        raise ValueError(
            "template is missing required master(s): "
            + ", ".join(missing)
            + f"\n\nMasters found in template:\n  {listing}\n\n"
            "Open the template in Visio, drop a Rectangle and a Dynamic "
            "Connector onto Page-1, then re-save."
        )
    return rect_id, conn_id


def _fmt(v: float) -> str:
    """Format a float for Visio cell values: trim trailing zeros."""
    return f"{v:.6f}".rstrip("0").rstrip(".") or "0"


def _shape_xml(
    shape_id: int, master_id: str, name: str,
    x: float, y: float, w: float, h: float, text: str,
) -> str:
    n = quoteattr(name)
    parts = [
        f'<Shape ID="{shape_id}" NameU={n} Name={n} '
        f'Type="Shape" Master="{master_id}">',
        f'<Cell N="PinX" V="{_fmt(x)}"/>',
        f'<Cell N="PinY" V="{_fmt(y)}"/>',
        f'<Cell N="Width" V="{_fmt(w)}"/>',
        f'<Cell N="Height" V="{_fmt(h)}"/>',
        f'<Cell N="LocPinX" V="{_fmt(w / 2)}"/>',
        f'<Cell N="LocPinY" V="{_fmt(h / 2)}"/>',
    ]
    if text:
        parts.append(f"<Text>{escape(text)}</Text>")
    parts.append("</Shape>")
    return "".join(parts)


def _connector_xml(
    shape_id: int, master_id: str, src: Node, dst: Node, label: str,
) -> str:
    parts = [
        f'<Shape ID="{shape_id}" NameU="Dynamic connector" '
        f'Name="Dynamic connector" Type="Shape" Master="{master_id}">',
        f'<Cell N="BeginX" V="{_fmt(src.x)}"/>',
        f'<Cell N="BeginY" V="{_fmt(src.y)}"/>',
        f'<Cell N="EndX" V="{_fmt(dst.x)}"/>',
        f'<Cell N="EndY" V="{_fmt(dst.y)}"/>',
    ]
    if label:
        parts.append(f"<Text>{escape(label)}</Text>")
    parts.append("</Shape>")
    return "".join(parts)


def _connect_xml(connector_id: int, from_part: int, from_cell: str, target_id: int) -> str:
    return (
        f'<Connect FromSheet="{connector_id}" FromCell="{from_cell}" '
        f'FromPart="{from_part}" ToSheet="{target_id}" ToCell="PinX" ToPart="3"/>'
    )


PAGE_BODY_RE = re.compile(
    r"(<PageContents\b[^>]*>)(.*?)(</PageContents\s*>)",
    re.DOTALL,
)


def rewrite_page_xml(
    page_xml_path: Path,
    page: Page,
    rect_master: str,
    conn_master: str,
    w: float,
    h: float,
) -> None:
    """Replace the body of <PageContents> via string substitution so the
    template's original opening-tag attributes (xmlns, xmlns:r, xml:space)
    and the XML declaration are preserved byte-for-byte.
    """
    text = page_xml_path.read_text(encoding="utf-8")
    m = PAGE_BODY_RE.search(text)
    if not m:
        raise ValueError(
            f"{page_xml_path} does not contain a <PageContents> element"
        )

    shapes_xml: list[str] = []
    connects_xml: list[str] = []
    sid = 1
    node_to_id: dict[str, int] = {}

    for n in page.nodes:
        shapes_xml.append(
            _shape_xml(sid, rect_master, n.id, n.x, n.y, w, h, n.text)
        )
        node_to_id[n.id] = sid
        sid += 1

    for e in page.edges:
        src = next(n for n in page.nodes if n.id == e.src)
        dst = next(n for n in page.nodes if n.id == e.dst)
        shapes_xml.append(_connector_xml(sid, conn_master, src, dst, e.label))
        # FromPart 9 = BeginX, 12 = EndX (Visio connector-end constants).
        connects_xml.append(_connect_xml(sid, 9, "BeginX", node_to_id[e.src]))
        connects_xml.append(_connect_xml(sid, 12, "EndX", node_to_id[e.dst]))
        sid += 1

    body = (
        "<Shapes>" + "".join(shapes_xml) + "</Shapes>"
        + "<Connects>" + "".join(connects_xml) + "</Connects>"
    )
    new_text = text[: m.start(2)] + body + text[m.end(2) :]
    page_xml_path.write_text(new_text, encoding="utf-8")


PAGE_CELL_RE_TEMPLATE = (
    r'<Cell\s+N=(["\']){name}\1\s+V=(["\'])[^"\']*\2'
    r'(\s+U=(["\'])[^"\']*\4)?\s*/?>'
)


def fit_page_size(pages_xml_path: Path, width: float, height: float) -> None:
    """Update PageWidth/PageHeight cells in pages.xml to fit the layout.
    Visio's default page is 8.5x11 portrait — too small for most diagrams.
    """
    if not pages_xml_path.exists():
        return
    text = pages_xml_path.read_text(encoding="utf-8")
    for cell, val in (("PageWidth", width), ("PageHeight", height)):
        pattern = re.compile(
            PAGE_CELL_RE_TEMPLATE.format(name=cell),
            re.DOTALL,
        )
        replacement = f'<Cell N="{cell}" V="{_fmt(val)}"/>'
        text, n_subs = pattern.subn(replacement, text, count=1)
        if n_subs == 0:
            # No existing cell — inject one inside the first <PageSheet>.
            text = re.sub(
                r"(<PageSheet\b[^>]*>)",
                r"\1" + replacement,
                text, count=1,
            )
    pages_xml_path.write_text(text, encoding="utf-8")


def _convert_template_to_drawing(tmp_dir: Path, out_ext: str) -> None:
    """If the template was a .vstx and we're producing a .vsdx, rewrite the
    template content type/relationship to the drawing equivalent. Leaving
    them as 'template' makes Visio reject the file with an invalid-format
    error.
    """
    if out_ext.lower() != ".vsdx":
        return

    ct_path = tmp_dir / CONTENT_TYPES_PATH
    if ct_path.exists():
        ct_text = ct_path.read_text(encoding="utf-8")
        for old, new in TEMPLATE_TO_DRAWING_CT.items():
            ct_text = ct_text.replace(old, new)
        ct_path.write_text(ct_text, encoding="utf-8")


def _write_ooxml_zip(src_dir: Path, out_path: Path) -> None:
    """Write OOXML package with [Content_Types].xml first and stored
    (uncompressed). Some readers (Visio included) reject packages where
    [Content_Types].xml isn't the first entry.
    """
    files = [p for p in src_dir.rglob("*") if p.is_file()]
    files.sort(key=lambda p: p.relative_to(src_dir).as_posix())
    ctypes = src_dir / "[Content_Types].xml"
    if ctypes in files:
        files.remove(ctypes)
        files.insert(0, ctypes)

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            arcname = path.relative_to(src_dir).as_posix()
            compress = (
                zipfile.ZIP_STORED if arcname == "[Content_Types].xml"
                else zipfile.ZIP_DEFLATED
            )
            zf.write(path, arcname, compress_type=compress)


def generate(yaml_path: Path, template_path: Path, out_path: Path) -> None:
    if not template_path.exists():
        raise FileNotFoundError(
            f"template not found: {template_path}\n"
            "create one in Visio with a rectangle and a dynamic connector "
            "on Page-1, then save as .vsdx or .vstx."
        )
    if template_path.suffix.lower() not in (".vsdx", ".vstx"):
        raise ValueError(
            f"template must be .vsdx or .vstx, got {template_path.suffix}"
        )

    diagram = load_diagram(yaml_path)
    if not diagram.pages:
        raise ValueError("diagram has no pages")

    page = diagram.pages[0]
    page_w, page_h = place(page, diagram.layout)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(template_path, "r") as zf:
            zf.extractall(tmp_dir)

        page_xml_path = tmp_dir / PAGE_PATH
        if not page_xml_path.exists():
            raise ValueError(f"template missing {PAGE_PATH}")

        rect_master, conn_master = find_template_masters(tmp_dir)

        rewrite_page_xml(
            page_xml_path, page, rect_master, conn_master,
            diagram.layout.w, diagram.layout.h,
        )
        fit_page_size(tmp_dir / PAGES_INDEX_PATH, page_w, page_h)

        _convert_template_to_drawing(tmp_dir, out_path.suffix)

        if out_path.exists():
            out_path.unlink()
        _write_ooxml_zip(tmp_dir, out_path)

    print(f"wrote {out_path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("yaml", type=Path, help="input YAML diagram")
    p.add_argument("--template", type=Path, required=True, help="template .vsdx")
    p.add_argument("--out", type=Path, required=True, help="output .vsdx")
    args = p.parse_args(argv)

    try:
        generate(args.yaml, args.template, args.out)
    except Exception as exc:
        print(f"vsdxgen: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
