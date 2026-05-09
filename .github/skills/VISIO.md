# VSDX Diagram Generation Skill (YAML → Visio .vsdx)

This skill generates a **Visio .vsdx** containing **native editable basic shapes** (e.g., rectangles) and **dynamic connectors that stay glued** when shapes move in Visio.

- Input: YAML (diagram-as-code)
- Output: `.vsdx`
- Layout: **auto-layout**, deterministic (ties resolved by YAML `nodes:` list order)

---

## YAML format (v1)

```yaml
version: 1
document:
  title: "System overview"     # optional

layout:
  engine: "layered"            # required (v1 supports: layered)
  direction: "LR"              # LR or TB (default LR)
  node_size:
    w: 2.2                     # inches (default 2.2)
    h: 1.0                     # inches (default 1.0)
  spacing:
    x: 1.0                     # inches (default 1.0)
    y: 0.8                     # inches (default 0.8)
  page_margin: 0.5             # inches (default 0.5)

pages:
  - name: "Main"               # required
    nodes:
      - id: "app"              # required; unique per page
        text: "App"            # optional; default = id
        kind: "rect"           # optional; v1 supports: rect (default rect)

      - id: "api"
        text: "API"

      - id: "db"
        text: "Database"

    edges:
      - from: "app"            # required; must exist in nodes
        to: "api"              # required; must exist in nodes
        label: "calls"         # optional
        kind: "dynamic"        # optional; v1 supports: dynamic (default dynamic)

      - from: "api"
        to: "db"
        label: "reads/writes"
```

---

## Deterministic auto-layout rules (v1)

### 1) Graph model

For each page:

- `nodes`: ordered list of node definitions.
- `edges`: ordered list of directed connections.

### 2) Layer assignment (engine: layered)

Goal: assign each node an integer **layer index** (0, 1, 2, …) so edges generally go left→right (LR) or top→bottom (TB).

1. Build adjacency lists from edges (`from` → `to`).
2. Initialize `layer[node] = 0` for all nodes.
3. Iterate nodes in **YAML node list order**, repeatedly relax edges to push targets to the right:
   - Repeat up to `N` passes (where `N = number of nodes`):
     - For each edge `(u → v)` in edge list order:
       - `layer[v] = max(layer[v], layer[u] + 1)`
4. This produces stable layers even with cycles; cycles will still converge to a consistent result under the bounded-pass rule, with choices implicitly driven by YAML node order and edge order.

**Tie-breaking:** When multiple nodes can occupy the same layer, their within-layer ordering is the YAML `nodes:` order.

### 3) Coordinate placement

Let:

- `w = layout.node_size.w`
- `h = layout.node_size.h`
- `sx = layout.spacing.x`
- `sy = layout.spacing.y`
- `m = layout.page_margin`

Group nodes by layer in increasing layer index.

**Direction LR (left-to-right):**

- For each layer `L`:
  - Column center X:
    - `x = m + L*(w + sx) + (w/2)`
  - For each node in that layer, in YAML node order, with vertical index `i = 0..`:
    - `y = m + i*(h + sy) + (h/2)`

**Direction TB (top-to-bottom):**

- Swap roles of X/Y:
  - `y = m + L*(h + sy) + (h/2)`
  - `x = m + i*(w + sx) + (w/2)`

These `(x,y,w,h)` map directly to Visio shape cells:

- `PinX = x`, `PinY = y`, `Width = w`, `Height = h`

---

## VSDX output requirements

### Nodes (basic shapes)

- Each node is emitted as a **native Visio shape** (v1: rectangle).
- Text is written into the shape’s text element so it is editable in Visio.
- Nodes must be movable/resizable.

### Edges (dynamic glued connectors)

- Each edge is emitted as a **dynamic connector shape**.
- The connector’s **begin/end** are **glued** to the `from` and `to` shapes, so:
  - moving a node keeps the connector attached
  - rerouting is handled by Visio’s dynamic connector behavior
- If `label` is present, it is emitted as connector text.

---

## Skill interface (CLI suggestion)

```bash
vsdxgen diagram.yaml --template template.vstx --out output.vsdx
```

- `diagram.yaml`: input YAML in the format above
- `template.vstx` (or `.vsdx`): a minimal Visio-authored template that contains:
  - at least one rectangle shape (for copying node shape structure)
  - at least one dynamic connector shape (for copying connector + glue structure)
- `output.vsdx`: generated result

**Why a template?**
Visio’s `.vsdx` is a ZIP of XML parts with specific relationships. Using a known-good template avoids having to author every relationship from scratch. The generator should:

1. unzip template
2. duplicate/modify page XML to include all nodes/edges
3. update IDs and glue/connect references
4. zip back into `.vsdx`

---

## Validation rules (v1)

- `version` must be `1`
- Each page must have:
  - `name`
  - `nodes` list with unique `id`s
- Each edge must reference existing node ids (`from`, `to`)
- Unknown keys should be ignored (forward compatibility)

---

## Notes / Non-goals (v1)

- No Visio stencils/masters beyond what’s embedded in the template
- No BPMN/UML “official” shapes; represent them as labeled boxes + connectors
- No crossing minimization beyond deterministic ordering
- Styling (fill/stroke/fonts) optional; can be added in v2
