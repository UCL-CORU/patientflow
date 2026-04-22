from graphviz import Digraph

g = Digraph("patientflow", format="png")
g.attr(
    rankdir="LR",
    bgcolor="white",
    fontname="Helvetica",
    splines="ortho",
    ranksep="1.2",
    nodesep="0.8",
)
g.attr(
    "node",
    fontname="Helvetica",
    fontsize="16",
    style="filled",
    shape="box",
    width="2.8",
    height="0.9",
)
g.attr("edge", fontname="Helvetica", fontsize="9")

PATIENT = {"fillcolor": "#dbeafe", "color": "#3b82f6", "fontcolor": "#1e3a5f"}
SPEC = {"fillcolor": "#dcfce7", "color": "#16a34a", "fontcolor": "#14532d"}
AGG = {"fillcolor": "#fef9c3", "color": "#ca8a04", "fontcolor": "#78350f"}
SKIP = {
    "fillcolor": "#f3f4f6",
    "color": "#9ca3af",
    "fontcolor": "#6b7280",
    "style": "filled,dashed",
}

# Row 1: M1 and M10 only
with g.subgraph() as s:
    s.attr(rank="same")
    s.node("M1", "1: P(admission after ED)\nML Classifier · XGBoost", **PATIENT)
    s.node("M10", "10: P(discharge from specialty)\nML Classifier · XGBoost", **PATIENT)

# Row 2: M2, M4a, M4b
with g.subgraph() as s:
    s.attr(rank="same")
    s.node("M2", "2: P(admission to subspecialty)\nProbabilistic mapping", **PATIENT)
    s.node(
        "M4a", "4a: P(admission in window) aspirational\nAspirational curve", **PATIENT
    )
    s.node("M4b", "4b: P(admission in window) observed\nSurvival curve", **PATIENT)

# Invisible edges to push M2, M4a, M4b below M1/M10
g.edge("M1", "M2", style="invis")
g.edge("M1", "M4a", style="invis")
g.edge("M10", "M4b", style="invis")

# Specialty-level: current ED patients
g.node("M3", "3: # admissions, current ED/SDEC patients\nAggregation: 1 × 2", **SPEC)
g.node(
    "M5a", "5a: # beds needed in window — aspirational\nAggregation: 1 × 2 × 4a", **SPEC
)
g.node(
    "M5b",
    "5b: # admissions in window — observed\nAggregation: 1 × 2 × 4b  [skip at UCLH]",
    **SKIP,
)

# Yet-to-arrive
g.node("M6", "6: Arrival rates (yet-to-arrive)\nHistorical 15-min rates", **SPEC)
g.node("M7a", "7a: # beds needed — aspirational\nTime-varying Poisson: 6 × 4a", **SPEC)
g.node(
    "M7b",
    "7b: # admissions in window — observed\nTime-varying Poisson: 6 × 4b  [skip at UCLH]",
    **SKIP,
)

# Other flows
g.node(
    "M8", "8: # emergency non-ED arrivals\nTime-varying Poisson · historical", **SPEC
)
g.node("M9", "9: # elective arrivals\nTime-varying Poisson · historical", **SPEC)

# Discharge & transfer
g.node(
    "M11",
    "11: # emergency discharges from specialty\nAggregation of 10 over emergency inpatients",
    **SPEC,
)
g.node(
    "M12",
    "12: # elective discharges from specialty\nAggregation of 10 over elective inpatients",
    **SPEC,
)
g.node("M13", "13: Transition matrix\nP(transfer to B | in specialty A)", **SPEC)
g.node(
    "M14", "14: # emergency transfers into specialty\nConvolve 11 thinned by 13", **SPEC
)
g.node(
    "M15", "15: # elective transfers into specialty\nConvolve 12 thinned by 13", **SPEC
)

# Aggregations
g.node("M16", "16: All incoming emergency\nConvolution: 5a + 7a + 8 + 14", **AGG)
g.node("M17", "17: All incoming elective\nConvolution: 9 + 15", **AGG)
g.node("M18", "18: Net flow — emergency\nConvolution: 16 − 11", **AGG)
g.node("M19", "19: Net flow — elective\nConvolution: 17 − 12", **AGG)

# Real edges
g.edges(
    [
        ("M1", "M3"),
        ("M2", "M3"),
        ("M1", "M5a"),
        ("M2", "M5a"),
        ("M4a", "M5a"),
        ("M1", "M5b"),
        ("M2", "M5b"),
        ("M4b", "M5b"),
        ("M6", "M7a"),
        ("M4a", "M7a"),
        ("M6", "M7b"),
        ("M4b", "M7b"),
        ("M10", "M11"),
        ("M10", "M12"),
        ("M11", "M14"),
        ("M13", "M14"),
        ("M12", "M15"),
        ("M13", "M15"),
        ("M5a", "M16"),
        ("M7a", "M16"),
        ("M8", "M16"),
        ("M14", "M16"),
        ("M9", "M17"),
        ("M15", "M17"),
        ("M16", "M18"),
        ("M11", "M18"),
        ("M17", "M19"),
        ("M12", "M19"),
    ]
)

g.render("evaluation_flowchart", cleanup=True)
print("Done")
