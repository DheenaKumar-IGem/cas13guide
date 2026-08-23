"""Self-contained HTML report (FR8.2, FR8.4, NFR5, NFR6).

Produces one file with no external requests: inline CSS and JS, an SVG
accessibility track drawn from the data, and no CDN. That matters more than it
sounds -- a report that fetches a script at open time is a report that stops
working on an air-gapped lab machine, behind a proxy, or in three years.

What this is *not* is a web server. FR1.2's file upload and NFR2's "no
command-line knowledge" need one, and which framework to serve it with is a
decision for whoever deploys it. Everything a server would need to render is
here as a pure function of :meth:`DesignReport.to_dict`, so wrapping it is a
handful of lines rather than a rewrite.

NFR6 -- explaining the scores
-----------------------------
The glossary is generated from :data:`SCORE_GLOSSARY` rather than written into
the template, so a score that gains a column gains an explanation in the same
edit. A reviewer who has never seen this pipeline should be able to read a row
without reading the source, and in particular should be told the two things the
numbers do not say on their own: that a bundled calibration is synthetic, and
that predicted signal is relative.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

#: Column key -> (heading, plain-English explanation). NFR6.
SCORE_GLOSSARY: dict[str, tuple[str, str]] = {
    "rank": ("Rank", "Position after the composite weighted score. Rank 1 is the best overall trade-off, not the best on any single axis."),
    "verdict": ("Verdict", "PASS cleared every gate. REVIEW cleared the hard gates but something needs a human — see the tier. FAIL never reaches this table."),
    "review_tier": ("Tier", "R1 a borderline number. R2 a gap in the evidence, such as an off-target whose expression is unknown. R3 nobody validated this case — the model was asked outside its domain."),
    "spacer": ("Spacer", "The guide RNA sequence to order, 5'→3'. It is the reverse complement of the target site, because Cas13 binds its target as an antiparallel RNA:RNA duplex."),
    "score": ("Score", "Weighted sum of the normalised objectives. Useful for ordering, meaningless as an absolute number — the weights are a preference, not a measurement."),
    "required_coverage": ("Coverage", "Lower confidence bound on activity against the intended target, taken at the worst required target. Read low on purpose: a high point estimate with a wide interval is not evidence of efficacy."),
    "worst_off_target": ("Worst off-target", "The single most hazardous unintended hit: upper bound on activity × upper bound on expression × a biological hazard weight. Read high on purpose — a low point estimate is not evidence of safety."),
    "accessibility": ("Accessibility", "Mean probability that the target site is unpaired, from a ViennaRNA local fold. Higher is better: a site buried in a hairpin is hard to invade. This is a prediction, not a measurement."),
    "duplex_dg_per_nt": ("ΔG/nt", "Guide:target hybridisation free energy per nucleotide, kcal/mol. There is an optimum, not a direction: too weak and the guide will not stay bound, too stable and it stops discriminating mismatches."),
    "predicted_signal": ("Signal", "Relative predicted fluorescence from the collateral-cleavage model. Comparable only within this run — the kinetic constants shipped by default are placeholders, so no absolute figure is offered."),
    "pfs_3prime": ("PFS", "The base immediately 3' of the target site. LwaCas13a prefers a non-G here; a G is a deprioritisation, not an exclusion."),
    "robustness": ("Robustness", "Worst coverage across every evaluated context — variants, isoforms, haplotypes. A guide that works on the reference and loses one variant shows it here."),
    "spans_junction": ("EEJ", "Whether the guide spans an exon-exon junction, which can be used to target a spliced isoform specifically."),
}

#: Ordered columns of the results table.
TABLE_COLUMNS: tuple[str, ...] = (
    "rank", "spacer", "score", "verdict", "review_tier",
    "required_coverage", "worst_off_target", "accessibility",
    "duplex_dg_per_nt", "predicted_signal", "pfs_3prime", "robustness",
)


@dataclass(frozen=True, slots=True)
class AccessibilityTrack:
    """Per-nucleotide unpaired probability for one target (FR8.4)."""

    name: str
    profile: Sequence[float]
    #: ``(start, length, label)`` of guides to mark along the track.
    guides: Sequence[tuple[int, int, str]] = ()


#: Shared with ``services/webapp.py`` so the form and the report look like one
#: product. Public because it crosses a module boundary.
REPORT_CSS = """
:root{--bg:#fbfbfd;--fg:#16181d;--muted:#5b6270;--line:#e3e6ec;--card:#fff;
--pass:#0f7b4f;--review:#a86a00;--fail:#b4232c;--accent:#2f5fd0;--track:#dfe5f3}
@media (prefers-color-scheme:dark){:root{--bg:#14161b;--fg:#e9ecf2;--muted:#9aa3b2;
--line:#2a2e37;--card:#1b1e25;--pass:#4bbd8b;--review:#e0a44a;--fail:#f0868d;
--accent:#7ea2f5;--track:#2b3243}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:28px 18px 72px}
h1{font-size:1.5rem;margin:0 0 4px} h2{font-size:1.1rem;margin:34px 0 10px}
.sub{color:var(--muted);font-size:.9rem;margin-bottom:22px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.card .n{font-size:1.6rem;font-weight:650} .card .l{color:var(--muted);font-size:.8rem}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--review);
border-radius:8px;padding:12px 14px;margin:16px 0;font-size:.9rem;color:var(--muted)}
.note.crit{border-left-color:var(--fail)}
.controls{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}
input,select{background:var(--card);color:var(--fg);border:1px solid var(--line);
border-radius:8px;padding:8px 10px;font:inherit;font-size:.9rem}
input{flex:1;min-width:180px}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:.86rem;font-variant-numeric:tabular-nums}
th,td{padding:9px 11px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{position:sticky;top:0;background:var(--card);cursor:pointer;user-select:none;font-weight:600}
th:hover{color:var(--accent)} th .arrow{opacity:.4;font-size:.75em;margin-left:3px}
tbody tr:hover{background:color-mix(in srgb,var(--accent) 7%,transparent)}
td.seq{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.82rem}
.tag{font-weight:600;font-size:.78rem;padding:2px 7px;border-radius:999px;
border:1px solid currentColor}
.PASS{color:var(--pass)} .REVIEW{color:var(--review)} .FAIL{color:var(--fail)}
.track{width:100%;height:92px;display:block}
dl.gloss{display:grid;grid-template-columns:minmax(120px,180px) 1fr;gap:8px 18px;
margin:0;font-size:.9rem}
dl.gloss dt{font-weight:600} dl.gloss dd{margin:0;color:var(--muted)}
pre{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;
overflow-x:auto;font-size:.8rem;color:var(--muted)}
ul.plan{padding-left:18px;color:var(--muted);font-size:.9rem}
.empty{padding:26px;text-align:center;color:var(--muted)}
@media(max-width:640px){.wrap{padding:18px 12px 56px}dl.gloss{grid-template-columns:1fr;gap:2px 0}
dl.gloss dd{margin-bottom:10px}}
"""

_JS = """
(function(){
  var table=document.getElementById('results'); if(!table) return;
  var tbody=table.tBodies[0], rows=Array.prototype.slice.call(tbody.rows);
  var search=document.getElementById('q'), verdict=document.getElementById('v');
  function apply(){
    var needle=(search.value||'').toLowerCase(), want=verdict.value;
    var shown=0;
    rows.forEach(function(r){
      var okText=!needle||r.textContent.toLowerCase().indexOf(needle)>-1;
      var okVerdict=!want||r.getAttribute('data-verdict')===want;
      var on=okText&&okVerdict; r.style.display=on?'':'none'; if(on)shown++;
    });
    document.getElementById('shown').textContent=shown+' of '+rows.length;
  }
  search.addEventListener('input',apply); verdict.addEventListener('change',apply);
  var dir={};
  Array.prototype.forEach.call(table.tHead.rows[0].cells,function(th,i){
    th.addEventListener('click',function(){
      var asc=dir[i]=!dir[i];
      var sorted=rows.slice().sort(function(a,b){
        var x=a.cells[i].getAttribute('data-sort'), y=b.cells[i].getAttribute('data-sort');
        var nx=parseFloat(x), ny=parseFloat(y);
        var cmp=(!isNaN(nx)&&!isNaN(ny))?nx-ny:String(x).localeCompare(String(y));
        return asc?cmp:-cmp;
      });
      sorted.forEach(function(r){tbody.appendChild(r);});
      Array.prototype.forEach.call(table.tHead.rows[0].cells,function(o){
        var s=o.querySelector('.arrow'); if(s)s.textContent='';});
      var span=th.querySelector('.arrow'); if(span)span.textContent=asc?'\\u25b2':'\\u25bc';
    });
  });
  apply();
})();
"""


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _sort_key(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _accessibility_svg(track: AccessibilityTrack, width: int = 1100, height: int = 92) -> str:
    """A per-nucleotide unpaired-probability track with guide positions marked.

    Drawn as an SVG path rather than a chart library: it has to survive being
    opened from a filesystem with no network, and a polyline over a few thousand
    points is not worth a dependency.
    """
    profile = list(track.profile)
    if not profile:
        return '<p class="empty">no accessibility profile available</p>'
    n = len(profile)
    pad = 6
    plot_h = height - 26
    step = (width - 2 * pad) / max(1, n - 1)
    points = " ".join(
        f"{pad + i * step:.2f},{pad + (1.0 - min(max(value, 0.0), 1.0)) * plot_h:.2f}"
        for i, value in enumerate(profile)
    )
    marks = []
    for start, length, label in track.guides:
        x1 = pad + start * step
        x2 = pad + min(n - 1, start + length) * step
        marks.append(
            f'<rect x="{x1:.2f}" y="{pad}" width="{max(1.0, x2 - x1):.2f}" '
            f'height="{plot_h}" fill="var(--accent)" opacity="0.15">'
            f"<title>{html.escape(label)} ({start}-{start + length})</title></rect>"
        )
    return (
        f'<svg class="track" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none" role="img" '
        f'aria-label="Per-nucleotide unpaired probability along the target">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="var(--track)" '
        f'opacity="0.35" rx="6"/>'
        + "".join(marks)
        + f'<polyline fill="none" stroke="var(--accent)" stroke-width="1.4" '
        f'points="{points}"/>'
        f'<text x="{pad}" y="{height - 8}" font-size="11" fill="var(--muted)">'
        f"1</text>"
        f'<text x="{width - pad}" y="{height - 8}" font-size="11" '
        f'text-anchor="end" fill="var(--muted)">{n} nt</text>'
        f"</svg>"
    )


def render_report(
    report: dict[str, Any],
    title: str = "Cas13 guide design",
    subtitle: str = "",
    tracks: Sequence[AccessibilityTrack] = (),
    notes: Sequence[str] = (),
) -> str:
    """Render a :meth:`DesignReport.to_dict` as one self-contained HTML page."""
    counts = report.get("verdict_counts", {})
    ranked = report.get("ranked", [])

    cards = "".join(
        f'<div class="card"><div class="n {cls}">{value}</div>'
        f'<div class="l">{label}</div></div>'
        for label, value, cls in (
            ("PASS", counts.get("PASS", 0), "PASS"),
            ("REVIEW", counts.get("REVIEW", 0), "REVIEW"),
            ("FAIL", counts.get("FAIL", 0), "FAIL"),
            ("Review rate", f"{report.get('review_rate', 0) * 100:.0f}%", ""),
            ("Ranked", len(ranked), ""),
        )
    )

    head = "".join(
        f'<th title="{html.escape(SCORE_GLOSSARY.get(c, (c, ""))[1])}">'
        f'{html.escape(SCORE_GLOSSARY.get(c, (c, ""))[0])}<span class="arrow"></span></th>'
        for c in TABLE_COLUMNS
    )

    body_rows = []
    for entry in ranked:
        metrics = entry.get("metrics") or {}
        cells = []
        for column in TABLE_COLUMNS:
            value = entry.get(column, metrics.get(column))
            attrs = ' class="seq"' if column == "spacer" else ""
            if column == "verdict":
                rendered = (
                    f'<span class="tag {html.escape(str(value))}">'
                    f"{html.escape(_fmt(value))}</span>"
                )
            else:
                rendered = html.escape(_fmt(value))
            cells.append(
                f'<td{attrs} data-sort="{html.escape(_sort_key(value))}">{rendered}</td>'
            )
        body_rows.append(
            f'<tr data-verdict="{html.escape(str(entry.get("verdict", "")))}">'
            + "".join(cells)
            + "</tr>"
        )

    table = (
        '<div class="tablewrap"><table id="results"><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table></div>"
        if body_rows
        else '<div class="tablewrap"><div class="empty">No guide cleared the gate. '
        "The shortfalls below say why.</div></div>"
    )

    glossary = "".join(
        f"<dt>{html.escape(SCORE_GLOSSARY[c][0])}</dt>"
        f"<dd>{html.escape(SCORE_GLOSSARY[c][1])}</dd>"
        for c in TABLE_COLUMNS
        if c in SCORE_GLOSSARY
    )

    track_html = "".join(
        f"<h3 style='font-size:.95rem;margin:18px 0 6px'>{html.escape(t.name)}</h3>"
        + _accessibility_svg(t)
        for t in tracks
    )

    shortfalls = report.get("shortfalls") or []
    shortfall_html = (
        "<h2>Shortfalls</h2><ul class='plan'>"
        + "".join(f"<li>{html.escape(str(s))}</li>" for s in shortfalls)
        + "</ul>"
        if shortfalls
        else ""
    )

    plan = report.get("validation_plan") or []
    plan_html = (
        "<h2>Validation plan</h2><ul class='plan'>"
        + "".join(f"<li>{html.escape(str(step))}</li>" for step in plan)
        + "</ul>"
        if plan
        else ""
    )

    controls = report.get("controls") or []
    controls_html = ""
    if controls:
        rows = "".join(
            f"<tr><td class='seq'>{html.escape(str(c.get('guide_id')))}</td>"
            f"<td class='seq'>{html.escape(str(c.get('spacer')) or '—')}</td>"
            f"<td>{_fmt(c.get('gc'))}</td>"
            f"<td>{_fmt(c.get('verified_no_hits'))}</td></tr>"
            for c in controls
        )
        controls_html = (
            "<h2>Non-targeting controls</h2><div class='tablewrap'><table>"
            "<thead><tr><th>Control</th><th>Spacer</th><th>GC</th>"
            "<th>Verified non-targeting</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )

    # Limitations lead, above the table. A reader who stops after the first
    # screen should have seen what the run did not check, not the ranking.
    note_html = "".join(
        f'<div class="note{" crit" if n.upper().startswith("CRITICAL") else ""}">'
        f"{html.escape(n)}</div>"
        for n in notes
    )
    provenance = json.dumps(report.get("provenance", {}), indent=2, default=str)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{REPORT_CSS}</style></head>
<body><div class="wrap">
<h1>{html.escape(title)}</h1>
<div class="sub">{html.escape(subtitle)}</div>
{note_html}
<div class="cards">{cards}</div>

<h2>Candidates</h2>
<div class="controls">
  <input id="q" type="search" placeholder="Filter by sequence, id, reason…"
         aria-label="Filter candidates">
  <select id="v" aria-label="Filter by verdict">
    <option value="">All verdicts</option>
    <option value="PASS">PASS only</option>
    <option value="REVIEW">REVIEW only</option>
  </select>
  <span class="sub" style="margin:0;align-self:center">showing <b id="shown"></b></span>
</div>
{table}

{"<h2>Target accessibility</h2>" + track_html if track_html else ""}
{controls_html}
{shortfall_html}
{plan_html}

<h2>What the scores mean</h2>
<dl class="gloss">{glossary}</dl>

<h2>Run parameters and versions</h2>
<pre>{html.escape(provenance)}</pre>
</div><script>{_JS}</script></body></html>"""


def write_report(
    report: dict[str, Any],
    path: str | Path,
    **kwargs: Any,
) -> Path:
    """Render and write the HTML report. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(report, **kwargs), encoding="utf-8")
    return path
