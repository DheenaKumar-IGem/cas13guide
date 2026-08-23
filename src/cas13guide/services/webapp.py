"""A browser front end (FR1.2 upload, NFR2 no command-line knowledge).

Standard library only -- ``http.server`` and ``email`` for multipart. That is a
deliberate constraint rather than an aesthetic one: picking Flask or FastAPI
would push a dependency, and a deployment shape, onto whoever runs this, and
neither choice is this package's to make. Everything here is a thin adapter over
:func:`cas13guide.services.workflow.run_request`, so a real deployment can throw
this module away and keep the pipeline.

What this is honest about
-------------------------
``http.server`` is a development server. It handles one request at a time, has
no authentication and no upload size limit beyond the one imposed below, and a
design run happens *inside* the request rather than in a job queue. For a
single user on ``localhost`` -- which is what NFR2 describes -- that is the right
amount of machinery. It binds to ``127.0.0.1`` by default for the same reason,
and says so when asked to bind anywhere else.
"""

from __future__ import annotations

import html
import threading
import webbrowser
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs

from ..contract.enums import AssayType, Effector, Task
from ..contract.presets import get_preset, preset_ids
from .report_html import AccessibilityTrack, _CSS, render_report
from .workflow import DesignRequest, DesignRequestError, run_request

#: Upload ceiling. A transcript is kilobytes; anything past this is a mistake or
#: an attack, and reading it into memory first is how a dev server falls over.
MAX_UPLOAD_BYTES = 8 * 1024 * 1024


def _form_page(message: str = "", is_error: bool = False) -> str:
    presets = "".join(
        f'<option value="{html.escape(p)}">{html.escape(p)} '
        f"({get_preset(p).length} nt)</option>"
        for p in preset_ids()
    )
    # Defaults come from DesignRequest so the form and the CLI cannot disagree.
    # Without the explicit `selected` the browser picks the first option, which
    # silently made RfxCas13d the default when FR1.5 specifies LwaCas13a.
    defaults = DesignRequest()

    def options(values, chosen):
        return "".join(
            f'<option{" selected" if v == chosen else ""}>{html.escape(v)}</option>'
            for v in values
        )

    effectors = options([e.value for e in Effector], defaults.effector)
    assays = options([a.value for a in AssayType], defaults.assay)
    tasks = options([t.value for t in Task], defaults.task)
    banner = (
        f'<div class="note" style="border-left-color:var(--fail)">{html.escape(message)}</div>'
        if is_error and message
        else (f'<div class="note">{html.escape(message)}</div>' if message else "")
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cas13 guide design</title><style>{_CSS}
label{{display:block;font-size:.85rem;font-weight:600;margin:14px 0 4px}}
textarea{{width:100%;min-height:130px;background:var(--card);color:var(--fg);
border:1px solid var(--line);border-radius:8px;padding:10px;
font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.82rem}}
.row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}
button{{background:var(--accent);color:#fff;border:0;border-radius:8px;
padding:11px 20px;font:inherit;font-weight:600;cursor:pointer;margin-top:18px}}
button:hover{{opacity:.9}}
.hint{{color:var(--muted);font-size:.8rem;margin-top:3px}}
</style></head><body><div class="wrap">
<h1>Cas13 guide design</h1>
<div class="sub">Paste a sequence, upload a FASTA file, or pick a built-in target.</div>
{banner}
<form method="post" action="/design" enctype="multipart/form-data">
  <label for="sequence">Sequence (raw or FASTA)</label>
  <textarea id="sequence" name="sequence" placeholder="AUGCAUGC...  or  &gt;my_target
AUGCAUGC..."></textarea>
  <div class="hint">DNA is transcribed to RNA. Anything outside A/C/G/U/T/N is refused.</div>

  <div class="row">
    <div><label for="fasta">…or upload a FASTA file</label>
      <input id="fasta" name="fasta" type="file" accept=".fa,.fasta,.fna,.txt"></div>
    <div><label for="preset">…or a built-in target</label>
      <select id="preset" name="preset"><option value="">none</option>{presets}</select></div>
  </div>

  <label for="variants">Variants / isoforms (optional FASTA)</label>
  <input id="variants" name="variants" type="file" accept=".fa,.fasta,.fna,.txt">
  <div class="hint">A shared guide must work against all of them; robustness shows which it loses.</div>

  <div class="row">
    <div><label for="effector">Effector</label>
      <select id="effector" name="effector">{effectors}</select></div>
    <div><label for="assay">Assay</label>
      <select id="assay" name="assay">{assays}</select></div>
    <div><label for="task">Task</label>
      <select id="task" name="task">{tasks}</select></div>
    <div><label for="spacer_length">Spacer length</label>
      <input id="spacer_length" name="spacer_length" type="number" min="15" max="40"
             placeholder="ortholog default"></div>
    <div><label for="max_mismatches">Off-target stringency</label>
      <input id="max_mismatches" name="max_mismatches" type="number" min="0" max="8" value="4"></div>
    <div><label for="min_accessibility">Min accessibility</label>
      <input id="min_accessibility" name="min_accessibility" type="number" step="0.05"
             min="0" max="1" value="0"></div>
  </div>
  <button type="submit">Design guides</button>
</form>
<p class="hint" style="margin-top:26px">A typical target returns in a few seconds.
The run happens in this request, so the page will wait.</p>
</div></body></html>"""


def _parse_multipart(body: bytes, content_type: str) -> dict[str, Any]:
    """Pull fields and file contents out of a multipart POST.

    Uses ``email`` rather than ``cgi``: ``cgi`` was removed in Python 3.13, and
    a hand-rolled boundary splitter is exactly the kind of code that quietly
    mishandles a filename with a quote in it.
    """
    headers = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
    message = BytesParser(policy=email_policy).parsebytes(headers + body)
    fields: dict[str, Any] = {}
    if not message.is_multipart():
        return fields
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            if payload:
                fields[f"{name}__file"] = payload.decode("utf-8", errors="replace")
        else:
            fields[name] = payload.decode("utf-8", errors="replace").strip()
    return fields


def _number(fields: dict[str, Any], key: str, cast, default=None):
    raw = (fields.get(key) or "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise DesignRequestError(f"{key.replace('_', ' ')}: {raw!r} is not a number") from exc


def _request_from_fields(fields: dict[str, Any]) -> DesignRequest:
    return DesignRequest(
        sequence_text=fields.get("sequence") or fields.get("fasta__file"),
        preset=fields.get("preset") or None,
        variants_text=fields.get("variants__file"),
        effector=fields.get("effector") or "LwaCas13a",
        assay=fields.get("assay") or "in_vitro_detection",
        task=fields.get("task") or "detection",
        spacer_length=_number(fields, "spacer_length", int),
        max_mismatches=_number(fields, "max_mismatches", int, 4),
        min_accessibility=_number(fields, "min_accessibility", float, 0.0),
    )


def _render_outcome(outcome) -> str:
    tracks = []
    if outcome.profile:
        tracks = [
            AccessibilityTrack(
                name=f"{outcome.records[0].name} "
                     f"({len(outcome.records[0].sequence)} nt)",
                profile=outcome.profile,
                guides=[
                    (row.get("start", 0), outcome.spec.spacer_length,
                     f"#{row['rank']} {row['guide_id']}")
                    for row in outcome.ranked[:5]
                ],
            )
        ]
    page = render_report(
        outcome.report,
        title=f"Cas13 guide design - {outcome.records[0].name}",
        subtitle=(
            f"{outcome.spec.profile.effector.value} / "
            f"{outcome.spec.profile.assay.value} / "
            f"{outcome.spec.spacer_length} nt spacer  |  "
            f"spec {outcome.spec.fingerprint()}"
        ),
        tracks=tracks,
        notes=outcome.notes,
    )
    return page.replace(
        "</div><script>",
        '<p style="margin-top:30px"><a href="/">&larr; design another</a></p>'
        "</div><script>",
        1,
    )


class DesignHandler(BaseHTTPRequestHandler):
    server_version = "cas13guide"

    def _send(self, body: str, status: int = 200) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        # This page renders user-supplied sequence names; everything is escaped
        # on the way in, and this is the belt to that pair of braces.
        self.send_header("Content-Security-Policy",
                         "default-src 'none'; style-src 'unsafe-inline'; "
                         "script-src 'unsafe-inline'; img-src data:")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - http.server's interface
        if self.path.split("?")[0] not in ("/", "/index.html"):
            self._send(_form_page("Not found.", is_error=True), status=404)
            return
        self._send(_form_page())

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?")[0] != "/design":
            self._send(_form_page("Not found.", is_error=True), status=404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length > MAX_UPLOAD_BYTES:
            self._send(
                _form_page(
                    f"Upload is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. "
                    "A transcript is kilobytes -- is that the right file?",
                    is_error=True,
                ),
                status=413,
            )
            return
        body = self.rfile.read(length) if length else b""
        content_type = self.headers.get("Content-Type", "")

        try:
            if content_type.startswith("multipart/form-data"):
                fields = _parse_multipart(body, content_type)
            else:
                fields = {
                    key: values[0]
                    for key, values in parse_qs(body.decode("utf-8", "replace")).items()
                }
            outcome = run_request(_request_from_fields(fields))
        except DesignRequestError as exc:
            self._send(_form_page(str(exc), is_error=True), status=400)
            return
        except Exception as exc:  # pragma: no cover - defensive
            # NFR3: a bad input must not return a stack trace to the browser.
            self._send(
                _form_page(f"The design could not be run: {exc}", is_error=True),
                status=500,
            )
            return
        self._send(_render_outcome(outcome))

    def log_message(self, fmt: str, *args: Any) -> None:
        # Quieter than the default, and without the reverse DNS lookup.
        print(f"  {self.address_string()} {fmt % args}")


def make_server(host: str = "127.0.0.1", port: int = 8000) -> HTTPServer:
    return HTTPServer((host, port), DesignHandler)


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> int:
    """Run the development server until interrupted."""
    server = make_server(host, port)
    url = f"http://{host}:{port}/"
    print(f"cas13guide is at {url}")
    if host not in ("127.0.0.1", "localhost"):
        print(
            "  WARNING: this is a single-threaded development server with no "
            "authentication.\n  Binding beyond localhost exposes it to the "
            "network; put a real server in front of it.",
        )
    print("  Ctrl-C to stop.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()
    return 0
