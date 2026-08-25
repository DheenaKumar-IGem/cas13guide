# Third-party notices

This repository does not only depend on third-party work — it **redistributes**
trained artifacts derived from it. Those are listed here.

Runtime Python dependencies (pydantic, numpy, and the optional ViennaRNA and
onnxruntime) are not listed individually: they are resolved from PyPI at install
time and their licences travel with their packages.

---

## Arc RNAtargeting CNN weights — `src/cas13guide/models/arc/fold_{0,1,2}.onnx`

**What is redistributed.** Three ONNX files holding the published
`guide_nolin_threef` fold weights from the Arc Institute's RNAtargeting web
tool. They arrive here by way of the sibling `crispr-guide-designer` project,
which performed the TensorFlow → ONNX conversion and verified it against
TensorFlow to a maximum difference of 1e-6. The conversion changes the
serialization only; the parameters are the authors' trained values.

**Source.** <https://github.com/ArcInstitute/RNAtargeting_web_custom>

**Paper.** Wei J. et al., "Deep learning and CRISPR-Cas13d ortholog discovery
for optimized RNA targeting", *Cell Systems* (2023).

**Licence.** MIT, reproduced in full below including the upstream copyright
line — the one clause of MIT that carries an obligation.

> MIT License
>
> Copyright (c) 2021 Jingyi Wei
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

---

## Arc parity fixture — `tests/fixtures/arc_reference_pipeline.json`

133 guides with their reference features, logits and scores, produced by
`tools/convert-model/reference_pipeline.py` in the `crispr-guide-designer`
project (MIT, Copyright (c) 2026 Dheena Kumar) driving the vendored reference
logic. Used here as the known-answer vectors for
`tests/test_arc_model.py`; it is test data, not shipped in the wheel.

---

## CONTRAfold Viterbi kernel — `src/cas13guide/services/_contrafold_kernel.py`

**What is redistributed.** One function, copied from `gpu_contrafold.cpu.viterbi`
with a single changed return statement: upstream returns only the structure and
discards the optimum it computed, and that discarded scalar is the CONTRAfold
score this package needs. It is generated from the installed source by a script
rather than transcribed, so the copy cannot drift by a typo, and it is held to
399 known answers by `tests/test_contrafold.py`.

**Source.** `gpu-contrafold` on PyPI. **Licence.** MIT.

**Why copied rather than patched at import.** Every Arc prediction rests on that
number. A textual patch applied at runtime to a 0.1.0 single-author package
would break silently on upgrade; a vendored copy breaks loudly, against a parity
test.

The CONTRAfold model parameters themselves are **not** vendored — they are read
from the installed `gpu-contrafold` package, which is an optional dependency
(`pip install "cas13guide[contrafold]"`).

---

## LinearFold — **not redistributed here**

`services/arc_model.py` needs LinearFold's **CONTRAfold** energies and this
repository ships no build of it. That is partly practical — it has no Windows
wheel on PyPI, no `win-64` build on bioconda, and no binary releases on GitHub —
and partly a licence matter: LinearFold's own terms restrict redistribution for
a fee, which is stricter than this project's licence and would travel with the
artifact.

`services/contrafold.py` reaches the same numbers by a different route: the
CONTRAfold CRF reimplemented in pure Python, plus LinearFold's `sharpturn=false`
default imposed through the parameter table. Verified equal to LinearFold's
output on all 399 reference folds. You can still supply your own LinearFold
binding through the `MFESource` protocol if you prefer.
