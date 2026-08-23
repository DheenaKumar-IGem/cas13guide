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

## LinearFold — **not redistributed here**

`services/arc_model.py` needs LinearFold's **CONTRAfold** energies and this
repository ships no build of it. That is partly practical and partly a licence
matter: LinearFold's own terms restrict redistribution for a fee, which is
stricter than this project's licence and would travel with the artifact. Supply
your own binding through the `MFESource` protocol.
