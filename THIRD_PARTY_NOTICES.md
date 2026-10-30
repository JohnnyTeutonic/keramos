# Third-party notices

## llama.cpp / ggml

Upstream: https://github.com/ggml-org/llama.cpp
Pinned reference: b6000, commit 4762ad7316dcdec20016ab5985fb46a27902204d.
License source: https://github.com/ggml-org/llama.cpp/blob/4762ad7316dcdec20016ab5985fb46a27902204d/LICENSE

Scope: tools/ir_quant.py implements the stored GGUF tensor layouts and scalar
f32 dequantization numerics documented in that revision's ggml/src/ggml-quants.c
and ggml/src/ggml-common.h. Those source references already appear in its module
documentation. The upstream copyright and permission notice is retained below.
The adapters/llamacpp tools build against an external llama.cpp checkout; this
repository does not vendor that checkout or relicense its code.

Exact upstream license text:

MIT License

Copyright (c) 2023-2024 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Other dependencies and fixtures

Python dependencies and external adapter engines are installed separately and
retain their respective licenses. They are not relicensed by this project's MIT
license. No third-party trained checkpoint is relicensed by this notice; the
bundled tp_ conformance models are documented as graphs with random weights.

The dependency-free cleanroom/rust implementation documents its JSON parser and
FIPS 180-4 SHA-256 implementation as written for this project. This notice does
not attribute those implementations to an external codebase without evidence.
