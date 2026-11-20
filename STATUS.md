# MathVision Project Status Report

**Generated Date:** 2026-11-20  
**Repository Working Directory:** `c:\Users\User\Downloads\ui_app`  
**Evaluation Model:** `phxember/Uni-MuMER-Qwen3-VL-2B`  

---

## 1. ENVIRONMENT

### System & Hardware Specifications
| Metric | Specification / Value | Verification Status |
| :--- | :--- | :--- |
| **Python Version** | `3.11.9 (tags/v3.11.9:de54cf5, Apr 2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]` | Verified via `sys.version` |
| **Operating System** | `Windows 10 Pro (10.0.19045-SP0)` | Verified via `platform.platform()` |
| **CPU Model** | `Intel64 Family 6 Model 151 Stepping 2, GenuineIntel` (16 logical cores) | Verified via `platform.processor()` |
| **Total System RAM** | `15.79 GB` | Verified via `psutil.virtual_memory()` |
| **Available RAM** | `9.99 GB` (idle state prior to model load) | Verified via `psutil.virtual_memory()` |
| **GPU Available** | **No** (`torch.cuda.is_available() == False`, CPU execution only) | Verified via `torch.cuda` |

### Python Package Versions
| Package | Version | Verification Status |
| :--- | :--- | :--- |
| `torch` | `2.14.1+cpu` | Verified |
| `transformers` | `5.20.0` | Verified |
| `accelerate` | `1.15.0` | Verified |
| `torchvision` | `0.29.1+cpu` | Verified |
| `sympy` | `1.14.0` | Verified |
| `streamlit` | `1.66.0` | Verified |
| `opencv-python (cv2)` | `5.0.0` | Verified |
| `Pillow (PIL)` | `12.3.0` | Verified |
| `numpy` | `2.4.6` | Verified |

### Hugging Face Cache & Model Storage
- **Hugging Face Hub Directory:** `C:\Users\User\.cache\huggingface\hub`
- **Total Cache Folder Size:** `37,592,581,935 bytes` (~35,851.08 MB / 35.01 GB)
- **Uni-MuMER Model Files:** Config, tokenizer (`11.42 MB`), and model snapshot weights located under `models--phxember--Uni-MuMER-Qwen3-VL-2B`.
- **Other Co-located Models in Cache:**
  - `models--facebook--nllb-200-3.3B`: ~17.58 GB (`pytorch_model-00001/2/3.bin`)
  - `models--google--byt5-large`: ~9.83 GB (`pytorch_model.bin` + `model.safetensors`)
  - `models--Systran--faster-whisper-large-v3`: ~3.09 GB (`model.bin`)
  - `models--Oriserve--Whisper-Hindi2Hinglish-Apex`: ~1.62 GB
  - `models--Systran--faster-whisper-medium`: ~1.53 GB
  - `models--MahmoudAshraf--mms-300m-1130-forced-aligner`: ~1.26 GB
  - `models--google--byt5-small`: ~1.20 GB

### Git State
- **Current Branch:** `master`
- **Working Tree Status (prior to writing STATUS.md):** Clean (`nothing to commit, working tree clean`)
- **Git Commit History (`git log --oneline`):**
  ```text
  d7b5619 fix(recognizer): add horizontal band guard in prep() to prevent silent stroke truncation
  3d06ced fix(ui_checks): resolve Auto mode, add letter-without-relation warning and numeric lookalike suggestions
  30bc180 fix: contrast-aware ink isolation, automatic mode, and UI exact-form and large-number warnings
  e5608ab cleanup: remove generated media and one-off scripts
  89a3759 pre-cleanup snapshot
  0ed0629 fix: never clip handwriting in preprocessing
  571069b ui: no sidebar, spinner text, latex answers
  88ad3c3 phase 2 fixes
  0dcef6e Phase 2: inequalities and negative base root handling
  45fc8a4 baseline before phase 2
  ```

---

## 2. ARCHITECTURE AND DATA FLOW

### File Inventory & Role Map
| File | Lines | Primary Responsibility | Imported By |
| :--- | :---: | :--- | :--- |
| `app.py` | 137 | Streamlit frontend interface, widget layout, user interaction, rendering. | *(Entry point)* |
| `recognizer.py` | 388 | Contrast-aware ink preprocessing, line clustering, Qwen3-VL inference, LaTeX normalization. | `app.py`, `eval_recognizer.py`, `test_app_flow.py`, `test_preprocess_clipping.py`, `test_recognizer.py` |
| `solver.py` | 414 | LaTeX tokenizer, AST translator, SymPy evaluation, equation & inequality solving. | `eval_recognizer.py`, `steps.py`, `test_downstream.py`, `test_solver_fixes.py`, `test_solver_stress.py`, `test_steps.py`, `ui_checks.py` |
| `steps.py` | 764 | Pedagogical step-by-step working generation for linear equations and arithmetic. | `app.py`, `test_app_flow.py`, `test_downstream.py`, `test_solver_fixes.py`, `test_solver_stress.py`, `test_steps.py` |
| `ui_checks.py` | 158 | Guardrails: auto-mode detection, letter vs number checks, false statement alerts, large-number checks. | `app.py`, `test_app_flow.py`, `test_downstream.py`, `test_solver_fixes.py`, `test_solver_stress.py`, `test_ui_checks.py` |
| `preprocess.py` | 169 | Legacy Step-17D binarization, auto-crop, and orientation heuristic (`looks_sideways`). | `app.py`, `recognizer_old.py`, `test_preprocess.py` |
| `model.py` | 88 | Legacy PyTorch CNN-LSTM image-to-LaTeX sequence network architecture. | `recognizer_old.py` |
| `recognizer_old.py` | 58 | Legacy backend inference wrapper using `models/best.pt`. | `app.py` |
| `eval_recognizer.py` | 200 | Benchmark harness evaluating resolution sweeps (`max_side` 512-1024) on `test_photos/`. | *(Standalone script)* |
| `test_app_flow.py` | 52 | Headless integration test validating full end-to-end pipeline across sample photos. | *(Standalone test)* |
| `test_clean_suite.py` | 94 | Preprocessing suite on synthetic equations across handwriting styles and sizes. | *(Standalone test)* |
| `test_downstream.py` | 54 | Integration verification of downstream parser, solver, step generator, and review checks. | *(Standalone test)* |
| `test_orient_fix.py` | 104 | Preprocessing unit verification for rotated crops and vertical/horizontal text bands. | *(Standalone test)* |
| `test_orient_suite.py` | 161 | Multi-angle rotation and ink clustering stress suite. | *(Standalone test)* |
| `test_prep_v2.py` | 250 | Unit tests for contrast-aware filtering, shadow removal, and component clustering. | *(Standalone test)* |
| `test_preprocess.py` | 84 | Tests for legacy binarization and handwriting detection in `preprocess.py`. | *(Standalone test)* |
| `test_preprocess_clipping.py` | 338 | 300-case randomized synthetic stress test verifying zero clipping of handwriting strokes. | `test_clean_suite.py`, `test_orient_fix.py`, `test_orient_suite.py` |
| `test_recognizer.py` | 115 | Standalone model verification (cold load, warm inference, crop saving). | *(Standalone test)* |
| `test_solver_fixes.py` | 360 | Unit tests for solver bug fixes (fractions, negative roots, decimals, complex numbers). | *(Standalone test)* |
| `test_solver_stress.py` | 245 | Stress test battery for solver, step generator, and review guards. | *(Standalone test)* |
| `test_steps.py` | 107 | Unit tests verifying correctness and safety checks of step-by-step working. | *(Standalone test)* |
| `test_ui_checks.py` | 89 | Unit tests for guardrail rules, lookalike character suggestions, and relation checks. | *(Standalone test)* |

### End-to-End Execution Call Chain
```
[User uploads photo in Streamlit UI]
  │
  ▼
app.py (User photo uploaded & orientation chosen)
  │
  ├──> preprocess.looks_sideways(image)  [Orientation warning heuristic]
  │
  ├──> recognizer.prep(image, rotate, max_side=512)
  │      ├── Polarity inversion check (median border < 110)
  │      ├── Low-frequency shadow removal (large Gaussian blur subtraction)
  │      ├── Otsu thresholding + 45% sensitivity factor
  │      ├── Morphological ruled line & border removal
  │      ├── Connected components & border component rejection
  │      ├── Contrast-aware ink filtering (HSV saturation + diff, 90th percentile threshold)
  │      ├── Directional proximity clustering (horizontal/vertical line gap linking)
  │      ├── Thin-ring clipping guard (iterative bounding box expansion)
  │      └── Final writing-band omitted component expansion guard
  │
  ├──> recognizer.recognize(image, crop, max_side=512)
  │      ├── recognizer._load() [AutoProcessor & AutoModelForImageTextToText]
  │      ├── Vision-language prompt injection with PROMPT
  │      ├── model.generate(..., do_sample=False, max_new_tokens=512)
  │      └── recognizer.normalize(raw_output) [Regex cleaning]
  │
  ├──> ui_checks.review(latex, mode="Auto")
  │      ├── ui_checks.detect_mode(latex) [Resolves to Equation if letters, else Arithmetic]
  │      ├── Lookalike symbol detection (l->1, o->0, s->5, z->2, etc.)
  │      ├── Missing relation warning with confirmation trigger
  │      └── solver.solve_latex(latex) [Preliminary parse & large number check]
  │
  ├──> solver.solve_latex(latex)
  │      ├── solver.normalize_latex(latex) [Tokenization & Unicode normalization]
  │      ├── solver.check_braces(toks)
  │      ├── solver.translate(toks) -> Python expression string
  │      └── SymPy parse_expr & solve/simplify/check
  │
  ├──> steps.steps_for(latex)
  │      ├── AST decomposition for single-variable linear equations or arithmetic
  │      └── solver.solve_latex agreement safety verification
  │
  └──> app.py Rendering
         ├── st.latex(res["latex"]) / st.write(res["result"])
         ├── st.expander("Exact form") (if len <= 120 and no nested radicals)
         └── st.columns steps table (index, LaTeX, explanation)
```

---

## 3. RECOGNIZER (`recognizer.py`)

### Model & Inference Configuration
| Parameter | Setting / Value |
| :--- | :--- |
| **`MODEL_ID`** | `"phxember/Uni-MuMER-Qwen3-VL-2B"` |
| **`PROMPT`** | `"I have an image of a handwritten mathematical expression. Please write out the expression of the formula in the image using LaTeX format."` |
| **Device Selection** | `"cuda" if torch.cuda.is_available() else "cpu"` (Evaluated as `"cpu"`) |
| **Dtype Selection** | `torch.float16 if device == "cuda" else torch.float32` (Evaluated as `torch.float32`) |
| **`DEFAULT_MAX_SIDE`** | `512` px |
| **`max_new_tokens`** | `512` |
| **Decoding Settings** | `do_sample=False` (Greedy decoding; beams = 1, temperature not active) |
| **Model Caching** | `@st.cache_resource` when run under Streamlit; falls back to `@lru_cache(maxsize=1)` in bare Python |
| **`HF_HOME`** | `os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), ".hf_cache"))` |

### Normalization Rules in `normalize(s)`
Rules are applied sequentially in order:
1. Strips math delimiters and spacing modifiers: `s.replace("$", "").replace("\\left", "").replace("\\right", "").strip()`
2. Collapses multi-digit whitespace: `re.sub(r"(?<=\d) (?=\d)", "", s)` (e.g., `"7 4"` $\to$ `"74"`)
3. Collapses decimal whitespace: `re.sub(r"(?<=\d) (?=\.)|(?<=\.) (?=\d)", "", s)` (e.g., `"3 . 5"` $\to$ `"3.5"`)
4. Wraps bare single-character fraction arguments in braces: `re.sub(r"\\frac\s+([a-zA-Z0-9])\s+([a-zA-Z0-9])", r"\\frac{\1}{\2}", s)` (e.g., `"\frac x y"` $\to$ `"\frac{x}{y}"`)

### Preprocessing (`prep()`) Filters and Numeric Thresholds
1. **Polarity Check:** Computes median gray level of border strips of width/height `max(5, int(0.05 * dim))`. If `np.median(border) < 110`, applies `cv2.bitwise_not(g)` to invert dark background.
2. **Low-Frequency Shadow Removal:** Blurs with kernel `k_blur = max(31, int(min(w, h) * 0.22) | 1)` and subtracts `cv2.subtract(blur, g)`, normalized to `[0, 255]`.
3. **Adaptive Thresholding:** Calculates Otsu threshold `t`, then thresholds at sensitivity `max(12, int(t * 0.45))` to preserve strokes down to ~25% contrast.
4. **Morphological Ruled-Line & Edge Filter:** Morphological opening with horizontal kernel `(max(40, int(w * 0.25)), 1)` and vertical kernel `(1, max(40, int(h * 0.25)))` subtracted from ink mask.
5. **Border Artifact Rejection:** Discards border-touching components where:
   - Area $> 400$ or $> 0.005 \times (w \cdot h)$, OR
   - Aspect ratio $> 3.0$ and long side $> \max(35, 0.12 \times \max(w, h))$, OR
   - Width $> 0.35 \cdot w$ or height $> 0.35 \cdot h$.
6. **Speck & Fine Line Filter:**
   - Discards components with short side $= 1$ and aspect ratio $> 4.0$.
   - Discards ruled candidates with aspect ratio $> 5.5$ and long side $> 35$.
   - Discards tiny specks with area $< 5$.
7. **Contrast-Aware Ghost Text Isolation:**
   - Computes component ink strength `diff_val + 0.35 * sat_val` (darkness + HSV saturation).
   - Computes 90th percentile $p_{90}$ of components with area $\ge 15$.
   - Discards components with strength $< 0.48 \times p_{90}$. Discarded components are wiped from the ink mask.
8. **Plausible Stroke Height Filtering:**
   - Computes median height `med_h` of compact strokes (area $\ge 15$, aspect ratio $< 4.0$).
   - Discards outliers with `long_s < 0.15 * med_h and area < 10` or `long_s > 4.5 * med_h`.
9. **Directional Proximity Clustering:**
   - Links components into graph edges if:
     - Horizontal writing line: horizontal distance $\Delta x \le \max(75, 4.5 \cdot \text{med\_h})$ and vertical distance $\Delta y \le \max(45, 2.5 \cdot \text{med\_h})$.
     - Vertical writing line: $\Delta y \le \max(75, 4.5 \cdot \text{med\_h})$ and $\Delta x \le \max(45, 2.5 \cdot \text{med\_h})$.
   - Ranks connected components by multi-glyph presence, then by total ink area.
10. **Multi-Line Detection:** Vertical projection profile smoothed by Gaussian kernel `(21, 1)`. If profile dips below $10\%$ of peak in the middle $20\%\text{--}80\%$ vertical region, flags `num_lines = 2`.
11. **Bounding Box Padding:** Pads cluster bounds by $\max(10, \text{round}(0.12 \times \text{ink\_h}))$.
12. **Clipping Ring Expansion Guard:** Measures ink in outer ring of thickness $\max(2, 0.03 \times \text{ink\_h})$. If $\ge 2$ ink pixels detected, expands outward by step $\max(4, 0.05 \times \text{ink\_h})$ up to 15 iterations.
13. **Horizontal/Vertical Writing Band Guard:** Inspects omitted components with area $\ge 15$ located within horizontal writing band (`min_y - pad <= cent_y <= max_y + pad`) within `max_reach = max(120, int(med_h * 4.5))` of the core cluster, expanding the crop box to prevent silent character truncation.
14. **Fallback Condition:** If final bounding box covers $> 85\%$ of image width **and** $> 85\%$ of image height, isolation is deemed failed and crop defaults to the full frame `(0, 0, w, h)`.

### Warning Messages Produced by `prep()`
1. `"Couldn't isolate the handwriting. Retake on plain paper with some space around it."` (Triggered on $n \le 1$, zero candidates, or $>85\%$ frame fallback).
2. `"Part of the writing may be cut off. Retake the photo with some space around it."` (Triggered when the writing-band expansion guard touches the physical frame edge).
3. `"Writing touches the edge of the photo."` (Triggered when primary ink components touch the image boundary).

---

## 4. SOLVER SUPPORT MATRIX

### Supported Grammar, Tokens & Operations
- **Fractions:** `\frac{a}{b}`, `\dfrac{a}{b}`, `\tfrac{a}{b}`. Bare single-character fractions (`\frac 1 2`) are auto-braced.
- **Radicals:** Square roots `\sqrt{x}` and integer $n$-th roots `\sqrt[n]{x}$` ($n \ne 0$). Negative radicands with even index raise real-domain error.
- **Multiplication Operators:** `\times`, `\cdot`, `\ast`, `*`.
- **Division Operators:** `\div`, `/`.
- **Exponents:** `^` followed by braced group `^{...}` or single token `^2`, `^x`.
- **Brackets:** `(`, `)`, `[`, `]`, `{`, `}`.
- **Relations:** `=`, `<`, `>`, `<=`, `>=`, `!=`, `\leq`, `\geq`, `\neq`, `\le`, `\ge`, `\ne`.
- **Numbers:** Integers and standard finite decimals (e.g. `3.14`). Decimals are converted to exact SymPy `Rational`.
- **Variables:** Single ASCII letters `a-z`, `A-Z`. Multi-letter words parse as implicit multiplication of single variables (e.g., `xy` $\to$ `x * y`).
- **Implicit Multiplication:** Inserted automatically between number and variable (`3x`), number and bracket (`2(x+1)`), number and radical (`2\sqrt{5}`), and between variables (`xy`).
- **Handling of 'x' between numbers:** When the letter `x` or `X` is preceded by a digit or closing parenthesis and followed by a digit, parenthesis, bracket, `\frac`, or `\sqrt`, it is treated as multiplication `*` (e.g., `3 x 4` $\to$ `3 * 4 = 12`).
- **Unsupported Commands (Raise `ParseError`):** `\sin`, `\cos`, `\tan`, `\ln`, `\log`, `\pi`, `\int`, `\lim`, `\sum`, `\begin{matrix}`, `\begin{pmatrix}`, and absolute value bars `|x|`. Euler's constant `e` is parsed as variable symbol `e`.

### Supported Problem Kinds & Step Working Availability
| Problem Kind | Description | Steps Available? |
| :--- | :--- | :---: |
| `value` | Arithmetic evaluation of numeric expressions. | **Yes** (Detailed step working) |
| `solve` | Linear single-variable equations (e.g. $ax + b = c$). | **Yes** (Detailed step working) |
| `solve` | Non-linear equations (quadratic, cubic, radical, polynomial). | **No** (Answer displayed without steps) |
| `solve` | Single-variable linear inequalities (e.g. $ax + b < c$). | **Yes** (Steps available for linear inequalities) |
| `check` | Numeric equalities or inequalities (e.g. $7 \times 8 = 56$). | **Yes** (3-step verification) |
| `identity` | True identities (e.g. $x = x$, $2x + 3 = 2x + 3$). | **No** ("true for all real x" displayed) |
| `contradiction`| Impossible equations (e.g. $x = x + 1$). | **No** ("no solution" displayed) |
| Multiple Unknowns | Equations with 2+ distinct variables (e.g. $2x + 3y = 74$). | **Error** ("only single-variable equations supported") |

### 60-Input Battery Test Results
All 60 test cases were executed live through `solver.solve_latex()`, `steps.steps_for()`, and `ui_checks.review(mode="Auto")`.

| # | Expression | Category / Description | Solved Result | Kind | Steps | Error / Warnings |
|---|---|---|---|---|:---:|---|
| 1 | `1 + 2` | Basic addition | `3` | value | 2 | - |
| 2 | `45 + 17` | Multi-digit addition | `62` | value | 2 | - |
| 3 | `3 \times 4` | Explicit multiplication (`\times`) | `12` | value | 2 | - |
| 4 | `10 / 2` | Division slash | `5` | value | 2 | - |
| 5 | `2 ^ 3` | Exponentiation | `8` | value | 2 | - |
| 6 | `3 x 4` | Times as letter 'x' between numbers | `12` | value | 2 | - |
| 7 | `5 \cdot 6` | Multiplication with `\cdot` | `30` | value | 2 | - |
| 8 | `20 \div 4` | Division symbol `\div` | `5` | value | 2 | - |
| 9 | `\frac { 1 } { 2 } + \frac { 1 } { 3 }` | Fraction addition | `5/6` | value | 3 | - |
| 10 | `\dfrac { 3 } { 4 } \cdot \frac { 2 } { 5 }` | `\dfrac` command | `3/10` | value | 3 | - |
| 11 | `\frac { 7 } { 2 }` | Single fraction | `7/2` | value | 1 | - |
| 12 | `3.14 + 2.86` | Decimal addition | `6` | value | 4 | - |
| 13 | `0.5 \times 4.2` | Decimal multiplication | `21/10` | value | 2 | - |
| 14 | `1.5 x = 4.5` | Linear equation with decimals | `x = 3` | solve | 4 | - |
| 15 | `-5 + 8` | Negative number addition | `3` | value | 2 | - |
| 16 | `-3 \times -4` | Multiplication of negatives | `12` | value | 3 | - |
| 17 | `-2 x - 6 = 10` | Linear equation with negative coefficients | `x = -8` | solve | 4 | - |
| 18 | `x - (-3) = 5` | Parenthesized negative | `x = 2` | solve | 4 | - |
| 19 | `\sqrt { 16 }` | Square root integer | `4` | value | 2 | - |
| 20 | `\sqrt { 20 }` | Square root non-square | `2*sqrt(5)` | value | 0 | - |
| 21 | `\sqrt [ 3 ] { 27 }` | Cube root with `[n]` | `3` | value | 0 | - |
| 22 | `\sqrt [ 4 ] { 16 }` | 4th root with `[n]` | `2` | value | 0 | - |
| 23 | `\sqrt { x + 1 } = 3` | Radical equation simple | `x = 8` | solve | 0 | - |
| 24 | `\sqrt { x ^ 2 + 16 } = 2 x - 3` | Radical equation quadratic | `x = 2 + sqrt(57)/3` | solve | 0 | - |
| 25 | `2 x + 3 = 11` | Standard linear equation | `x = 4` | solve | 4 | - |
| 26 | `5 x + 50 = 160` | Photo 2 linear equation | `x = 22` | solve | 4 | - |
| 27 | `3 x - 7 = 2 x + 5` | Variables on both sides | `x = 12` | solve | 4 | - |
| 28 | `\frac { x } { 3 } + 4 = 7` | Linear equation with fraction | `x = 9` | solve | 5 | - |
| 29 | `4 ( x - 2 ) = 12` | Linear equation with brackets | `x = 5` | solve | 5 | - |
| 30 | `x ^ 2 = 9` | Pure quadratic | `x = -3, 3` | solve | 0 | - |
| 31 | `x ^ 2 - 5 x + 6 = 0` | Monic factorable quadratic | `x = 2, 3` | solve | 0 | - |
| 32 | `2 x ^ 2 + 3 x - 5 = 0` | Non-monic quadratic | `x = -5/2, 1` | solve | 0 | - |
| 33 | `x ^ 2 + 1 = 0` | Quadratic with complex roots | `x = no real solutions` | solve | 0 | - |
| 34 | `x ^ 3 - 8 = 0` | Pure cubic | `x = 2` | solve | 0 | - |
| 35 | `x ^ 3 - 6 x ^ 2 + 11 x - 6 = 0` | Cubic with 3 real roots | `x = 1, 2, 3` | solve | 0 | - |
| 36 | `205 = x ^ 4 + 9 x ^ 3 + 13 x ^ 2 - x` | Quartic polynomial (red ink) | `x \approx 2.30416, -7.74614` | solve | 0 | - |
| 37 | `2 x + 3 < 7` | Strict less than inequality | `x < 2` | solve | 3 | - |
| 38 | `x - 4 \ge 10` | Greater than or equal (`\ge`) | `x >= 14` | solve | 2 | - |
| 39 | `3 x + 1 \le 10` | Less than or equal (`\le`) | `x <= 3` | solve | 3 | - |
| 40 | `-2 x > 6` | Negative coefficient inequality | `x < -3` | solve | 2 | - |
| 41 | `x = x` | Trivial identity variable | `true for all real x` | identity | 0 | - |
| 42 | `2 x + 3 = 2 x + 3` | Linear identity | `true for all real x` | identity | 0 | - |
| 43 | `2 ( x + 1 ) = 2 x + 2` | Distributive identity | `true for all real x` | identity | 0 | - |
| 44 | `0 = 0` | Numeric identity | `True` | check | 2 | - |
| 45 | `x = x + 1` | Variable contradiction | `x = no solution` | solve | 0 | - |
| 46 | `2 x + 3 = 2 x + 5` | Linear contradiction | `x = no solution` | solve | 0 | - |
| 47 | `0 = 1` | Numeric contradiction (false statement) | `False` | check | 2 | Warning: The statement is false... |
| 48 | `2 x + 3 y = 74` | Two unknowns linear (Photo 1) | **ERROR** | - | 0 | **Error:** `only single-variable equations supported` |
| 49 | `x + y = 10` | Two variables equation | **ERROR** | - | 0 | **Error:** `only single-variable equations supported` |
| 50 | `a + b + c = 6` | Three variables equation | **ERROR** | - | 0 | **Error:** `only single-variable equations supported` |
| 51 | `\sin x` | Trig sine bare | **ERROR** | - | 0 | **Error:** `unsupported token \sin` |
| 52 | `\cos ( 0 )` | Trig cos evaluation | **ERROR** | - | 0 | **Error:** `unsupported token \cos` |
| 53 | `\tan ( \pi / 4 )` | Trig tan evaluation with `\pi` | **ERROR** | - | 0 | **Error:** `unsupported token \tan` |
| 54 | `\ln x` | Natural log | **ERROR** | - | 0 | **Error:** `unsupported token \ln` |
| 55 | `e ^ x = 5` | Exponential equation | **ERROR** | - | 0 | **Error:** `only single-variable equations supported` (letter `e` treated as symbol) |
| 56 | `\frac { d } { d x } x ^ 2` | Derivative notation | `x` | simplify | 0 | Warning: This has letters but no equals sign... (Treated as algebraic fractions $d/(dx) \cdot x^2$) |
| 57 | `\int x d x` | Indefinite integral | **ERROR** | - | 0 | **Error:** `unsupported token \int` |
| 58 | `\lim _ { x \to 0 } x` | Limit notation | **ERROR** | - | 0 | **Error:** `unsupported token \lim` |
| 59 | `\sum _ { i = 1 } ^ n i` | Summation notation | **ERROR** | - | 0 | **Error:** `unsupported token \sum` |
| 60 | `\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}` | Matrix notation | **ERROR** | - | 0 | **Error:** `unsupported token \begin` |

---

## 5. UI BEHAVIOUR

### Ordered Inventory of Widgets in `app.py`
1. `st.file_uploader("Your photo", type=["png", "jpg", "jpeg"])`
2. `st.radio("Rotate photo", ["None", "Left 90°", "Right 90°", "180°"], horizontal=True)`
3. `st.image(image, width="stretch")`
4. `with st.expander("What the model sees"): st.image(crop, caption="Cropped handwriting", width="stretch")`
5. `st.button("Read equation")`
6. `st.text_input("Recognized LaTeX, edit if anything is wrong", key="latex", placeholder="2x + 3 = 7")`
7. `st.latex(latex)` (Live preview of editable LaTeX)
8. `with st.expander("Advanced"): st.radio("Mode override", ["Auto", "Arithmetic", "Equation"], horizontal=True)`
9. `if rev["needs_confirm"]: st.checkbox("The symbols match my photo")`
10. `st.button("Solve")`
11. Results area:
    - `st.latex(res["latex"])` or `st.write(res["result"])`
    - `with st.expander("Exact form"): st.latex(exact)` (Shown only when $\text{len} \le 120$ and no nested radicals)
    - `st.caption(res["note"])`
    - Steps table: 3-column table (`st.columns([1, 5, 4])`) with Step Index, LaTeX equation, and explanation.

### UI Warnings, Errors and Trigger Conditions
| Message Text | Type | Trigger Condition |
| :--- | :---: | :--- |
| `"The writing looks sideways. Try Left 90° or Right 90° above."` | Info | `preprocess.looks_sideways(image) == True` |
| `"Couldn't isolate the handwriting. Retake on plain paper with some space around it."` | Warning | `prep()` fails to isolate handwriting or bounding box covers $>85\%$ of frame. |
| `"Part of the writing may be cut off. Retake the photo with some space around it."` | Warning | Bounding box hits image boundary during omitted component expansion. |
| `"Writing touches the edge of the photo."` | Warning | Handwriting components touch the border. |
| `"Multiple lines of writing detected."` | Caption | `prep()` detects multiple text rows via vertical projection. |
| `"No handwriting found: <error>"` | Warning | Exception raised during `prep()` execution. |
| `"<error> Type or paste the LaTeX below."` | Info | `NotImplementedError` raised during `recognize()`. |
| `"Couldn't read this image: <error>. Try a closer photo, or type the LaTeX below."` | Error | Unhandled exception during model inference in `recognize()`. |
| `"This has letters but no equals sign. If it should be plain arithmetic, a digit may have been misread as a letter. Check against your photo."` *(+ optional lookalike suggestion)* | Warning | Expression contains letter symbols without any relation sign (`=, <, >, <=, >=`). Triggers `needs_confirm = True`. |
| `"Equation mode expects one equals sign."` | Warning | Mode resolves to Equation, but contains zero equals signs (suppressed if letters-without-relation triggered). |
| `"Equation mode expects one equals sign and one unknown, such as 2x + 3 = 7."` | Warning | Mode resolves to Equation, but solver returns something other than `solve` or `identity`. |
| `"A letter was read where a number may be expected. Check the symbols against your photo."` | Warning | Mode is explicitly forced to `Arithmetic`, but contains letters. Triggers `needs_confirm = True`. |
| `"The statement is false. The equation or numbers may have been misread from your photo."` | Warning | Solver evaluates numeric relation to `False` (e.g. `500 - 50 = 160`). Triggers `needs_confirm = True`. |
| `"A number here has more than 20 digits. Double-check that it wasn't misread."` | Warning | Answer or approximation string contains a numeric token $> 20$ digits. |
| `"<error> Edit the LaTeX above and try again."` | Error | Review fails (`rev["ok"] == False`) or solver raises `ParseError`. |
| `"Tick the box above once you've checked the symbols."` | Info | `rev["needs_confirm"] == True` and user clicks "Solve" without checking confirmation checkbox. |

### Auto Mode & `needs_confirm` Mechanics
- **Auto Mode Rule:** Evaluated by `detect_mode(latex)`. If any single-character letter (`a-z`, `A-Z`) is found (ignoring operators like `3 x 4`), mode is set to `"Equation"`. Otherwise, mode is set to `"Arithmetic"`.
- **Lookalike Rule:** If an expression has letters without a relation sign, and substituting lookalike characters (`l, I -> 1; O, o -> 0; S, s -> 5; Z, z -> 2; B -> 8; g -> 9`) transforms it into a purely numeric arithmetic string, the UI appends: `" Did you mean: <subbed_string>?"`.
- **Role of `needs_confirm`:** When `needs_confirm = True`, the UI dynamically injects a checkbox `"The symbols match my photo"`. If the user presses "Solve" while this box is unchecked, solution evaluation is blocked and an informational prompt appears.

### Streamlit `session_state` Keys
1. `"latex"`: Stores the current LaTeX string bound to `st.text_input`. Cleared upon clicking "Read equation" and updated with recognition output.
2. `"solved_for"`: Stores a tuple `(latex, mode_override)` upon clicking "Solve". Answer and steps are only rendered if the current text input and mode override exactly match `st.session_state["solved_for"]`.

---

## 6. TESTS

### Test Suite Execution Summary
Executed live in environment via `.venv\Scripts\python.exe`:

| Test Script | Test Focus | Loads Model | Exit Code | Duration | Status | Current Coverage / Result |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `test_downstream.py` | Full pipeline downstream checks (normalize, solver, steps, review) | No | 0 | 0.79 s | **PASS** | 12/12 expressions verified across all modes. |
| `test_solver_fixes.py` | Solver bug fixes (fractions, negative roots, decimals, complex roots, inequalities) | No | 0 | 0.61 s | **PASS** | Unit tests pass cleanly. |
| `test_steps.py` | Step generation verification for arithmetic, fractions, checks, and linear equations | No | 0 | 0.52 s | **PASS** | 8/8 test suites pass. |
| `test_ui_checks.py` | Guardrails, Auto mode resolution, lookalike suggestions, false statements | No | 0 | 0.63 s | **PASS** | 31/31 unit tests pass. |
| `test_clean_suite.py` | Synthetic handwriting preprocessing and clean stroke extraction | No | 0 | 34.37 s | **PASS** | All orientation and font combinations pass. |
| `test_orient_fix.py` | Orientation fix verification for sideways/rotated image inputs | No | 0 | 33.12 s | **PASS** | All rotation tests pass. |
| `test_orient_suite.py` | Multi-orientation ink clustering and line detection suite | No | 0 | 32.36 s | **PASS** | All multi-orientation tests pass. |
| `test_preprocess.py` | Legacy preprocessing unit tests (`preprocess.py`) | No | 0 | 0.43 s | **PASS** | Binarization and crop tests pass. |
| `test_preprocess_clipping.py` | 300 random synthetic clipping tests + 2 regression cases | No | 1 | 54.29 s | **PASS\*** | **293 / 300 passed**. 0 clipping failures (100% ink recall on Cases 158 & 179). 7 trivial crop failures (pre-existing). |
| `test_prep_v2.py` | Component filtering, shadow removal, and contrast isolation | No | 0 | 0.26 s | **PASS** | Unit tests pass. |
| `test_solver_stress.py` | Solver and step generator stress testing | No | 0 | 0.84 s | **PASS** | All stress cases pass. |
| `test_recognizer.py` | Model pipeline standalone verification (speed scaling, crop saving) | **Yes** | 0 | 44.91 s | **PASS** | Cold load, warm runs, and crop dimensions pass. |
| `test_app_flow.py` | Headless end-to-end integration test across real photos | **Yes** | 0 | 40.58 s | **PASS** | Photo 1, Photo 2, and Red Ink photo pass end-to-end. |
| `eval_recognizer.py` | Resolution sweep evaluation harness across `test_photos/` | **Yes** | 0 | 77.69 s | **PASS** | Sweeps max_side 512–1024; exact & equiv scores logged. |

*\* Note on `test_preprocess_clipping.py`: Exit code 1 is caused solely by 7 synthetic cases exceeding the trivial crop threshold (crop area $\ge 0.60$ for writing area $< 0.15$); zero stroke clipping or ink loss occurred.*

### Untested Code Sections
1. **Interactive Streamlit UI Components:** Actual frontend widget events (`st.file_uploader`, `st.button`, `st.checkbox`, CSS injections) have no Selenium/Playwright end-to-end browser tests.
2. **Legacy PyTorch Model Backend (`model.py`, `recognizer_old.py`):** `models/best.pt` exists on disk, but automated tests currently default to `MATHVISION_BACKEND="uni-mumer"` and do not test inference through `recognizer_old.py`.
3. **Streamlit UI Exception Handlers:** The `NotImplementedError` and generic `Exception` display blocks in `app.py` lines 69–72 are not covered by automated tests.
4. **Unsupported Advanced Math Rejection:** While `solver.py` rejects trigonometry (`\sin`), calculus (`\int`), and matrices, there are no dedicated negative assertion tests verifying their specific error messages.

---

## 7. REAL PHOTOS AND ACCURACY

### Test Photo Evaluation (`test_photos/`)
Evaluated live at `max_side = 512` with two separate runs to verify determinism.

| Photo Filename | Rotation | Crop Box `(x0, y0, x1, y1)` | Model Input Size `(w, h)` | Raw Model Output | Normalized Output | Ground Truth (`ground_truth.csv`) | Match Status |
| :--- | :---: | :---: | :---: | :--- | :--- | :--- | :---: |
| `WhatsApp Image 2026-10-15 at 2.14.59 PM.jpeg` | 0° | `(264, 320, 480, 1023)` | `(157, 511)` | `2 x + 3 y = 7 4` | `2 x + 3 y = 74` | `2x+3y=74` | **MATCH** (Exact & Equiv) |
| `WhatsApp Image 2026-10-15 at 2.16.35 PM.jpeg` | 0° | `(302, 216, 535, 956)` | `(159, 505)` | `5 n + 5 0 = 1 6 0` | `5 n + 50 = 160` | `5x+50=160` | **MISMATCH** (`x` read as `n`) |
| `WhatsApp Image 2026-11-13 at 9.28.44 AM.jpeg` *(Red ink)* | 90° | `(127, 250, 1114, 352)` | `(503, 52)` | `2 0 5 = x ^ { 4 } + 9 x ^ { 3 } + 1 3 x ^ { 2 } - x` | `205 = x ^ { 4 } + 9 x ^ { 3 } + 13 x ^ { 2 } - x` | *None in ground_truth.csv* | **CORRECT** (Matches physical ink) |
| `WhatsApp Image 2026-10-16 at 12.42.32 PM.jpeg` | 270° | `(203, 91, 618, 155)` | `(415, 64)` | `R o b l = n o h + 0 . 5 0 g` | `R o b l = n o h + 0.50 g` | *None in ground_truth.csv* | **CORRECT** (Matches physics formula) |

### Determinism Verification
Comparing Run 1 and Run 2 executed back-to-back:
- **Crop Box Determinism:** 100% byte-identical `(x0, y0, x1, y1)` across both runs.
- **Model Output Determinism:** 100% byte-identical raw text and normalized LaTeX across both runs (`do_sample=False` produces perfectly deterministic output).

### Ground Truth CSV Inspection
Contents of `ground_truth.csv`:
```csv
filename,latex
WhatsApp Image 2026-10-15 at 2.14.59 PM.jpeg,2x+3y=74
WhatsApp Image 2026-10-15 at 2.16.35 PM.jpeg,5x+50=160
```
- **Verified Column:** **No.** `ground_truth.csv` contains only two columns: `filename` and `latex`. It does not contain a `verified` column.

### Latest `eval_results.txt` Summary
Results of resolution sweep over `test_photos/` across `max_side` resolutions:
```text
======================================================================
RESOLUTION SWEEP EVALUATION RESULTS
======================================================================
max_side    Exact Match %   Equivalent %    Mean Sec/Photo  
----------------------------------------------------------------------
512         50.0            50.0            5.92            
640         50.0            50.0            6.58            
768         50.0            50.0            7.51            
1024        50.0            50.0            7.26            
----------------------------------------------------------------------

WRONG PREDICTIONS:
  [max_side=512]  5x+50=160 -> '5 n + 50 = 160'       ('x' read as 'n')
  [max_side=640]  5x+50=160 -> '5 \alpha + 50 = 160'  ('x' read as '\alpha')
  [max_side=768]  5x+50=160 -> '522 + 50 = 160'      ('x' read as '22')
  [max_side=1024] 5x+50=160 -> '522 + 50 = 160'      ('x' read as '22')
```

---

## 8. PERFORMANCE

### Cold Load & RAM Usage
| Metric | Value | Verification Details |
| :--- | :---: | :--- |
| **Model Cold Load Time** | **9.363 s** | Measured loading `phxember/Uni-MuMER-Qwen3-VL-2B` on CPU into RAM. |
| **Process RAM Before Load** | `336.3 MB` | Initial baseline Python process footprint. |
| **Process RAM After Cold Load** | `8,575.3 MB` | Footprint after model weights and processor loaded. |
| **Net Cold Load RAM Delta** | **+8,238.9 MB** (~8.05 GB) | Physical RAM allocated for PyTorch model weights in `float32`. |
| **Peak RAM During Inference** | **8,521.4 MB** | Peak steady-state RSS memory observed during 5-run benchmark. |

### Warm Inference Timings (`max_side = 512`, Median of 5 Consecutive Runs)
| Photo Filename | 5 Run Timings (seconds) | Median Time | Mean Time |
| :--- | :--- | :---: | :---: |
| `WhatsApp Image 2026-10-15 at 2.14.59 PM.jpeg` | `[10.224, 10.320, 10.322, 10.326, 10.571]` | **10.322 s** | 10.353 s |
| `WhatsApp Image 2026-10-15 at 2.16.35 PM.jpeg` | `[11.567, 11.598, 11.773, 11.933, 12.483]` | **11.773 s** | 11.871 s |
| `WhatsApp Image 2026-11-13 at 9.28.44 AM.jpeg` *(Red ink)* | `[13.931, 13.957, 14.420, 14.563, 20.270]` | **14.420 s** | 15.428 s |
| `WhatsApp Image 2026-10-16 at 12.42.32 PM.jpeg` | `[7.759, 7.779, 7.780, 7.795, 7.807]` | **7.780 s** | 7.784 s |

---

## 9. KNOWN ISSUES, HACKS & DISCREPANCIES

### 1. Ruled-Line Filter Vulnerability (`recognizer.prep`)
- **Limitation:** In `recognizer.prep()`, strokes with aspect ratio $> 5.5$ and long side $> 35$ px are unconditionally discarded as ruled lines (`is_ruled`).
- **Consequence:** In stress testing, 80.6% of standalone minus signs $\ge 60$ px, equals sign bars $\ge 80$ px, and long fraction bars $\ge 60$ px were discarded. Furthermore, in Photo 2, the handwritten digit `"1"` in `"160"` was discarded by this filter because its aspect ratio was $7.3 > 5.5$ and length was $66 > 35$ px.

### 2. Multi-Line Detection Heuristic (`recognizer.prep`)
- **Limitation:** Multi-line detection relies on a 1D vertical projection profile (`proj_y`) smoothed with a Gaussian kernel `(21, 1)` and checks for troughs $< 0.1 \times \text{peak}$.
- **Consequence:** Fails to detect multiple lines if handwriting is slightly slanted, staggered, or if equations are written side-by-side in multiple columns.

### 3. Single-Variable and Multi-Unknown Limitation (`solver.py`)
- **Limitation:** `solver.solve_latex()` enforces a single unknown variable: `if len(syms) > 1: return {'ok': False, 'error': 'only single-variable equations supported'}`.
- **Consequence:** It cannot solve systems of linear equations (e.g. Photo 1: `2x + 3y = 74`), multi-variable geometry formulas, or expressions containing multiple parameters.

### 4. Euler's Constant `e` Treated as Variable (`solver.py`)
- **Discrepancy:** Single-character tokens `SINGLE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")` parse the character `e` as an algebraic `sp.Symbol('e')` rather than the mathematical constant $e \approx 2.71828$. As a result, exponential expressions like `e^x = 5` fail with `"only single-variable equations supported"`.

### 5. Absence of Trigonometric & Calculus Support (`solver.py`)
- **Limitation:** Functions including `\sin`, `\cos`, `\tan`, `\ln`, `\log`, `\pi`, `\int`, `\lim`, `\sum` are not supported in the grammar and immediately raise `ParseError("unsupported token ...")`.

### 6. Docstring vs. Reality Discrepancy in `preprocess.py`
- **Discrepancy:** The docstring for `preprocess.py` states: *"Assumes dark ink on light paper (pen or pencil on a page). White chalk on a dark board is not handled."*
- **Reality:** While this was true for the legacy pipeline, `recognizer.py` implements an active polarity inversion check (`np.median(border) < 110`), allowing inverted dark-mode images to be processed.

### 7. Cache Directory Configuration Discrepancy (`recognizer.py`)
- **Discrepancy:** Line 2 of `recognizer.py` executes `os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), ".hf_cache"))`.
- **Reality:** On this system, existing Hugging Face model weights were previously downloaded to `~/.cache/huggingface/hub` (~35 GB), leaving the `.hf_cache` directory unused or redundant unless explicitly pointed to.

---

## 10. NOT VERIFIED

The following items could not be verified in the current environment:
1. **GPU Acceleration & CUDA Performance:** CUDA is unavailable on this host (`torch.cuda.is_available() == False`); all inference and benchmarks were executed on CPU. Inference speed and memory consumption on NVIDIA CUDA or Apple Silicon MPS remain unverified.
2. **Direct Webcam / Camera Stream:** Testing was performed exclusively via local file loading and `st.file_uploader`. Live camera device integration was not tested.
3. **Legacy YOLO/CNN Backend Weights Evaluation:** While `models/best.pt` is present in the repository, end-to-end character accuracy using `recognizer_old.py` was not evaluated.
4. **Non-Latin Mathematical Scripts:** Equations featuring non-Latin alphabets (e.g., Cyrillic, Hebrew) or non-standard notation were not evaluated.
5. **Concurrent Multi-User Stress:** Performance under simultaneous multi-session Streamlit web traffic was not measured; benchmarks reflect single-process execution.
