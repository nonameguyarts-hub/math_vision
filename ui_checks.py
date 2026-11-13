"""Pure-Python checks the UI runs before solving (no Streamlit needed, so they are testable)."""
import re
from solver import normalize_latex, solve_latex, split_relation

MODES = ("Auto", "Arithmetic", "Equation")


def detect_mode(latex: str) -> str:
    """Choose mode from recognized LaTeX: letters present -> Equation; no letters -> Arithmetic."""
    norm = normalize_latex(latex)
    cleaned = re.sub(r"\\[a-zA-Z]+", "", norm)
    cleaned = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", " * ", cleaned)
    has_letters = bool(re.search(r"[a-zA-Z]", cleaned))
    return "Equation" if has_letters else "Arithmetic"


def has_nested_radicals(latex: str) -> bool:
    """Return True if LaTeX contains a radical inside another radical."""
    radical_depth = 0
    brace_stack = []
    i = 0
    n = len(latex)
    while i < n:
        if latex.startswith(r"\sqrt", i):
            if radical_depth > 0:
                return True
            i += 5
            if i < n and latex[i] == "[":
                j = latex.find("]", i)
                if j != -1:
                    i = j + 1
            while i < n and latex[i].isspace():
                i += 1
            if i < n and latex[i] == "{":
                brace_stack.append(True)
                radical_depth += 1
                i += 1
            continue
        elif latex[i] == "{":
            brace_stack.append(False)
        elif latex[i] == "}":
            if brace_stack:
                is_rad = brace_stack.pop()
                if is_rad:
                    radical_depth -= 1
        i += 1
    return False


def friendly(error: str) -> str:
    """Plain-language version of a solver error."""
    e = str(error or "")
    if e.startswith("unsupported token"):
        tok = e.split("token", 1)[1].strip()
        if tok in ("_", "\\\\sum", "\\\\int", "\\\\lim"):
            return f"The symbol {tok} (subscripts, sums, integrals, limits) isn't supported. If your photo is a plain equation, the recognizer probably misread it."
        return f"The symbol {tok} isn't supported yet. Check it against your photo."
    if e == "unbalanced braces":
        return "The brackets { } don't match, so part of the equation was probably misread."
    if e.startswith("only single-variable"):
        return "Only equations with one unknown (such as x) are supported."
    if e == "empty":
        return "Nothing to solve yet. Read a photo or type an equation."
    if e.startswith("undefined"):
        return "This divides by zero."
    if e == "the equation is incomplete: nothing after the equals sign":
        return "The equation is incomplete: nothing after the equals sign."
    return e


def review(latex: str, mode: str = "Auto") -> dict:
    """Return the solver result plus warnings the UI must show before an answer is trusted.

    needs_confirm=True means the user has to tick 'the symbols look right' first.
    """
    r = solve_latex(latex)
    out = {"ok": r["ok"], "error": friendly(r["error"]) if r["error"] else None, "warnings": [], "needs_confirm": False, "result": r}
    toks = normalize_latex(latex).split()
    parts, ops = split_relation(toks)
    has_eq = any(op == "=" for op in ops)
    has_ineq = any(op in ("<", ">", "<=", ">=") for op in ops)
    if has_eq and has_ineq:
        out["warnings"].append(
            "The statement contains both an equals sign and an inequality sign. Check against your photo.")
        out["needs_confirm"] = True
    if not r["ok"]:
        return out
    if mode == "Arithmetic" and r["kind"] in ("simplify", "solve", "identity"):
        out["warnings"].append(
            "A letter was read where a number may be expected. Check the symbols against your photo.")
        out["needs_confirm"] = True
    if mode == "Equation" and r["kind"] not in ("solve", "identity"):
        out["warnings"].append("Equation mode expects one equals sign and one unknown, such as 2x + 3 = 7.")
    if r["kind"] == "check" and r["result"] == "False":
        out["warnings"].append(
            "The statement is false. The equation or numbers may have been misread from your photo.")
        out["needs_confirm"] = True
    displayed = r.get("latex") if r.get("exact_latex") else (r.get("result") or "")
    if displayed:
        has_large = any(len(d) > 20 for d in re.findall(r"\d+", re.sub(r"\\[a-zA-Z]+", "", displayed)))
        if not has_large and not r.get("exact_latex"):
            has_large = len(re.findall(r"\d", displayed)) > 20
        if has_large:
            out["warnings"].append("The result is a very large number. A digit may have been misread.")
            out["needs_confirm"] = True
    return out