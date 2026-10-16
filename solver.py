import re
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

class ParseError(Exception):
    pass

SINGLE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

def safe_root(arg, n):
    n = int(n)
    if not arg.free_symbols:
        s = sp.simplify(arg)
        if s.is_number and s.is_real and s < 0:
            if n % 2 == 0:
                raise ParseError("even root of a negative number is not a real number")
            return sp.real_root(arg, n)
        if n % 2 == 1:
            return sp.real_root(arg, n)
    return sp.root(arg, n)

def safe_sqrt(arg):
    if not arg.free_symbols:
        s = sp.simplify(arg)
        if s.is_number and s.is_real and s < 0:
            raise ParseError("square root of a negative number is not a real number")
    return sp.sqrt(arg)

GLOBALS = {"sqrt": safe_sqrt, "root": safe_root, "Integer": sp.Integer, "Float": sp.Float,
           "Rational": sp.Rational, "Symbol": sp.Symbol}

def check_braces(toks):
    depth = 0
    for t in toks:
        if t == "{": depth += 1
        elif t == "}":
            depth -= 1
            if depth < 0: raise ParseError("unbalanced braces")
    if depth != 0: raise ParseError("unbalanced braces")

def read_bracket(toks, i):
    if i >= len(toks) or toks[i] != "[":
        raise ParseError("expected [")
    depth = 0
    for j in range(i, len(toks)):
        if toks[j] == "[": depth += 1
        elif toks[j] == "]":
            depth -= 1
            if depth == 0:
                return toks[i + 1:j], j + 1
    raise ParseError("unbalanced brackets")

def read_group(toks, i):
    if i >= len(toks) or toks[i] != "{":
        raise ParseError("expected {")
    depth = 0
    for j in range(i, len(toks)):
        if toks[j] == "{": depth += 1
        elif toks[j] == "}":
            depth -= 1
            if depth == 0:
                return toks[i + 1:j], j + 1
    raise ParseError("unbalanced braces")

def translate(toks):
    if not toks:
        raise ParseError("empty group")
    out, n, i = [], len(toks), 0
    prev_end = False
    def start_operand():
        if prev_end: out.append("*")
    while i < n:
        t = toks[i]
        if t.isdigit() or t == ".":
            j = i
            while j < n and (toks[j].isdigit() or toks[j] == "."): j += 1
            num = "".join(toks[i:j])
            if not re.fullmatch(r"\d+(\.\d+)?", num):
                raise ParseError(f"bad number '{num}'")
            start_operand()
            out.append(f"Rational('{num}')" if "." in num else str(int(num)))
            prev_end, i = True, j
        elif t in ("(", "["):
            start_operand(); out.append("("); prev_end = False; i += 1
        elif t in (")", "]"):
            out.append(")"); prev_end = True; i += 1
        elif t in ("+", "-"):
            out.append(t); prev_end = False; i += 1
        elif t in ("\\times", "\\cdot", "\\ast"):
            out.append("*"); prev_end = False; i += 1
        elif t in ("\\div", "/"):
            out.append("/"); prev_end = False; i += 1
        elif t == "\\frac":
            a, k = read_group(toks, i + 1)
            b, k = read_group(toks, k)
            start_operand()
            out.append(f"(({translate(a)})/({translate(b)}))")
            prev_end, i = True, k
        elif t == "\\sqrt":
            if i + 1 < n and toks[i + 1] == "[":
                b_toks, k1 = read_bracket(toks, i + 1)
                idx_expr = to_sympy(b_toks)
                if not idx_expr.is_integer:
                    raise ParseError("root index must be an integer")
                idx_val = int(idx_expr)
                if idx_val == 0:
                    raise ParseError("root index cannot be zero")
                a, k2 = read_group(toks, k1)
                start_operand()
                out.append(f"root({translate(a)}, {idx_val})")
                prev_end, i = True, k2
            else:
                a, k = read_group(toks, i + 1)
                start_operand()
                out.append(f"sqrt({translate(a)})")
                prev_end, i = True, k
        elif t == "^":
            if not out or not prev_end:
                raise ParseError("dangling ^")
            if i + 1 < n and toks[i + 1] == "{":
                a, k = read_group(toks, i + 1)
                out.append(f"**({translate(a)})"); i = k
            elif i + 1 < n and (toks[i + 1].isdigit() or toks[i + 1] in SINGLE):
                out.append(f"**({translate([toks[i + 1]])})"); i += 2
            else:
                raise ParseError("bad exponent")
            prev_end = True
        elif (t in ("x", "X") and prev_end and out and re.search(r"[\d)]$", out[-1])
              and i + 1 < n and (toks[i + 1].isdigit() or toks[i + 1] in ("(", "[", "\\frac", "\\sqrt"))):
            out.append("*"); prev_end = False; i += 1
        elif t in SINGLE:
            start_operand(); out.append(t); prev_end = True; i += 1
        else:
            raise ParseError(f"unsupported token {t}")
    return "".join(out)

def split_relation(toks):
    parts, ops, depth = [[]], [], 0
    for t in toks:
        if t in ("{", "[", "("): depth += 1
        elif t in ("}", "]", ")"): depth -= 1
        if t in ("=", "<", ">", "\\leq", "\\geq", "\\neq") and depth == 0:
            canonical_op = {"\\leq": "<=", "\\geq": ">=", "=": "=", "<": "<", ">": ">", "\\neq": "!="}[t]
            ops.append(canonical_op)
            parts.append([])
        else:
            parts[-1].append(t)
    return parts, ops

def split_eq(toks):
    parts, _ = split_relation(toks)
    return parts

def to_sympy(toks):
    py = translate(toks)
    try:
        e = parse_expr(py, global_dict=dict(GLOBALS),
                       transformations=standard_transformations, evaluate=True)
    except Exception as ex:
        if isinstance(ex, ParseError): raise
        raise ParseError(f"sympy parse failed ({type(ex).__name__})")
    if e.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ParseError("undefined (division by zero)")
    return e

_TOKEN = re.compile(r"\\[A-Za-z]+|\\.|\S")
_UNICODE = {"\u00d7": " \\times ", "\u00f7": " \\div ", "\u2212": "-", "\u00b7": " \\cdot ", "\u2260": " \\neq ",
            "\u2264": " \\leq ", "\u2265": " \\geq "}

def normalize_latex(s):
    """Any LaTeX spelling -> space-separated tokens, e.g. '2x+3=7' -> '2 x + 3 = 7'."""
    s = str(s)
    for k, v in _UNICODE.items():
        s = s.replace(k, v)
    s = s.replace("<=", " \\leq ").replace(">=", " \\geq ")
    s = s.replace("!=", " \\neq ")
    s = re.sub(r"\\(dfrac|tfrac)\b", r"\\frac", s)
    s = re.sub(r"\\(left|right)\b", "", s)
    s = re.sub(r"\\le\b", r"\\leq", s)
    s = re.sub(r"\\ge\b", r"\\geq", s)
    s = re.sub(r"\\ne\b", r"\\neq", s)
    return " ".join(_TOKEN.findall(s))

def has_letters(latex):
    return any(t in SINGLE for t in str(latex).split())

def parse_and(and_expr, var):
    if isinstance(and_expr, sp.Equality):
        return f"{var} = {and_expr.rhs if and_expr.lhs == var else and_expr.lhs}"
    args = and_expr.args if isinstance(and_expr, sp.And) else [and_expr]
    low_val, low_strict = None, None
    high_val, high_strict = None, None
    for a in args:
        if isinstance(a, sp.Equality):
            return f"{var} = {a.rhs if a.lhs == var else a.lhs}"
        if isinstance(a, (sp.StrictLessThan, sp.LessThan, sp.StrictGreaterThan, sp.GreaterThan)):
            lhs, rhs = a.lhs, a.rhs
            if lhs == var:
                val = rhs
                if val in (sp.oo, -sp.oo): continue
                if isinstance(a, sp.StrictLessThan): high_val, high_strict = val, True
                elif isinstance(a, sp.LessThan): high_val, high_strict = val, False
                elif isinstance(a, sp.StrictGreaterThan): low_val, low_strict = val, True
                elif isinstance(a, sp.GreaterThan): low_val, low_strict = val, False
            elif rhs == var:
                val = lhs
                if val in (sp.oo, -sp.oo): continue
                if isinstance(a, sp.StrictLessThan): low_val, low_strict = val, True
                elif isinstance(a, sp.LessThan): low_val, low_strict = val, False
                elif isinstance(a, sp.StrictGreaterThan): high_val, high_strict = val, True
                elif isinstance(a, sp.GreaterThan): high_val, high_strict = val, False
    vname = str(var)
    if low_val is not None and high_val is not None:
        op1 = "<" if low_strict else "<="
        op2 = "<" if high_strict else "<="
        return f"{low_val} {op1} {vname} {op2} {high_val}"
    elif low_val is not None:
        op = ">" if low_strict else ">="
        return f"{vname} {op} {low_val}"
    elif high_val is not None:
        op = "<" if high_strict else "<="
        return f"{vname} {op} {high_val}"
    return str(and_expr)

def format_inequality_solution(expr, var):
    if expr is sp.S.true:
        return f"true for all real {var}"
    if expr is sp.S.false:
        return "no solution"
    if isinstance(expr, sp.Equality):
        return f"{var} = {expr.rhs if expr.lhs == var else expr.lhs}"
    if isinstance(expr, sp.Or):
        parts = [parse_and(arg, var) for arg in expr.args]
        return " or ".join(sorted(parts, key=lambda p: ("<" not in p, p)))
    return parse_and(expr, var)

def format_sigfigs(val, sigfigs=6):
    try:
        f = val.evalf(sigfigs)
        return f"{float(f):.{sigfigs}g}"
    except Exception:
        return str(val.evalf(sigfigs))

def is_ugly(sol):
    try:
        ltx = sp.latex(sol)
        if len(ltx) > 60:
            return True
        if sol.has(sp.I):
            return True
        if re.search(r"\\sqrt.*\\sqrt", ltx):
            return True
    except Exception:
        pass
    return False

def is_ugly_complex(sol):
    try:
        ltx = sp.latex(sol)
        if len(ltx) > 60:
            return True
        if re.search(r"\\sqrt.*\\sqrt", ltx):
            return True
    except Exception:
        pass
    return False

def format_complex_note(c_sols, var='x'):
    evals = [s.evalf(6) for s in c_sols]
    if len(evals) == 2:
        re1, im1 = sp.re(evals[0]), sp.im(evals[0])
        re2, im2 = sp.re(evals[1]), sp.im(evals[1])
        if abs(re1 - re2) < 1e-4 and abs(abs(im1) - abs(im2)) < 1e-4:
            re_val = format_sigfigs(re1, 6)
            im_val = format_sigfigs(abs(im1), 6)
            if abs(float(re1)) < 1e-4:
                return f"{var} \\approx \\pm {im_val}i"
            else:
                return f"{var} \\approx {re_val} \\pm {im_val}i"
    parts = []
    for s in evals:
        re1, im1 = sp.re(s), sp.im(s)
        re_val = format_sigfigs(re1, 6)
        im_val = format_sigfigs(abs(im1), 6)
        sign = "+" if im1 >= 0 else "-"
        if abs(float(re1)) < 1e-4:
            parts.append(f"{'-' if im1 < 0 else ''}{im_val}i")
        else:
            parts.append(f"{re_val} {sign} {im_val}i")
    return f"{var} \\approx " + ", ".join(parts)

def format_inequality_latex(formatted_str):
    if formatted_str == "no solution":
        return r"\text{no solution}"
    if formatted_str.startswith("true for all real "):
        var = formatted_str.split()[-1]
        return rf"\text{{true for all real }} {var}"
    s = formatted_str.replace("<=", "\\leq ").replace(">=", "\\geq ")
    s = re.sub(r"\bor\b", lambda _: r"\text{ or }", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def solve_latex(s):
    res = {"input": s, "ok": False, "kind": None, "result": None, "approx": None, "error": None, "note": None, "latex": None}
    try:
        toks = normalize_latex(s).split()
        if not toks: raise ParseError("empty")
        check_braces(toks)
        parts, ops = split_relation(toks)
        if "!=" in ops:
            raise ParseError("unsupported token \\neq")
        if any(op == "=" for op in ops) and any(op in ("<", ">", "<=", ">=") for op in ops):
            raise ParseError("expression contains both '=' and inequality sign")
        if len(parts) > 1 and parts[-1] == []:
            if any(any(t in SINGLE for t in p) for p in parts[:-1]):
                if ops and ops[-1] == "=":
                    raise ParseError("the equation is incomplete: nothing after the equals sign")
                raise ParseError("the expression is incomplete")
            parts = parts[:-1]
            ops = ops[:-1]
        if any(len(p) == 0 for p in parts): raise ParseError("empty side of '='")
        exprs = [to_sympy(p) for p in parts]
        if len(exprs) == 1:
            e = exprs[0]
            if e.free_symbols:
                simplified = sp.simplify(e)
                res.update(kind="simplify", result=str(simplified), latex=sp.latex(simplified))
            else:
                v = sp.simplify(e)
                res.update(kind="value", result=str(v))
                if not v.is_Integer: res["approx"] = round(float(v), 6)
                latex_val = sp.latex(v)
                if not v.is_Integer and res.get("approx") is not None:
                    res["latex"] = f"{latex_val} = {res['approx']}"
                else:
                    res["latex"] = latex_val
        elif len(ops) >= 1 and all(op in ("<", ">", "<=", ">=") for op in ops):
            op_map = {"<": sp.StrictLessThan, "<=": sp.LessThan, ">": sp.StrictGreaterThan, ">=": sp.GreaterThan}
            if all(not e.free_symbols for e in exprs):
                all_ok = True
                for k in range(len(exprs) - 1):
                    op = ops[k]
                    diff = sp.simplify(exprs[k] - exprs[k + 1])
                    cmp_ok = {"<": diff < 0, "<=": diff <= 0, ">": diff > 0, ">=": diff >= 0}[op]
                    if not cmp_ok:
                        all_ok = False
                        break
                res.update(kind="check", result=str(bool(all_ok)), latex=r"\text{True}" if all_ok else r"\text{False}")
            else:
                syms = sorted(set().union(*(e.free_symbols for e in exprs)), key=str)
                if len(syms) != 1: raise ParseError("only single-variable inequalities supported")
                var = syms[0]
                rels = [op_map[ops[k]](exprs[k], exprs[k + 1]) for k in range(len(exprs) - 1)]
                sol = sp.reduce_inequalities(rels, var)
                formatted = format_inequality_solution(sol, var)
                lat = format_inequality_latex(formatted)
                if sol is sp.S.true:
                    res.update(kind="identity", result=formatted, latex=lat)
                else:
                    res.update(kind="solve", result=formatted, latex=lat)
        elif all(op == "=" for op in ops):
            if all(not e.free_symbols for e in exprs):
                ok = all(sp.simplify(exprs[k] - exprs[k + 1]) == 0 for k in range(len(exprs) - 1))
                res.update(kind="check", result=str(ok), latex=r"\text{True}" if ok else r"\text{False}")
            elif len(exprs) == 2:
                syms = sorted(exprs[0].free_symbols | exprs[1].free_symbols, key=str)
                if len(syms) != 1: raise ParseError("only single-variable equations supported")
                var = syms[0]
                diff = sp.simplify(exprs[0] - exprs[1])
                if diff == 0:
                    res.update(kind="identity", result=f"true for all real {var}", latex=rf"\text{{true for all real }} {var}")
                else:
                    sols = sp.solve(sp.Eq(exprs[0], exprs[1]), var)
                    real_sols = [s for s in sols if s.is_real or (s.is_real is None and sp.simplify(sp.im(s)) == 0)]
                    complex_sols = [s for s in sols if s not in real_sols]
                    if not sols:
                        res.update(kind="solve", result=f"{var} = no solution", latex=r"\text{no solution}")
                    elif not real_sols and complex_sols:
                        if any(is_ugly_complex(s) for s in complex_sols):
                            c_note = f"Complex roots: ${format_complex_note(complex_sols, var)}$"
                        else:
                            c_note = f"Complex roots: {', '.join(map(str, complex_sols))}"
                        res.update(kind="solve", result=f"{var} = no real solutions",
                                   note=c_note, latex=r"\text{no real solutions}")
                    else:
                        if any(is_ugly(s) for s in real_sols):
                            latex_sol = f"{var} \\approx " + ",\\; ".join(format_sigfigs(s, 6) for s in real_sols)
                            exact_sol = f"{var} = " + ",\\; ".join(sp.latex(s) for s in real_sols)
                            res["exact_latex"] = exact_sol
                        else:
                            latex_sol = f"{var} = " + ",\\; ".join(sp.latex(s) for s in real_sols)

                        update_kwargs = {
                            "kind": "solve",
                            "result": f"{var} = {', '.join(map(str, real_sols))}",
                            "latex": latex_sol
                        }
                        if complex_sols:
                            if any(is_ugly_complex(s) for s in complex_sols):
                                c_note = f"Complex roots: ${format_complex_note(complex_sols, var)}$"
                            else:
                                c_note = f"Complex roots: {', '.join(map(str, complex_sols))}"
                            update_kwargs["note"] = c_note
                        res.update(**update_kwargs)
            else:
                raise ParseError("unsupported equation form")
        else:
            raise ParseError("unsupported compound relation")
        res["ok"] = True
    except ParseError as e:
        res["error"] = str(e)
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    return res