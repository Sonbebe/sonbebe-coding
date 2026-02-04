import re
import os
import argparse
import logging
import sys
from typing import List, Dict, Any, Optional, Tuple, Union, Set

# ==================================================================================
# 0. LOGGING CONFIGURATION
# ==================================================================================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("VCAST_GEN")

# ==================================================================================
# 1. UTILITIES & HELPER FUNCTIONS
# ==================================================================================
STUB_UNIT_NAME = "uut_prototype_stubs"


def remove_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*?\n", "\n", text)
    return text


def find_matching_paren(
    text: str, start_idx: int, open_char: str = "(", close_char: str = ")"
) -> int:
    depth = 0
    for i in range(start_idx, len(text)):
        if text[i] == open_char:
            depth += 1
        elif text[i] == close_char:
            depth -= 1
            if depth == 0:
                return i
    return -1


def safe_eval(expr: str) -> Optional[Union[int, float]]:
    if not expr:
        return None
    clean_expr = re.sub(r"(\b0x[0-9a-fA-F]+|\b\d+)[uUlLfF]+\b", r"\1", expr)
    clean_expr = clean_expr.replace("(", "").replace(")", "").strip()
    if re.match(r"^[0-9a-fA-FxX\s\+\-\*\/\.]+$", clean_expr):
        try:
            return int(eval(clean_expr))
        except:
            return None
    return None


def mask_switch_blocks(body: str) -> str:
    masked_body = list(body)
    for match in re.finditer(r"\bswitch\s*\((.*?)\)\s*\{", body):
        start = match.start()
        brace_open = match.end() - 1
        brace_close = find_matching_paren(body, brace_open, "{", "}")
        if brace_close != -1:
            for i in range(start, brace_close + 1):
                masked_body[i] = " "
    return "".join(masked_body)


# ==================================================================================
# 2. CODE PARSER
# ==================================================================================


class CodeParser:
    @staticmethod
    def extract_functions(content: str) -> List[Dict[str, str]]:
        header_pattern = re.compile(r"([\w\*\s]+?)\s+(\w+)\s*\((.*?)\)\s*\{", re.DOTALL)
        CONTROL_KEYWORDS = [
            "if",
            "else",
            "switch",
            "case",
            "default",
            "for",
            "while",
            "do",
            "return",
            "break",
            "continue",
            "struct",
        ]
        functions = []
        pos = 0
        while True:
            match = header_pattern.search(content, pos)
            if not match:
                break
            return_type = match.group(1).strip()
            func_name = match.group(2).strip()
            params_raw = match.group(3).strip()
            if func_name in CONTROL_KEYWORDS or not return_type:
                pos = match.end()
                continue
            start_index = match.end() - 1
            body_end_index = find_matching_paren(content, start_index, "{", "}")
            if body_end_index != -1:
                body = content[start_index + 1 : body_end_index]
                functions.append(
                    {
                        "return_type": return_type,
                        "func_name": func_name,
                        "params": params_raw,
                        "body": body,
                    }
                )
                pos = body_end_index + 1
            else:
                pos = match.end()
        return functions

    @staticmethod
    def extract_if_with_body(body_content: str) -> List[Dict[str, Any]]:
        if_structures = []
        for match in re.finditer(r"\bif\s*\(", body_content):
            start_paren = match.end() - 1
            end_paren = find_matching_paren(body_content, start_paren)
            if end_paren == -1:
                continue
            condition = body_content[start_paren + 1 : end_paren]
            body_start = end_paren + 1
            while body_start < len(body_content) and body_content[body_start].isspace():
                body_start += 1
            code_block = ""
            struct_end_index = end_paren
            if body_start < len(body_content):
                if body_content[body_start] == "{":
                    body_end = find_matching_paren(body_content, body_start, "{", "}")
                    if body_end != -1:
                        code_block = body_content[body_start + 1 : body_end]
                        struct_end_index = body_end
                else:
                    semi = body_content.find(";", body_start)
                    if semi != -1:
                        code_block = body_content[body_start : semi + 1]
                        struct_end_index = semi + 1
            if_structures.append(
                {
                    "condition": condition,
                    "body": code_block,
                    "start_index": match.start(),
                    "end_index": struct_end_index,
                }
            )
        return if_structures

    @staticmethod
    def extract_if_conditions(body_content: str) -> List[str]:
        conditions = []
        for match in re.finditer(r"\bif\s*\(", body_content):
            start = match.end() - 1
            end = find_matching_paren(body_content, start)
            if end != -1:
                conditions.append(body_content[start + 1 : end])
        return conditions

    @staticmethod
    def extract_loop_ranges(body: str) -> List[Dict[str, Any]]:
        loops = []
        for match in re.finditer(r"\bfor\s*\(", body):
            start = match.start()
            paren_start = match.end() - 1
            paren_end = find_matching_paren(body, paren_start)
            if paren_end == -1:
                continue
            for_args = body[paren_start + 1 : paren_end]
            parts = for_args.split(";")
            entry_condition = "1"
            loop_info = {"condition": "1"}

            if len(parts) >= 2:
                init_part = parts[0].strip()
                cond_part = parts[1].strip()
                loop_var = None
                init_val = None
                init_func = None

                # Relaxed regex to capture function call with arguments: i = func(...)
                m_func = re.search(
                    r"(?:[\w\s]+\s)?(\w+)\s*=\s*(\w+)\s*\(.*?\)", init_part
                )
                if m_func:
                    loop_var = m_func.group(1)
                    init_func = m_func.group(2)
                else:
                    m_const = re.search(
                        r"(?:[\w\s]+\s)?(\w+)\s*=\s*([-\w\d]+)", init_part
                    )
                    if m_const:
                        loop_var, init_val = m_const.groups()

                if loop_var:
                    entry_condition = re.sub(
                        r"\b" + re.escape(loop_var) + r"\b",
                        init_val if init_val else "0",
                        cond_part,
                    )
                    bound_match = re.search(
                        r"\b" + re.escape(loop_var) + r"\b\s*([<>!]=?)\s*([^;]+)",
                        cond_part,
                    )
                    if bound_match:
                        operator = bound_match.group(1)
                        bound_expr = bound_match.group(2).strip()
                        if "//" in bound_expr:
                            bound_expr = bound_expr.split("//")[0].strip()
                        bound_var = bound_expr
                    else:
                        operator = None
                        bound_var = None

                    loop_info = {
                        "condition": entry_condition,
                        "iterator": loop_var,
                        "bound_var": bound_var,
                        "operator": operator,
                        "init_val": init_val,
                        "init_func": init_func,
                    }
                else:
                    entry_condition = cond_part
                    loop_info = {"condition": entry_condition}

            body_start = paren_end + 1
            while body_start < len(body) and body[body_start].isspace():
                body_start += 1
            if body_start < len(body) and body[body_start] == "{":
                body_end = find_matching_paren(body, body_start, "{", "}")
                if body_end != -1:
                    loops.append(
                        {"start": body_start, "end": body_end, "info": loop_info}
                    )

        for match in re.finditer(r"\bwhile\s*\(", body):
            start = match.start()
            paren_start = match.end() - 1
            paren_end = find_matching_paren(body, paren_start)
            if paren_end == -1:
                continue
            condition = body[paren_start + 1 : paren_end].strip()
            body_start = paren_end + 1
            while body_start < len(body) and body[body_start].isspace():
                body_start += 1
            if body_start < len(body) and body[body_start] == "{":
                body_end = find_matching_paren(body, body_start, "{", "}")
                if body_end != -1:
                    loops.append(
                        {
                            "start": body_start,
                            "end": body_end,
                            "info": {"condition": condition},
                        }
                    )

        for match in re.finditer(r"\bdo\s*\{", body):
            start = match.start()
            brace_open = match.end() - 1
            brace_close = find_matching_paren(body, brace_open, "{", "}")
            if brace_close != -1:
                loops.append(
                    {
                        "start": brace_open,
                        "end": brace_close,
                        "info": {"condition": "1"},
                    }
                )
        return loops

    @staticmethod
    def extract_path_constraints(
        body: str,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        constraints_false = {}
        constraints_true = {}

        def parse_recursive(text, current_guards, parent_conds):
            idx = 0
            persistent_guards = current_guards[:]
            chain_negations = []
            while idx < len(text):
                match = re.search(r"\bif\s*\(", text[idx:])
                if not match:
                    break
                start = idx + match.start()
                p_open = start + len("if")
                while p_open < len(text) and text[p_open].isspace():
                    p_open += 1
                p_close = find_matching_paren(text, p_open)
                if p_close == -1:
                    idx = start + 1
                    continue
                cond = text[p_open + 1 : p_close].strip()
                if cond not in constraints_false:
                    constraints_false[cond] = []
                if cond not in constraints_true:
                    constraints_true[cond] = []
                total_neg = list(set(persistent_guards + chain_negations))
                constraints_false[cond].extend(total_neg)
                constraints_true[cond].extend(parent_conds)
                rest = p_close + 1
                while rest < len(text) and text[rest].isspace():
                    rest += 1
                body_start = rest
                body_end = -1
                has_brace = False
                if rest < len(text) and text[rest] == "{":
                    has_brace = True
                    body_end = find_matching_paren(text, rest, "{", "}")
                else:
                    semi = text.find(";", rest)
                    if semi != -1:
                        body_end = semi + 1
                if body_end == -1:
                    idx = rest
                    continue
                inner_body = (
                    text[body_start + 1 : body_end]
                    if has_brace
                    else text[body_start:body_end]
                )
                parse_recursive(inner_body, persistent_guards, parent_conds + [cond])
                if re.search(r"\b(break|continue)\s*;|\breturn\b", inner_body):
                    if cond not in persistent_guards:
                        persistent_guards.append(cond)
                next_scan = body_end + 1
                if re.match(r"\s*else", text[next_scan:]):
                    else_m = re.match(r"\s*else", text[next_scan:])
                    else_end = next_scan + else_m.end()
                    if re.match(r"\s*if\s*\(", text[else_end:]):
                        chain_negations.append(cond)
                        idx = else_end
                        continue
                    else:
                        chain_negations = []
                        e_open = else_end
                        while e_open < len(text) and text[e_open].isspace():
                            e_open += 1
                        if e_open < len(text) and text[e_open] == "{":
                            e_close = find_matching_paren(text, e_open, "{", "}")
                            if e_close != -1:
                                e_body = text[e_open + 1 : e_close]
                                parse_recursive(
                                    e_body, persistent_guards + [cond], parent_conds
                                )
                                idx = e_close + 1
                                continue
                        else:
                            semi = text.find(";", e_open)
                            if semi != -1:
                                e_body = text[e_open : semi + 1]
                                parse_recursive(
                                    e_body, persistent_guards + [cond], parent_conds
                                )
                                idx = semi + 1
                                continue
                else:
                    chain_negations = []
                idx = next_scan

        parse_recursive(body, [], [])
        return constraints_false, constraints_true

    @staticmethod
    def extract_switch_cases(body: str) -> List[Dict[str, Any]]:
        scenarios = []
        pos = 0
        while pos < len(body):
            match = re.search(r"\bswitch\s*\((.*?)\)\s*\{", body[pos:])
            if not match:
                break
            var = match.group(1).strip()
            start = pos + match.start()
            brace_open = pos + match.end() - 1
            brace_close = find_matching_paren(body, brace_open, "{", "}")
            if brace_close != -1:
                switch_body = body[brace_open + 1 : brace_close]
                iter_cases = list(
                    re.finditer(r"\b(case\s+([a-zA-Z0-9_]+)|default)\s*:", switch_body)
                )
                for i, m in enumerate(iter_cases):
                    c_head_end = m.end()
                    c_code_end = (
                        iter_cases[i + 1].start()
                        if i + 1 < len(iter_cases)
                        else len(switch_body)
                    )
                    c_block = switch_body[c_head_end:c_code_end]
                    is_def = m.group(1).startswith("default")
                    val_str = "default" if is_def else m.group(2)
                    real_val = "99"
                    if not is_def:
                        if val_str.isdigit():
                            real_val = int(val_str)
                        else:
                            real_val = f"MACRO={val_str}"  # [UPDATED] Use MACRO= for switch cases

                    base = {var: real_val, "_conditions": [f"{var} == {val_str}"]}
                    base["_branch_path_str"] = f"[{var} == {val_str}]"
                    nested = CodeParser.extract_if_conditions(c_block)
                    if nested:
                        for n_cond in nested:
                            pos_s = LogicSolver.solve_full_coverage(n_cond, True)
                            neg_s = LogicSolver.solve_full_coverage(n_cond, False)
                            for s in pos_s + neg_s:
                                comb = base.copy()
                                comb.update(
                                    {
                                        k: v
                                        for k, v in s.items()
                                        if k != "_conditions" and k != "_desc"
                                    }
                                )
                                comb["_conditions"] = base["_conditions"] + s.get(
                                    "_conditions", []
                                )
                                scenarios.append(comb)
                    else:
                        scenarios.append(base)
                pos = brace_close + 1
            else:
                pos = start + 1
        return scenarios

    @staticmethod
    def extract_ternary_operators(body: str) -> List[str]:
        conds = []
        regex = r"([^;{}]+?)\s*\?\s*([^;:]+)\s*:\s*([^;,\)]+)"
        for match in re.finditer(regex, body):
            raw = match.group(1).strip()
            clean = re.sub(r"^.*?(?:return|(?<!=)=(?!=)|,)\s*", "", raw).strip()
            if clean:
                conds.append(clean)
        return conds

    @staticmethod
    def extract_assignments(body: str) -> Dict[str, List[str]]:
        assigns = {}
        regex = r"\b(\w+)\s*=\s*(?:\([\w\s\*]+\)\s*)?(\w+)\s*\("
        for match in re.finditer(regex, body):
            var, func = match.groups()
            if func not in ["if", "while", "for", "switch", "sizeof"]:
                if var not in assigns:
                    assigns[var] = []
                if func not in assigns[var]:
                    assigns[var].append(func)
        return assigns

    @staticmethod
    def extract_state_dependencies(body: str) -> Dict[str, Dict[str, str]]:
        deps = {}
        for match in re.finditer(r"\b(?:if|else\s+if)\s*\(", body):
            start = match.end() - 1
            end = find_matching_paren(body, start)
            if end == -1:
                continue
            cond = body[start + 1 : end].strip()
            rest = end + 1
            while rest < len(body) and body[rest].isspace():
                rest += 1
            blk = ""
            if rest < len(body):
                if body[rest] == "{":
                    e = find_matching_paren(body, rest, "{", "}")
                    if e != -1:
                        blk = body[rest + 1 : e]
                else:
                    s = body.find(";", rest)
                    if s != -1:
                        blk = body[rest : s + 1]
            for am in re.finditer(r"([\w\->\.]+)\s*=\s*([\w\d_]+)\s*;", blk):
                var, val_str = am.groups()
                var, val_str = var.strip(), val_str.strip()
                # Store everything as string to match later
                real = str(val_str)
                if val_str.isdigit():
                    real = str(int(val_str))

                if var not in deps:
                    deps[var] = {}
                deps[var][real] = cond
        return deps

    @staticmethod
    def extract_guard_values(body: str) -> Dict[str, Any]:
        guards = {}
        for match in re.finditer(r"\bif\s*\(", body):
            start = match.end() - 1
            end = find_matching_paren(body, start)
            if end == -1:
                continue
            cond = body[start + 1 : end]
            rest = end + 1
            while rest < len(body) and body[rest].isspace():
                rest += 1
            blk = ""
            if rest < len(body):
                if body[rest] == "{":
                    e = find_matching_paren(body, rest, "{", "}")
                    if e != -1:
                        blk = body[rest + 1 : e]
                else:
                    s = body.find(";", rest)
                    if s != -1:
                        blk = body[rest : s + 1]
            if blk and re.search(r"return\s*[;]", blk):
                neg = LogicSolver.solve_full_coverage(cond, False)
                if neg:
                    s = neg[0]
                    guards.update(
                        {
                            k: v
                            for k, v in s.items()
                            if k not in ["_conditions", "_desc"]
                        }
                    )
        return guards

    @staticmethod
    def extract_called_functions(body: str) -> Set[str]:
        called = set()
        for match in re.finditer(r"\b(\w+)\s*\(", body):
            name = match.group(1)
            if name not in [
                "if",
                "while",
                "for",
                "switch",
                "sizeof",
                "return",
                "else",
                "break",
                "continue",
            ]:
                called.add(name)
        return called


# ==================================================================================
# 3. LOGIC SOLVER
# ==================================================================================


class LogicSolver:
    @staticmethod
    def find_split_index(text: str, operator: str) -> int:
        depth = 0
        op_len = len(operator)
        for i in range(len(text) - op_len + 1):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
            elif depth == 0 and text[i : i + op_len] == operator:
                return i
        return -1

    @staticmethod
    def parse_atomic(
        condition: str, desired_state: bool = True
    ) -> List[Dict[str, Any]]:
        vals = {"_conditions": []}
        match = re.search(
            r"([\w\->\.\[\]\(\)]+)\s*(==|!=|>|<|>=|<=)\s*([a-zA-Z0-9_\(\)\+\-\*\/]+)",
            condition,
        )

        if match:
            var, op, target = match.groups()
            var = var.strip()
            target = target.strip()

            final_val_expr = target

            def get_clean_number(t):
                # Catch hex number (0xFF) or Decimal (123) with tail U, L, UL...
                m = re.match(r"^(\d+|0x[0-9a-fA-F]+)[uUlLfF]*$", t)
                if m:
                    try:
                        return int(eval(m.group(1)))
                    except:
                        return None
                return None

            clean_num = get_clean_number(target)

            if target == "NULL":
                if desired_state:
                    if op == "==":
                        final_val_expr = "<<null>>"
                    elif op == "!=":
                        final_val_expr = "<<malloc 1>>"
                else:
                    if op == "==":
                        final_val_expr = "<<malloc 1>>"
                    elif op == "!=":
                        final_val_expr = "<<null>>"

            elif clean_num is not None:
                target_val = clean_num
                if desired_state:
                    if op == "==":
                        final_val_expr = str(target_val)
                    elif op == "!=":
                        final_val_expr = str(target_val + 1)
                    elif op == ">":
                        final_val_expr = str(target_val + 1)
                    elif op == "<":
                        final_val_expr = str(target_val - 1)
                    elif op == ">=":
                        final_val_expr = str(target_val)
                    elif op == "<=":
                        final_val_expr = str(target_val)
                else:
                    if op == "==":
                        final_val_expr = str(target_val + 1)
                    elif op == "!=":
                        final_val_expr = str(target_val)
                    elif op == ">":
                        final_val_expr = str(target_val)
                    elif op == "<":
                        final_val_expr = str(target_val)
                    elif op == ">=":
                        final_val_expr = str(target_val - 1)
                    elif op == "<=":
                        final_val_expr = str(target_val + 1)
            else:
                # [UPDATED] Handling Macros with MACRO= syntax for Equality
                is_complex_expr = any(op in target for op in ["+", "-", "*", "/"])
                if desired_state:
                    if op == "==" or op == ">=" or op == "<=":
                        final_val_expr = (
                            f"<<{target}>>" if is_complex_expr else f"MACRO={target}"
                        )
                    elif op == "!=" or op == ">":
                        final_val_expr = f"<<{target} + 1>>"
                    elif op == "<":
                        final_val_expr = f"<<{target} - 1>>"
                else:
                    if op == "==" or op == "<=":
                        final_val_expr = f"<<{target} + 1>>"
                    elif op == ">=":
                        final_val_expr = f"<<{target} - 1>>"
                    elif op == "!=" or op == ">" or op == "<":
                        final_val_expr = (
                            f"<<{target}>>" if is_complex_expr else f"MACRO={target}"
                        )

            vals["_conditions"].append(
                f"{var} need {op} {target} -> Set {final_val_expr}"
            )
            vals[var] = final_val_expr
            return [vals]

        return [vals]

    @staticmethod
    def solve_full_coverage(cond: str, desired: bool = True) -> List[Dict[str, Any]]:
        cond = cond.strip()
        while cond.startswith("(") and cond.endswith(")"):
            d = 0
            valid = True
            for i in range(len(cond) - 1):
                if cond[i] == "(":
                    d += 1
                elif cond[i] == ")":
                    d -= 1
                if d == 0:
                    valid = False
                    break
            if valid:
                cond = cond[1:-1].strip()
            else:
                break

        idx = LogicSolver.find_split_index(cond, "||")
        if idx != -1:
            L, R = cond[:idx], cond[idx + 2 :]
            if desired:
                res = []
                tL = LogicSolver.solve_full_coverage(L, True)
                fR = LogicSolver.solve_full_coverage(R, False)
                for l in tL:
                    for r in fR:
                        c = r.copy()
                        c.update(l)
                        c["_conditions"] = l.get("_conditions", []) + r.get(
                            "_conditions", []
                        )
                        res.append(c)
                fL = LogicSolver.solve_full_coverage(L, False)
                tR = LogicSolver.solve_full_coverage(R, True)
                for l in fL:
                    for r in tR:
                        c = l.copy()
                        c.update(r)
                        c["_conditions"] = l.get("_conditions", []) + r.get(
                            "_conditions", []
                        )
                        res.append(c)
                tL2 = LogicSolver.solve_full_coverage(L, True)
                tR2 = LogicSolver.solve_full_coverage(R, True)
                for l in tL2:
                    for r in tR2:
                        c = r.copy()
                        c.update(l)
                        c["_conditions"] = l.get("_conditions", []) + r.get(
                            "_conditions", []
                        )
                        res.append(c)
                return res
            else:
                fL = LogicSolver.solve_full_coverage(L, False)
                fR = LogicSolver.solve_full_coverage(R, False)
                res = []
                for l in fL:
                    for r in fR:
                        c = l.copy()
                        c.update(r)
                        c["_conditions"] = l.get("_conditions", []) + r.get(
                            "_conditions", []
                        )
                        res.append(c)
                return res

        idx = LogicSolver.find_split_index(cond, "&&")
        if idx != -1:
            L, R = cond[:idx], cond[idx + 2 :]
            if desired:
                tL = LogicSolver.solve_full_coverage(L, True)
                tR = LogicSolver.solve_full_coverage(R, True)
                res = []
                for l in tL:
                    for r in tR:
                        c = l.copy()
                        c.update(r)
                        c["_conditions"] = l.get("_conditions", []) + r.get(
                            "_conditions", []
                        )
                        res.append(c)
                return res
            else:
                res = []
                tL = LogicSolver.solve_full_coverage(L, True)
                fR = LogicSolver.solve_full_coverage(R, False)
                for l in tL:
                    for r in fR:
                        c = l.copy()
                        c.update(r)
                        c["_desc"] = " (Neg:Right)"
                        c["_conditions"] = l.get("_conditions", []) + r.get(
                            "_conditions", []
                        )
                        res.append(c)
                fL = LogicSolver.solve_full_coverage(L, False)
                tR = LogicSolver.solve_full_coverage(R, True)
                for l in fL:
                    for r in tR:
                        c = r.copy()
                        c.update(l)
                        c["_desc"] = " (Neg:Left)"
                        c["_conditions"] = l.get("_conditions", [])
                        res.append(c)
                fL2 = LogicSolver.solve_full_coverage(L, False)
                fR2 = LogicSolver.solve_full_coverage(R, False)
                for l in fL2:
                    for r in fR2:
                        c = r.copy()
                        c.update(l)
                        c["_desc"] = " (Neg:Both)"
                        c["_conditions"] = l.get("_conditions", []) + r.get(
                            "_conditions", []
                        )
                        res.append(c)
                return res
        return LogicSolver.parse_atomic(cond, desired)


# ==================================================================================
# 4. VECTORCAST GENERATOR
# ==================================================================================


class VectorCastGenerator:
    def __init__(self, c_file: str, unit_name: str, env_name: str):
        self.c_file = c_file
        self.unit_name = unit_name
        self.env_name = env_name
        self.defines = {}
        self.tst_content = []

    def run(self, output_file: str = "Result_Final.tst"):
        if not os.path.exists(self.c_file):
            logger.error(f"C file not found: {self.c_file}")
            return
        with open(self.c_file, "r", encoding="utf-8") as f:
            content = remove_comments(f.read())
        funcs = CodeParser.extract_functions(content)
        self._write_header()
        for func in funcs:
            if func["func_name"] == "main":
                continue
            self._process_function(func)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(self.tst_content))
        logger.info(f"🚀 DONE! Generated: {output_file}")

    def _write_header(self):
        self.tst_content.extend(
            [
                f"-- VectorCAST Script Generated",
                f"-- Environment: {self.env_name}",
                f"-- Unit: {self.unit_name}",
                "TEST.SCRIPT_FEATURE:C_DIRECT_ARRAY_INDEXING",
                "TEST.SCRIPT_FEATURE:CPP_CLASS_OBJECT_REVISION",
                "TEST.SCRIPT_FEATURE:MULTIPLE_UUT_SUPPORT",
                "TEST.SCRIPT_FEATURE:REMOVED_CL_PREFIX",
                "TEST.SCRIPT_FEATURE:MIXED_CASE_NAMES",
                "TEST.SCRIPT_FEATURE:STATIC_HEADER_FUNCS_IN_UUTS",
                "TEST.SCRIPT_FEATURE:VCAST_MAIN_NOT_RENAMED",
                "--",
                "",
            ]
        )

    def _process_function(self, func):
        fname = func["func_name"]
        body = func["body"]
        self.tst_content.append(f"-- Subprogram: {fname}")
        params_info_list = self._parse_params(func["params"])
        param_names = [p["name"] for p in params_info_list]

        stubs = CodeParser.extract_assignments(body)
        guards = CodeParser.extract_guard_values(body)
        deps = CodeParser.extract_state_dependencies(body)
        path_constr_false, path_constr_true = CodeParser.extract_path_constraints(body)
        loop_ranges = CodeParser.extract_loop_ranges(body)
        called_funcs = CodeParser.extract_called_functions(body)

        scenarios = []
        masked = mask_switch_blocks(body)
        if_structs = CodeParser.extract_if_with_body(masked)

        for ifs in if_structs:
            cond = ifs["condition"].strip()
            blk = ifs["body"]
            if_start_idx = ifs["start_index"]
            if_end_idx = ifs["end_index"]

            pos = LogicSolver.solve_full_coverage(cond, True)
            neg = LogicSolver.solve_full_coverage(cond, False)
            for s in pos:
                s["_branch_node"] = f"[{cond}]"
            for s in neg:
                s["_branch_node"] = f"[!({cond})]"

            ternaries = CodeParser.extract_ternary_operators(blk)
            if ternaries:
                expanded = []
                for base in pos:
                    for t_cond in ternaries:
                        t_pos = LogicSolver.solve_full_coverage(t_cond, True)
                        t_neg = LogicSolver.solve_full_coverage(t_cond, False)
                        for t in t_pos + t_neg:
                            new_s = base.copy()
                            new_s.update(
                                {
                                    k: v
                                    for k, v in t.items()
                                    if k not in ["_conditions", "_desc", "_branch_node"]
                                }
                            )
                            new_s["_conditions"] = base.get("_conditions", []) + t.get(
                                "_conditions", []
                            )
                            expanded.append(new_s)
                if expanded:
                    pos = expanded
            curr = pos + neg

            req_neg = path_constr_false.get(cond, [])
            req_pos = path_constr_true.get(cond, [])

            req_loop = []
            for loop in loop_ranges:
                if loop["start"] < if_start_idx < loop["end"]:
                    req_loop.append(loop["info"])

            parent_ifs = []
            for p_if in if_structs:
                if p_if == ifs:
                    continue
                if (
                    p_if["start_index"] < if_start_idx
                    and p_if["end_index"] > if_end_idx
                ):
                    parent_ifs.append(p_if["condition"])

            for s in curr:
                if "_desc" in s:
                    del s["_desc"]
                branch_stack = []
                for l_info in req_loop:
                    branch_stack.append(f"[{l_info.get('condition', '')}]")
                for p_cond in parent_ifs:
                    branch_stack.append(f"[{p_cond}]")
                if "_branch_node" in s:
                    branch_stack.append(s["_branch_node"])
                s["_branch_path_str"] = "".join(branch_stack)

                for p in req_neg:
                    inps = LogicSolver.solve_full_coverage(p, False)
                    if inps:
                        s.update(
                            {
                                k: v
                                for k, v in inps[0].items()
                                if k != "_conditions" and k not in s
                            }
                        )
                for p in req_pos:
                    inps = LogicSolver.solve_full_coverage(p, True)
                    if inps:
                        s.update(
                            {
                                k: v
                                for k, v in inps[0].items()
                                if k != "_conditions" and k not in s
                            }
                        )

                for l_info in req_loop:
                    iterator = l_info.get("iterator")
                    bound_expr = l_info.get("bound_var")
                    operator = l_info.get("operator")

                    entry_cond = l_info.get("condition", "1")
                    inps = LogicSolver.solve_full_coverage(entry_cond, True)
                    if inps:
                        s.update(
                            {
                                k: v
                                for k, v in inps[0].items()
                                if k != "_conditions" and k not in s
                            }
                        )

                    if iterator and iterator in s:
                        required_val = s[iterator]
                        if bound_expr and (bound_expr in param_names):
                            min_bound_needed = 0
                            try:
                                val_int = int(required_val)
                                if operator == "<":
                                    min_bound_needed = val_int + 1
                                elif operator == "<=":
                                    min_bound_needed = val_int
                                else:
                                    min_bound_needed = val_int + 1
                            except:
                                pass

                            current_bound_val = s.get(bound_expr)
                            # Avoid crash if current_bound_val is a Macro/String
                            if current_bound_val is not None and isinstance(
                                current_bound_val, int
                            ):
                                s[bound_expr] = max(current_bound_val, min_bound_needed)
                            else:
                                if current_bound_val is None:
                                    s[bound_expr] = min_bound_needed
                        del s[iterator]
                scenarios.append(s)

        for loop in loop_ranges:
            l_info = loop["info"]
            iterator = l_info.get("iterator")
            bound_expr = l_info.get("bound_var")
            operator = l_info.get("operator")
            init_func = l_info.get("init_func")

            if iterator and bound_expr and operator:
                s = {}
                vars_in_bound = re.findall(r"[a-zA-Z_]\w*", bound_expr)
                param_var = next((v for v in vars_in_bound if v in param_names), None)

                # STRATEGY 1: Control ITERATOR (Stub or Param)
                final_iter_val = None

                if param_var:
                    # If bound is controlled by param, assume a base value
                    base_val = 5
                    try:
                        eval_expr = re.sub(
                            r"\b" + re.escape(param_var) + r"\b",
                            str(base_val),
                            bound_expr,
                        )
                        bound_val = safe_eval(eval_expr)
                    except:
                        bound_val = None
                    if bound_val is None:
                        bound_val = base_val

                    if bound_val is not None:
                        if operator == "<":
                            final_iter_val = bound_val
                        elif operator == "<=":
                            final_iter_val = bound_val + 1
                        elif operator == ">":
                            final_iter_val = bound_val
                        elif operator == ">=":
                            final_iter_val = bound_val - 1
                        elif operator == "!=":
                            final_iter_val = bound_val
                else:
                    # Bound is local/constant
                    safe_bound = safe_eval(bound_expr)
                    if safe_bound is not None:
                        # Constant bound (e.g. i < 10) -> i = 10
                        if operator == "<":
                            final_iter_val = safe_bound
                        elif operator == "<=":
                            final_iter_val = safe_bound + 1
                        elif operator == ">":
                            final_iter_val = safe_bound
                        elif operator == ">=":
                            final_iter_val = safe_bound - 1
                    else:
                        # Local var bound (e.g. i < length + 1). Heuristic for iter val
                        if "<" in operator:
                            final_iter_val = 100
                        elif ">" in operator:
                            final_iter_val = -1
                        else:
                            final_iter_val = 0

                # Apply Strategy 1
                if final_iter_val is not None:
                    if init_func:
                        s[init_func] = final_iter_val
                        s["_branch_node"] = f"[FALSE-SKIP: {l_info.get('condition')}]"
                        scenarios.append(s)
                    elif iterator in param_names:
                        s[iterator] = final_iter_val
                        s["_branch_node"] = f"[FALSE-SKIP: {l_info.get('condition')}]"
                        scenarios.append(s)

                # STRATEGY 2: Control BOUND (Param)
                # Only if bound depends on a parameter and iterator is NOT controllable (e.g. i=0)
                if param_var and (not init_func) and (iterator not in param_names):
                    s2 = {}
                    # Try to make bound satisfy !condition with fixed iterator (usually 0)
                    # Simple assumption: i=0. Make loop false (0 >= bound)
                    # So set bound param to 0 (for <)
                    if operator == "<":
                        s2[param_var] = 0
                    elif operator == "<=":
                        s2[param_var] = -1
                    # Add more complex solving if needed

                    if s2:
                        s2["_branch_node"] = (
                            f"[FALSE-SKIP-BOUND: {l_info.get('condition')}]"
                        )
                        scenarios.append(s2)

        scenarios.extend(CodeParser.extract_switch_cases(body))
        for t in CodeParser.extract_ternary_operators(body):
            scenarios.extend(LogicSolver.solve_full_coverage(t, True))
            scenarios.extend(LogicSolver.solve_full_coverage(t, False))

        if not scenarios:
            scenarios.append({})

        for i, s in enumerate(scenarios):
            res = s.copy()
            if "_conditions" not in res:
                res["_conditions"] = []

            # Resolve State Dependencies [UPDATED] to handle MACRO= prefix
            for k, v in s.items():
                check_val = str(v)
                if check_val.startswith("<<") and check_val.endswith(">>"):
                    check_val = check_val[2:-2].strip()
                elif check_val.startswith("MACRO="):
                    check_val = check_val.replace("MACRO=", "").strip()

                if k in deps:
                    if check_val in deps[k]:
                        needed = deps[k][check_val]
                        inps = LogicSolver.solve_full_coverage(needed, True)
                        if inps:
                            res.update(
                                {
                                    dk: dv
                                    for dk, dv in inps[0].items()
                                    if dk != "_conditions"
                                }
                            )
                    elif check_val.isdigit() and check_val in deps[k]:
                        needed = deps[k][check_val]
                        inps = LogicSolver.solve_full_coverage(needed, True)
                        if inps:
                            res.update(
                                {
                                    dk: dv
                                    for dk, dv in inps[0].items()
                                    if dk != "_conditions"
                                }
                            )

            self._write_test_case(
                fname, i + 1, res, params_info_list, stubs, guards, body, called_funcs
            )

    def _parse_params(self, params_str):
        info = []
        raw = params_str.split(",") if params_str and "void" not in params_str else []
        for p in raw:
            p = p.strip()
            is_ptr = "*" in p
            parts = p.split()
            if not parts:
                continue
            nm = (
                parts[-1]
                .replace("*", "")
                .replace("[", "")
                .replace("]", "")
                .replace(";", "")
            )
            type_str = " ".join(parts[:-1])
            info.append({"name": nm, "is_ptr": is_ptr, "type": type_str})
        return info

    def _write_test_case(
        self, fname, case_num, scenario, params, stubs, guards, body, called_funcs
    ):
        # 1. Add Guards to scenario if missing
        for k, v in guards.items():
            if k not in scenario:
                scenario[k] = v

        # 2. Build Description
        desc = ""
        raw_c = scenario.get("_conditions", [])
        if raw_c:
            desc = f"[Objective] {', '.join(raw_c)}"
        elif "_branch_node" in scenario:
            desc = f"[Objective] {scenario['_branch_node']}"
        else:
            desc = "[Objective] Check default execution path"

        # 3. Write Header
        cid = f"{case_num:03d}"
        self.tst_content.append("")
        self.tst_content.append(f"--Test Case:{fname}.{cid}")
        self.tst_content.append(f"TEST.UNIT:{self.unit_name}")
        self.tst_content.append(f"TEST.SUBPROGRAM:{fname}")
        self.tst_content.append("TEST.NEW")
        self.tst_content.append(f"TEST.NAME:{fname}.{cid}")

        # 4. Write Notes
        branch_str = scenario.get("_branch_path_str", "")
        if branch_str:
            self.tst_content.append("TEST.NOTES:")
            self.tst_content.append(f"{desc}")
            self.tst_content.append(f"[Conditions] {branch_str}")
            self.tst_content.append(f"[Test method] Requirements-base test")
            self.tst_content.append(f"[Test techniques] Analysis of requirements")
            self.tst_content.append("TEST.END_NOTES:")
        elif "_branch_node" in scenario:  # Notes for Loop Skip cases
            self.tst_content.append("TEST.NOTES:")
            self.tst_content.append(f"{desc}")
            self.tst_content.append("TEST.END_NOTES:")

        cur_stub = set()

        # 5. Handle PARAMETERS
        for p in params:
            nm = p["name"]
            val = str(scenario.get(nm, ""))
            if p["is_ptr"]:
                malloc_size = 1
                for k in scenario.keys():
                    pattern = r"^" + re.escape(nm) + r"\[([^\]]+)\]"
                    match = re.search(pattern, k)
                    if match:
                        idx_content = match.group(1).strip()
                        idx = 0
                        if idx_content.isdigit():
                            idx = int(idx_content)
                        elif idx_content in scenario:
                            try:
                                idx = int(scenario[idx_content])
                            except:
                                idx = 0
                        if idx + 1 > malloc_size:
                            malloc_size = idx + 1
                if "<<" in val or "MACRO=" in val:
                    final_v = val
                elif val == "0":
                    final_v = "<<null>>"
                else:
                    final_v = f"<<malloc {malloc_size}>>"
                self.tst_content.append(
                    f"TEST.VALUE:{self.unit_name}.{fname}.{nm}:{final_v}"
                )
            else:
                has_members = any(
                    k.startswith(nm + ".") or k.startswith(nm + "->")
                    for k in scenario.keys()
                )
                type_str = p.get("type", "")
                is_complex = (
                    "struct" in type_str
                    or "union" in type_str
                    or type_str.endswith("_u")
                )

                # If struct/complex and no direct value, skip (wait for member assignment)
                if not val and (has_members or is_complex):
                    continue
                final_v = val if val else "0"
                self.tst_content.append(
                    f"TEST.VALUE:{self.unit_name}.{fname}.{nm}:{final_v}"
                )

        # 6. Handle GLOBALS, STUBS & MEMBERS
        for k, v in scenario.items():
            if k in ["_conditions", "_desc", "_branch_node", "_branch_path_str"]:
                continue
            is_param_exact = any(x["name"] == k for x in params)
            if is_param_exact:
                continue

            if k in ["NULL", "return"]:
                continue

            vk = k
            if "[" in vk and "]" in vk:

                def replace_idx(m):
                    i = m.group(1).strip()
                    return (
                        f"[{scenario[i]}]"
                        if i in scenario
                        else (f"[{i}]" if i.isdigit() else "[0]")
                    )

                vk = re.sub(r"\[(.*?)\]", replace_idx, vk)

            is_direct_stub = k in called_funcs
            if k in stubs:
                for sf in stubs[k]:
                    if sf not in cur_stub:
                        self.tst_content.append(
                            f"TEST.VALUE:{STUB_UNIT_NAME}.{sf}.return:{v}"
                        )
                        cur_stub.add(sf)
            elif is_direct_stub:
                if k not in cur_stub:
                    self.tst_content.append(
                        f"TEST.VALUE:{STUB_UNIT_NAME}.{k}.return:{v}"
                    )
                    self.tst_content.append(
                        f"TEST.EXPECTED:{STUB_UNIT_NAME}.{k}.hit_count:1"
                    )
                    cur_stub.add(k)
            elif not is_param_exact:
                type_pattern = r"(?:int|float|double|bool|char|void|short|long|[\w]+_t|struct\s+[\w]+|union\s+[\w]+|enum\s+[\w]+)"
                is_loc = re.search(
                    r"\b(?:const\s+)?(?:unsigned\s+)?"
                    + type_pattern
                    + r"\s+(?:[\w\*]+\s+)?\b"
                    + re.escape(k)
                    + r"\b",
                    body,
                )
                if not is_loc:
                    if "->" in vk:
                        vk = vk.replace("->", "[0].")

                    # Check Param Member vs Global Variable
                    root_name = re.split(r"->|\.|\[", vk)[0].replace("*", "").strip()
                    is_param_member = any(x["name"] == root_name for x in params)

                    if is_param_member:
                        self.tst_content.append(
                            f"TEST.VALUE:{self.unit_name}.{fname}.{vk}:{v}"
                        )
                    else:
                        # if variable is global -> add <<GLOBAL>>
                        self.tst_content.append(
                            f"TEST.VALUE:{self.unit_name}.<<GLOBAL>>.{vk}:{v}"
                        )

        for k, v in guards.items():
            if k not in scenario:
                if k in stubs:
                    for sf in stubs[k]:
                        if sf not in cur_stub:
                            self.tst_content.append(
                                f"TEST.VALUE:{STUB_UNIT_NAME}.{sf}.return:{v}"
                            )
                            cur_stub.add(sf)
                else:
                    type_pattern = r"(?:int|float|double|bool|char|void|short|long|[\w]+_t|struct\s+[\w]+|union\s+[\w]+|enum\s+[\w]+)"
                    is_loc = re.search(
                        r"\b(?:const\s+)?(?:unsigned\s+)?"
                        + type_pattern
                        + r"\s+(?:[\w\*]+\s+)?\b"
                        + re.escape(k)
                        + r"\b",
                        body,
                    )
                    if not is_loc:
                        root_name = re.split(r"->|\.|\[", k)[0].replace("*", "").strip()
                        is_param_member = any(x["name"] == root_name for x in params)

                        if is_param_member:
                            self.tst_content.append(
                                f"TEST.VALUE:{self.unit_name}.{fname}.{k}:{v}"
                            )
                        else:
                            self.tst_content.append(
                                f"TEST.VALUE:{self.unit_name}.<<GLOBAL>>.{k}:{v}"
                            )

        self.tst_content.append("TEST.END")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="VectorCAST Test Script Generator")
        parser.add_argument("c_file", help="Path to the C source file")
        parser.add_argument("--unit", required=True, help="Unit name (UUT)")
        parser.add_argument("--env", default="TEST_SV", help="Environment name")
        parser.add_argument(
            "--output", default="Result_Final.tst", help="Output TST file name"
        )
        parser.add_argument(
            "--verbose", action="store_true", help="Enable verbose debug logging"
        )
        args = parser.parse_args()
        if args.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        if not os.path.exists(args.c_file):
            logger.error(f"Source file not found: {args.c_file}")
            exit(1)
        generator = VectorCastGenerator(args.c_file, args.unit, args.env)
        generator.run(args.output)
    else:
        try:
            from Generate_Script_VectorCAST_UI import launch_ui

            launch_ui()
        except ImportError:
            print(
                "Error: Could not import UI module 'Generate_Script_VectorCAST_UI.py'."
            )
