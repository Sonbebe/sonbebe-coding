import re
import os
import argparse
import logging
import sys
from typing import List, Dict, Any, Optional, Tuple, Union, Set

# ==================================================================================
# 0. TREE-SITTER IMPORTS & CONFIGURATION
# ==================================================================================
try:
    from tree_sitter import Language, Parser
    import tree_sitter_c as tsc
    import tree_sitter_cpp as tscpp
except ImportError:
    print(
        "Error: tree-sitter is not installed yet! Please  run cmd: pip install tree-sitter tree-sitter-c tree-sitter-cpp"
    )
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("VCAST_GEN_V2")

STUB_UNIT_NAME = "uut_prototype_stubs"


# ==================================================================================
# 1. UTILITIES & HELPER FUNCTIONS
# ==================================================================================
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


def safe_eval(expr: str, context: Dict[str, Any] = None) -> Optional[Union[int, float]]:
    if not expr:
        return None
    clean_expr = re.sub(r"(\b0x[0-9a-fA-F]+|\b\d+)[uUlLfF]+\b", r"\1", expr)
    clean_expr = clean_expr.replace("(", "").replace(")", "").strip()
    try:
        return int(eval(clean_expr, {}, {}))
    except:
        pass

    if context:
        try:
            expanded = clean_expr
            # Sort keys by length desc to replace longest match first
            for k in sorted(context.keys(), key=len, reverse=True):
                if k in expanded:
                    val = str(context[k])
                    expanded = re.sub(r"\b" + re.escape(k) + r"\b", val, expanded)
            return int(eval(expanded, {}, {}))
        except:
            pass
    return None


# ==================================================================================
# 2. TREE-SITTER AST PARSER
# ==================================================================================
class TreeSitterParser:
    def __init__(self, is_cpp=False):
        self.is_cpp = is_cpp
        self.language = Language(tscpp.language() if is_cpp else tsc.language())
        self.parser = Parser(self.language)

    def _get_text(self, node, source_bytes: bytes) -> str:
        if node is None:
            return ""
        return source_bytes[node.start_byte : node.end_byte].decode(
            "utf-8", errors="ignore"
        )

    def _find_child_of_type(self, node, types: List[str]):
        if node.type in types:
            return node
        for child in node.children:
            found = self._find_child_of_type(child, types)
            if found:
                return found
        return None

    def _parse_parameters(self, params_node, source_bytes) -> List[Dict[str, Any]]:
        params = []
        if not params_node:
            return params

        for child in params_node.children:
            if (
                child.type == "parameter_declaration"
                or child.type == "optional_parameter_declaration"
            ):
                type_node = child.child_by_field_name("type")
                decl_node = child.child_by_field_name("declarator")

                type_str = self._get_text(type_node, source_bytes)
                is_ptr = False
                is_ref = False
                name_str = ""

                if decl_node:
                    if decl_node.type == "pointer_declarator":
                        is_ptr = True
                        id_node = self._find_child_of_type(
                            decl_node, ["identifier", "field_identifier"]
                        )
                        name_str = self._get_text(id_node, source_bytes)
                    elif decl_node.type == "reference_declarator":
                        is_ref = True
                        id_node = self._find_child_of_type(
                            decl_node, ["identifier", "field_identifier"]
                        )
                        name_str = self._get_text(id_node, source_bytes)
                    else:
                        name_str = self._get_text(decl_node, source_bytes)

                if name_str and name_str != "void":
                    params.append(
                        {
                            "name": name_str.strip(),
                            "type": type_str.strip(),
                            "is_ptr": is_ptr,
                            "is_ref": is_ref,
                        }
                    )
        return params

    def extract_functions(self, c_file_path: str) -> List[Dict[str, Any]]:
        with open(c_file_path, "rb") as f:
            source_bytes = f.read()

        tree = self.parser.parse(source_bytes)
        root_node = tree.root_node
        functions = []
        # Manual traversal to avoid Tree-sitter Query version incompatibilities
        def traverse(node):
            if node.type == "function_definition":
                decl_node = node.child_by_field_name("declarator")
                body_node = node.child_by_field_name("body")
                if decl_node and body_node:
                    func_name_node = self._find_child_of_type(
                        decl_node, ["identifier", "scoped_identifier", "destructor_name"]
                    )
                    params_node = self._find_child_of_type(decl_node, ["parameter_list"])
                    func_name = (
                        self._get_text(func_name_node, source_bytes)
                        if func_name_node
                        else "unknown"
                    )
                    params = self._parse_parameters(params_node, source_bytes)

                    body_text = self._get_text(body_node, source_bytes)
                    if body_text.startswith("{") and body_text.endswith("}"):
                        body_text = body_text[1:-1]
                    
                    # [FIX] Remove comments to avoid regex pitfalls in CodeParser
                    body_text = remove_comments(body_text)
                    
                    if func_name and func_name != "main":
                        functions.append({
                            "func_name": func_name,
                            "params": params,
                            "body": body_text
                        })
                return

            for child in node.children:
                traverse(child)

        traverse(root_node)

        return functions

    def extract_if_nodes(self, body_text: str) -> List[Dict[str, Any]]:
        """
        Extracts IF statements using Tree-sitter to avoid regex pitfalls (comments, strings).
        Returns a list of dicts compatible with the legacy CodeParser format.
        """
        # Wrap body to make it a valid function for parsing
        wrapped_text = f"void wrapper() {{\n{body_text}\n}}"
        source_bytes = wrapped_text.encode("utf-8")

        tree = self.parser.parse(source_bytes)
        root_node = tree.root_node
        
        wrapper_header_len = len("void wrapper() {\n".encode("utf-8"))
        results = []

        def get_char_index(byte_offset):
            if byte_offset < wrapper_header_len:
                return 0
            segment = source_bytes[wrapper_header_len:byte_offset]
            return len(segment.decode("utf-8", errors="ignore"))

        def is_inside_switch(node):
            p = node.parent
            while p:
                if p.type == "switch_statement":
                    return True
                p = p.parent
            return False

        def traverse(node):
            if node.type == "if_statement" and not is_inside_switch(node):
                cond_node = node.child_by_field_name("condition")
                then_node = node.child_by_field_name("consequence")
                
                if cond_node:
                    cond_text = self._get_text(cond_node, source_bytes)
                    if cond_text.startswith("(") and cond_text.endswith(")"):
                        cond_text = cond_text[1:-1].strip()
                    
                    body_code = self._get_text(then_node, source_bytes) if then_node else ""
                    if body_code.strip().startswith("{") and body_code.strip().endswith("}"):
                        body_code = body_code.strip()[1:-1].strip()

                    # Use then_node end to avoid treating 'else if' as nested in parent's THEN block
                    end_byte = then_node.end_byte if then_node else node.end_byte

                    results.append({
                        "condition": cond_text,
                        "body": body_code,
                        "start_index": get_char_index(node.start_byte),
                        "end_index": get_char_index(end_byte)
                    })
            
            for child in node.children:
                traverse(child)

        traverse(root_node)
        return results

    def extract_assignment_conditions(self, body_text: str) -> Dict[str, Dict[str, List[str]]]:
        """
        Traverses the AST to find assignments and the conditions required to reach them.
        Handles nested IF/ELSE logic correctly.
        Returns: {var_name: {value_str: [condition_path_1, condition_path_2]}}
        """
        wrapped_text = f"void wrapper() {{\n{body_text}\n}}"
        source_bytes = wrapped_text.encode("utf-8")
        tree = self.parser.parse(source_bytes)
        
        deps = {} 

        def traverse(node, current_conds):
            # [FIX] Handle Compound Statement (Block) - Sequential processing with guard detection
            if node.type in ["compound_statement", "translation_unit"]:
                block_conds = current_conds[:]
                for child in node.children:
                    traverse(child, block_conds)
                    
                    # Check for Guard Clause: if (cond) { ... return; }
                    if child.type == "if_statement":
                        cond_node = child.child_by_field_name("condition")
                        then_node = child.child_by_field_name("consequence")
                        
                        # Check if THEN block definitely returns (Early Return)
                        is_returning = False
                        if then_node:
                            if then_node.type == "return_statement":
                                is_returning = True
                            elif then_node.type == "compound_statement":
                                # Check if any direct child is a return statement
                                for sub in then_node.children:
                                    if sub.type == "return_statement":
                                        is_returning = True
                                        break
                        
                        if is_returning and cond_node:
                            raw_cond = self._get_text(cond_node, source_bytes)
                            # [FIX] Strip ALL outer parentheses safely (check balance)
                            while raw_cond.startswith('(') and raw_cond.endswith(')'):
                                depth = 0
                                is_full = True
                                for i in range(len(raw_cond)-1):
                                    if raw_cond[i] == '(': depth += 1
                                    elif raw_cond[i] == ')': depth -= 1
                                    if depth == 0: is_full = False; break
                                if is_full: raw_cond = raw_cond[1:-1].strip()
                                else: break
                            # Add negation of this condition to subsequent statements in this block
                            block_conds = block_conds + [f"!({raw_cond})"]
                return

            # Handle IF / ELSE IF / ELSE
            if node.type == "if_statement":
                cond_node = node.child_by_field_name("condition")
                then_node = node.child_by_field_name("consequence")
                else_node = node.child_by_field_name("alternative")
                
                if cond_node:
                    raw_cond = self._get_text(cond_node, source_bytes)
                    # [FIX] Strip ALL outer parentheses safely (check balance)
                    while raw_cond.startswith('(') and raw_cond.endswith(')'):
                        depth = 0
                        is_full = True
                        for i in range(len(raw_cond)-1):
                            if raw_cond[i] == '(': depth += 1
                            elif raw_cond[i] == ')': depth -= 1
                            if depth == 0: is_full = False; break
                        if is_full: raw_cond = raw_cond[1:-1].strip()
                        else: break
                    
                    # Traverse THEN block with condition
                    traverse(then_node, current_conds + [raw_cond])
                    
                    # Traverse ELSE block with negated condition
                    if else_node:
                        traverse(else_node, current_conds + [f"!({raw_cond})"])
                return

            # Handle Assignments (LHS = RHS)
            if node.type == "assignment_expression":
                left = node.child_by_field_name("left")
                right = node.child_by_field_name("right")
                if left and right:
                    var_name = self._get_text(left, source_bytes).strip()
                    val_str = self._get_text(right, source_bytes).strip()
                    
                    if var_name not in deps: deps[var_name] = {}
                    if val_str not in deps[var_name]: deps[var_name][val_str] = []
                    
                    cond_str = " && ".join(current_conds) if current_conds else "1"
                    deps[var_name][val_str].append(cond_str)
                return

            for child in node.children:
                traverse(child, current_conds)

        traverse(tree.root_node, [])
        return deps

    def extract_declarations(self, body_text: str) -> Set[str]:
        """
        Extracts local variable names to avoid expensive regex checks later.
        """
        wrapped_text = f"void wrapper() {{\n{body_text}\n}}"
        source_bytes = wrapped_text.encode("utf-8")
        tree = self.parser.parse(source_bytes)
        locals_set = set()

        def get_ident(node):
            while node.type in ["init_declarator", "pointer_declarator", "array_declarator", "parenthesized_declarator"]:
                node = node.child_by_field_name("declarator")
            if node.type == "identifier":
                return self._get_text(node, source_bytes)
            return None

        def traverse(node):
            if node.type == "declaration":
                for child in node.children:
                    name = get_ident(child)
                    if name:
                        locals_set.add(name)
            for child in node.children:
                traverse(child)
        
        traverse(tree.root_node)
        return locals_set


# ==================================================================================
# 3. LEGACY CODE PARSER & LogicSolver
# ==================================================================================
class CodeParser:

    # --- [NEW: EXPECTED LOGIC] Hàm trích xuất giá trị Return ---
    @staticmethod
    def extract_expected_return(text: str) -> Optional[str]:
        matches = list(re.finditer(r"\breturn\s+([^;]+)\s*;", text))
        if not matches:
            return None
        raw_return = matches[-1].group(1).strip()
        clean_return = re.sub(r"^\(.*?\)\s*", "", raw_return)
        if clean_return in ["NULL", "nullptr"]:
            return "<<null>>"
        return clean_return

    # --- [NEW: EXPECTED LOGIC] Hàm trích xuất phép gán cho Output (Con trỏ/Tham chiếu) ---
    @staticmethod
    def extract_expected_outputs(
        text: str, param_names_ptr_ref: List[str]
    ) -> Dict[str, str]:
        expected = {}
        if not param_names_ptr_ref:
            return expected

        names_pattern = "|".join(re.escape(n) for n in param_names_ptr_ref)
        pattern = (
            r"(?:\*\s*)?\b("
            + names_pattern
            + r")\b((?:\[.*?\]|\-\>[\w\.]+)?)\s*=(?!=)\s*([^;]+);"
        )

        for match in re.finditer(pattern, text):
            pname = match.group(1)
            suffix = match.group(2).strip()
            rhs = match.group(3).strip()

            # VectorCAST syntax for pointers: ptr->val is ptr[0].val
            if suffix.startswith("->"):
                suffix = "[0]." + suffix[2:]

            rhs = re.sub(r"^\(.*?\)\s*", "", rhs)
            if rhs in ["NULL", "nullptr"]:
                rhs = "<<null>>"

            expected_key = pname + suffix
            expected[expected_key] = rhs

        return expected

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
                    loop_var, init_func = m_func.groups()
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
                        operator, bound_expr = bound_match.groups()
                        bound_var = bound_expr.split("//")[0].strip()
                    else:
                        operator, bound_var = None, None
                    loop_info = {
                        "condition": entry_condition,
                        "iterator": loop_var,
                        "bound_var": bound_var,
                        "operator": operator,
                        "init_val": init_val,
                        "init_func": init_func,
                    }
                else:
                    loop_info = {"condition": cond_part}

            body_start = paren_end + 1
            while body_start < len(body) and body[body_start].isspace():
                body_start += 1
            if body_start < len(body) and body[body_start] == "{":
                body_end = find_matching_paren(body, body_start, "{", "}")
                if body_end != -1:
                    loops.append(
                        {"start": body_start, "end": body_end, "info": loop_info}
                    )
        return loops

    @staticmethod
    def extract_path_constraints(
        body: str,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        constraints_false, constraints_true = {}, {}

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
                                parse_recursive(
                                    text[e_open + 1 : e_close],
                                    persistent_guards + [cond],
                                    parent_conds,
                                )
                                idx = e_close + 1
                                continue
                        else:
                            semi = text.find(";", e_open)
                            if semi != -1:
                                parse_recursive(
                                    text[e_open : semi + 1],
                                    persistent_guards + [cond],
                                    parent_conds,
                                )
                                idx = semi + 1
                                continue
                else:
                    chain_negations = []
                idx = next_scan

        parse_recursive(body, [], [])
        return constraints_false, constraints_true

    @staticmethod
    def extract_switch_cases(
        body: str, defines: Dict[str, Any], ast_parser: TreeSitterParser
    ) -> List[Dict[str, Any]]:
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
                        # Check defines first
                        if val_str in defines:
                            real_val = defines[val_str]
                        elif val_str.isdigit():
                            real_val = int(val_str)
                        else:
                            real_val = f"MACRO={val_str}"

                    base = {
                        var: real_val,
                        "_conditions": [f"{var} == {val_str}"],
                        "_branch_path_str": f"[{var} == {val_str}]",
                    }
                    # [REFACTOR] Use AST parser for nested IFs instead of Regex
                    nested_nodes = ast_parser.extract_if_nodes(c_block)
                    if nested_nodes:
                        for n_node in nested_nodes:
                            n_cond = n_node["condition"]
                            pos_s = LogicSolver.solve_full_coverage(
                                n_cond, defines, True
                            )
                            neg_s = LogicSolver.solve_full_coverage(
                                n_cond, defines, False
                            )
                            for s in pos_s + neg_s:
                                comb = base.copy()
                                comb.update(
                                    {
                                        k: v
                                        for k, v in s.items()
                                        if k not in ["_conditions", "_desc"]
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
        for match in re.finditer(r"([^;{}]+?)\s*\?\s*([^;:]+)\s*:\s*([^;,\)]+)", body):
            raw = match.group(1).strip()
            clean = re.sub(r"^.*?(?:return|(?<!=)=(?!=)|,)\s*", "", raw).strip()
            if clean:
                conds.append(clean)
        return conds

    @staticmethod
    def extract_assignments(body: str) -> Dict[str, List[str]]:
        assigns = {}
        for match in re.finditer(
            r"\b(\w+)\s*=(?!=)\s*(?:\([\w\s\*]+\)\s*)?(\w+)\s*\(", body
        ):
            var, func = match.groups()
            if func not in ["if", "while", "for", "switch", "sizeof"]:
                if var not in assigns:
                    assigns[var] = []
                if func not in assigns[var]:
                    assigns[var].append(func)
        return assigns

    @staticmethod
    def extract_guard_values(body: str, defines: Dict[str, Any]) -> Dict[str, Any]:
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
                neg = LogicSolver.solve_full_coverage(cond, defines, False)
                if neg:
                    guards.update(
                        {
                            k: v
                            for k, v in neg[0].items()
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

    @staticmethod
    def parse_header_defines(
        c_file_path: str, include_dirs: List[str] = None
    ) -> Dict[str, Any]:
        """
        Scans for #include "..." or <...> and searches in include_dirs to parse defines.
        """
        defines = {}
        base_dir = os.path.dirname(os.path.abspath(c_file_path))

        # Default search path includes the directory of the source file
        search_paths = [base_dir]
        if include_dirs:
            # Resolve relative paths in arguments to absolute paths
            search_paths.extend([os.path.abspath(d) for d in include_dirs])

        logger.info(f"Header Search Paths: {search_paths}")

        try:
            with open(c_file_path, "r", encoding="utf-8") as f:
                content = remove_comments(f.read())
        except:
            return {}

        # Matches: #include "path/file.h" OR #include <path/file.h>
        includes = re.findall(r'#include\s+["<]([\w\.\/\\\-]+)[">]', content)
        for inc in includes:
            found = False
            for path in search_paths:
                h_path = os.path.join(path, inc)
                if os.path.exists(h_path):
                    found = True
                    try:
                        with open(h_path, "r", encoding="utf-8") as f:
                            h_content = remove_comments(f.read())
                            for match in re.finditer(
                                r"#define\s+(\w+)(?!\()\s+(.+)", h_content
                            ):
                                key = match.group(1)
                                val = match.group(2).split("//")[0].strip()
                                num = safe_eval(val, defines)
                                defines[key] = num if num is not None else val
                            
                            # [FIX] Parse Enums to resolve values like ECU_STATE_NORMAL
                            for enum_match in re.finditer(r"enum\s*(?:\w+\s*)?\{([^}]+)\}", h_content, re.DOTALL):
                                enum_body = enum_match.group(1)
                                val_counter = 0
                                for item in enum_body.split(','):
                                    item = item.strip()
                                    if not item: continue
                                    item = re.sub(r"/\*.*?\*/", "", item, flags=re.DOTALL)
                                    item = item.split('//')[0].strip()
                                    if not item: continue
                                    
                                    if '=' in item:
                                        parts = item.split('=')
                                        nm = parts[0].strip()
                                        v_str = parts[1].strip()
                                        ev = safe_eval(v_str, defines)
                                        if ev is not None:
                                            val_counter = ev
                                        defines[nm] = val_counter
                                    else:
                                        defines[nm] = val_counter
                                    val_counter += 1
                    except:
                        logger.warning(f"  -> Error reading header: {h_path}")
                    break  # Stop searching other paths if found

            if not found:
                logger.warning(f"  -> Header NOT found: {inc}")

        return defines

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
        condition: str, defines: Dict[str, Any], desired_state: bool = True
    ) -> List[Dict[str, Any]]:
        # [FIX] Handle negation !(A > B) -> A <= B
        condition = condition.strip()
        is_negated = False
        if condition.startswith("!(") and condition.endswith(")"):
             # Check if parens are balanced for the content
             inner = condition[2:-1]
             d = 0
             balanced = True
             for c in inner:
                 if c == '(': d+=1
                 elif c == ')': d-=1
                 if d < 0: balanced = False; break
             if balanced and d == 0:
                 condition = inner.strip()
                 is_negated = True
        
        if is_negated:
            desired_state = not desired_state

        vals = {"_conditions": []}
        match = re.search(
            r"([\w\->\.\[\]\(\)]+)\s*(==|!=|(?<!-)>|<|>=|<=)\s*([a-zA-Z0-9_\(\)\+\-\*\/]+)",
            condition,
        )

        if match:
            var, op, target = match.groups()
            var, target = var.strip(), target.strip()
            
            # [FIX] Clean variable name (remove extra parens like '((input')
            while var.startswith('(') and var.endswith(')'):
                var = var[1:-1].strip()
            var = var.lstrip('(')

            # [FIX] Handle case where LHS is number and RHS is variable (e.g. 0 < size)
            if (var.isdigit() or re.match(r"^\d", var)) and re.match(r"^[a-zA-Z_]", target):
                 swap_map = {'==': '==', '!=': '!=', '<': '>', '<=': '>=', '>': '<', '>=': '<='}
                 var, target = target, var
                 op = swap_map.get(op, op)

            if var.isdigit() or re.match(r"^\d", var):
                vals["_conditions"].append(f"Check {var} {op} {target}")
                return [vals]

            final_val_expr = target
            clean_target = target.replace("(", "").replace(")", "").strip()
            target_val = safe_eval(clean_target)

            if target_val is None and clean_target in defines:
                val = defines[clean_target]
                if isinstance(val, (int, float)):
                    target_val = val
                elif isinstance(val, str):
                    v_num = safe_eval(val)
                    if v_num is not None:
                        target_val = v_num

            need_exact = (
                (op in ["==", ">=", "<="])
                if desired_state
                else (op in ["!=", ">", "<"])
            )
            is_simple_id = (
                bool(re.match(r"^[a-zA-Z_]\w*$", clean_target))
                and not clean_target.isdigit()
                and clean_target not in ["NULL", "true", "false", "nullptr"]
            )

            if need_exact and is_simple_id:
                final_val_expr = f"MACRO={clean_target}"

            # 2. CALCULATION: Use Numeric Value if we need to change it (+1/-1)
            elif target_val is not None:
                if desired_state:
                    if op in ["==", ">=", "<="]:
                        final_val_expr = str(target_val)
                    elif op in ["!=", ">"]:
                        final_val_expr = str(target_val + 1)
                    elif op == "<":
                        final_val_expr = str(target_val - 1)
                else:
                    if op in ["==", "<="]:
                        final_val_expr = str(target_val + 1)
                    elif op in ["!=", ">", "<"]:
                        final_val_expr = str(target_val)
                    elif op == ">=":
                        final_val_expr = str(target_val - 1)
            else:
                if clean_target in ["NULL", "nullptr"]:
                    final_val_expr = (
                        "<<null>>"
                        if (desired_state and op == "==")
                        or (not desired_state and op == "!=")
                        else "<<malloc 1>>"
                    )
                elif clean_target in ["true", "false"]:
                    must_equal = True
                    if op == "==":
                        must_equal = desired_state
                    elif op == "!=":
                        must_equal = not desired_state
                    
                    if must_equal:
                        final_val_expr = clean_target
                    else:
                        final_val_expr = "false" if clean_target == "true" else "true"
                elif not need_exact and is_simple_id:
                    # [FIX] Fallback for inequality of unknown MACRO
                    # VectorCAST does not allow expressions like MACRO+1. Use a distinct literal '99'.
                    final_val_expr = "99"
                    vals["_conditions"].append(
                        f"NOTE: {var} {op} {target} -> Set {final_val_expr} (Fallback for != MACRO)"
                    )
                else:
                    final_val_expr = (
                        f"<<{target}>>"
                        if any(o in target for o in ["+", "-", "*", "/"])
                        else f"MACRO={target}"
                    )
                    vals["_conditions"].append(
                        f"NOTE: {var} {op} {target} -> Set {final_val_expr} (Fallback)"
                    )

            if var not in vals:
                vals[var] = final_val_expr

            display_op = op
            if not desired_state:
                if op == "==":
                    display_op = "!="
                elif op == "!=":
                    display_op = "=="
                elif op == ">":
                    display_op = "<="
                elif op == "<":
                    display_op = ">="
                elif op == ">=":
                    display_op = "<"
                elif op == "<=":
                    display_op = ">"

            # Add descriptive condition
            if "NOTE" not in str(vals["_conditions"]):
                vals["_conditions"].append(
                    f"{var} need {display_op} {target} -> Set {final_val_expr}"
                )
        else:
            # [FIX] Handle implicit boolean checks (e.g. "if (ptr->valid)")
            # If regex didn't match an operator, treat as boolean check
            clean_cond = condition.strip()
            is_not = False
            if clean_cond.startswith('!'):
                clean_cond = clean_cond[1:].strip()
                is_not = True
                
            while clean_cond.startswith('(') and clean_cond.endswith(')'):
                 clean_cond = clean_cond[1:-1].strip()

            if re.match(r"^[\w\->\.\[\]\(\)]+$", clean_cond):
                 var = clean_cond
                 effective_true = desired_state if not is_not else not desired_state
                 val = "1" if effective_true else "0"
                 op = "!=" if effective_true else "=="
                 vals[var] = val
                 vals["_conditions"].append(f"Check {var} {op} 0 -> Set {val}")

        return [vals]

    @staticmethod
    def solve_full_coverage(
        cond: str, defines: Dict[str, Any], desired: bool = True
    ) -> List[Dict[str, Any]]:
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
                for tL_item in LogicSolver.solve_full_coverage(L, defines, True):
                    for fR_item in LogicSolver.solve_full_coverage(R, defines, False):
                        c = fR_item.copy()
                        c.update(tL_item)
                        c["_conditions"] = tL_item.get("_conditions", []) + fR_item.get(
                            "_conditions", []
                        )
                        res.append(c)
                for fL_item in LogicSolver.solve_full_coverage(L, defines, False):
                    for tR_item in LogicSolver.solve_full_coverage(R, defines, True):
                        c = fL_item.copy()
                        c.update(tR_item)
                        c["_conditions"] = fL_item.get("_conditions", []) + tR_item.get(
                            "_conditions", []
                        )
                        res.append(c)
                for tL2_item in LogicSolver.solve_full_coverage(L, defines, True):
                    for tR2_item in LogicSolver.solve_full_coverage(R, defines, True):
                        c = tR2_item.copy()
                        c.update(tL2_item)
                        c["_conditions"] = tL2_item.get(
                            "_conditions", []
                        ) + tR2_item.get("_conditions", [])
                        res.append(c)
                return res
            else:
                res = []
                for fL_item in LogicSolver.solve_full_coverage(L, defines, False):
                    for fR_item in LogicSolver.solve_full_coverage(R, defines, False):
                        c = fL_item.copy()
                        c.update(fR_item)
                        c["_conditions"] = fL_item.get("_conditions", []) + fR_item.get(
                            "_conditions", []
                        )
                        res.append(c)
                return res

        idx = LogicSolver.find_split_index(cond, "&&")
        if idx != -1:
            L, R = cond[:idx], cond[idx + 2 :]
            if desired:
                res = []
                for tL_item in LogicSolver.solve_full_coverage(L, defines, True):
                    for tR_item in LogicSolver.solve_full_coverage(R, defines, True):
                        c = tL_item.copy()
                        c.update(tR_item)
                        c["_conditions"] = tL_item.get("_conditions", []) + tR_item.get(
                            "_conditions", []
                        )
                        res.append(c)
                return res
            else:
                res = []
                for tL_item in LogicSolver.solve_full_coverage(L, defines, True):
                    for fR_item in LogicSolver.solve_full_coverage(R, defines, False):
                        c = tL_item.copy()
                        c.update(fR_item)
                        c["_desc"] = " (Neg:Right)"
                        c["_conditions"] = tL_item.get("_conditions", []) + fR_item.get(
                            "_conditions", []
                        )
                        res.append(c)
                for fL_item in LogicSolver.solve_full_coverage(L, defines, False):
                    for tR_item in LogicSolver.solve_full_coverage(R, defines, True):
                        c = tR_item.copy()
                        c.update(fL_item)
                        c["_desc"] = " (Neg:Left)"
                        c["_conditions"] = fL_item.get("_conditions", [])
                        res.append(c)
                for fL2_item in LogicSolver.solve_full_coverage(L, defines, False):
                    for fR2_item in LogicSolver.solve_full_coverage(R, defines, False):
                        c = fR2_item.copy()
                        c.update(fL2_item)
                        c["_desc"] = " (Neg:Both)"
                        c["_conditions"] = fL2_item.get(
                            "_conditions", []
                        ) + fR2_item.get("_conditions", [])
                        res.append(c)
                return res
        return LogicSolver.parse_atomic(cond, defines, desired)

    @staticmethod
    def solve_for_variable(expr: str, target_val: int, defines: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
        """
        Numerically solves 'expr(var) >= target_val' for an unknown variable 'var'.
        Supports +, -, *, /, pow, etc. via Python eval().
        """
        # 1. Clean expression
        expr = re.sub(r"/\*.*?\*/", "", expr)
        py_expr = expr.replace("/", "//") # Integer division
        
        # 2. Identify the variable
        # Capture identifiers including struct members and array access
        ids = set(re.findall(r"[a-zA-Z_][\w\->\.\[\]]*", py_expr))
        
        knowns = set(defines.keys()) | {'pow', 'abs', 'min', 'max', 'sizeof'}
        unknowns = [i for i in ids if i not in knowns and not i.isdigit()]
        
        if len(unknowns) != 1:
            return None, None
            
        var_name = unknowns[0]
        
        # 3. Define evaluation function
        def f(x):
            # Replace variable with value x
            eval_str = re.sub(r"\b" + re.escape(var_name) + r"\b", str(x), py_expr)
            try:
                ctx = defines.copy()
                ctx.update({'pow': pow, 'abs': abs})
                return int(eval(eval_str, {"__builtins__": {}}, ctx))
            except:
                return -float('inf')

        # 4. Solve f(x) >= target_val using Binary Search
        # Check monotonicity
        y0 = f(0)
        y1 = f(1)
        
        if y1 >= y0: # Increasing
            low, high = 0, 2000 # Reasonable range for loop bounds
            ans = None
            while low <= high:
                mid = (low + high) // 2
                if f(mid) >= target_val:
                    ans = mid
                    high = mid - 1
                else:
                    low = mid + 1
            return var_name, ans
        else: # Decreasing
            if y0 >= target_val:
                return var_name, 0
            
        return None, None


# ==================================================================================
# 4. VECTORCAST GENERATOR
# ==================================================================================
class VectorCastGenerator:
    def __init__(
        self,
        c_file: str,
        unit_name: str,
        env_name: str,
        include_dirs: List[str] = None,
        is_cpp=False,
    ):
        self.c_file = c_file
        self.unit_name = unit_name
        self.env_name = env_name
        self.include_dirs = include_dirs if include_dirs else []
        self.defines = {}
        self.tst_content = []

        self.is_cpp = is_cpp
        self.ast_parser = TreeSitterParser(is_cpp=is_cpp)
        self.defined_internal_funcs = set()

    def run(self, output_file: str = "Result_Final.tst"):
        if not os.path.exists(self.c_file):
            logger.error(f"File not found: {self.c_file}")
            return

        # [STEP 1] Load Defines from Headers
        self.defines = CodeParser.parse_header_defines(self.c_file, self.include_dirs)
        logger.info(f"Loaded {len(self.defines)} defines from headers.")

        # Use AST to get infomation of function and param
        logger.info("Parsing AST...")
        funcs = self.ast_parser.extract_functions(self.c_file)
        self.defined_internal_funcs = {f["func_name"] for f in funcs}

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
        params_info_list = func["params"]
        param_names = [p["name"] for p in params_info_list]

        self.tst_content.append(f"-- Subprogram: {fname}")

        # --- [NEW: EXPECTED LOGIC] Lấy Expected Default cho toàn bộ hàm ---
        ptr_ref_names = [
            p["name"] for p in params_info_list if p["is_ptr"] or p.get("is_ref")
        ]
        base_expected_output = CodeParser.extract_expected_outputs(body, ptr_ref_names)
        base_expected_return = CodeParser.extract_expected_return(body)
        # ------------------------------------------------------------------

        stubs = CodeParser.extract_assignments(body)
        guards = CodeParser.extract_guard_values(body, self.defines)
        deps = self.ast_parser.extract_assignment_conditions(body)
        path_constr_false, path_constr_true = CodeParser.extract_path_constraints(body)
        loop_ranges = CodeParser.extract_loop_ranges(body)
        called_funcs = CodeParser.extract_called_functions(body)
        local_vars = self.ast_parser.extract_declarations(body)

        scenarios = []
        if_structs = self.ast_parser.extract_if_nodes(body)

        for ifs in if_structs:
            cond = ifs["condition"].strip()
            blk = ifs["body"]
            if_start_idx, if_end_idx = ifs["start_index"], ifs["end_index"]

            # --- [NEW: EXPECTED LOGIC] Lấy Expected Cụ Thể cho nhánh IF này ---
            specific_expected_output = CodeParser.extract_expected_outputs(
                blk, ptr_ref_names
            )
            specific_expected_return = CodeParser.extract_expected_return(blk)

            pos = LogicSolver.solve_full_coverage(cond, self.defines, True)
            neg = LogicSolver.solve_full_coverage(cond, self.defines, False)
            for s in pos:
                s["_branch_node"] = f"[{cond}]"
            for s in neg:
                s["_branch_node"] = f"[!({cond})]"

            ternaries = CodeParser.extract_ternary_operators(blk)
            if ternaries:
                expanded = []
                for base in pos:
                    for t_cond in ternaries:
                        t_pos = LogicSolver.solve_full_coverage(
                            t_cond, self.defines, True
                        )
                        t_neg = LogicSolver.solve_full_coverage(
                            t_cond, self.defines, False
                        )
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

            req_neg, req_pos = path_constr_false.get(cond, []), path_constr_true.get(
                cond, []
            )
            req_loop = [
                loop["info"]
                for loop in loop_ranges
                if loop["start"] < if_start_idx < loop["end"]
            ]
            parent_ifs = [
                p_if["condition"]
                for p_if in if_structs
                if p_if != ifs
                and p_if["start_index"] < if_start_idx
                and p_if["end_index"] > if_end_idx
            ]

            for s in curr:
                # --- [NEW: EXPECTED LOGIC] Gán Expected vào Scenario ---
                exp_combined = base_expected_output.copy()
                exp_combined.update(specific_expected_output)  # Specific ghi đè Base
                s["_expected"] = exp_combined

                final_return = (
                    specific_expected_return
                    if specific_expected_return is not None
                    else base_expected_return
                )
                if final_return is not None:
                    s["_expected_return"] = final_return
                # --------------------------------------------------------

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
                    inps = LogicSolver.solve_full_coverage(p, self.defines, False)
                    if inps:
                        s.update(
                            {
                                k: v
                                for k, v in inps[0].items()
                                if k != "_conditions" and k not in s
                            }
                        )
                for p in req_pos:
                    inps = LogicSolver.solve_full_coverage(p, self.defines, True)
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
                    inps = LogicSolver.solve_full_coverage(
                        entry_cond, self.defines, True
                    )
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
                        min_bound_needed = 0
                        try:
                            val_int = int(required_val)
                            min_bound_needed = (
                                val_int + 1 if operator in ["<", "!="] else val_int
                            )
                        except:
                            pass

                        current_bound_val = (
                            self.defines.get(bound_expr, s.get(bound_expr))
                            if bound_expr
                            else None
                        )
                        target_val = (
                            max(current_bound_val, min_bound_needed)
                            if isinstance(current_bound_val, int)
                            else min_bound_needed
                        )
                        if bound_expr:
                            if re.match(r"^[\w\->\.\[\]]+$", bound_expr):
                                s[bound_expr] = target_val
                            else:
                                # [FIX] Use numerical solver for complex expressions
                                var_name, solved_val = LogicSolver.solve_for_variable(bound_expr, target_val, self.defines)
                                if var_name and solved_val is not None:
                                    s[var_name] = solved_val
                                    if bound_expr in s: del s[bound_expr]
                        del s[iterator]
                scenarios.append(s)

        scenarios.extend(
            CodeParser.extract_switch_cases(body, self.defines, self.ast_parser)
        )
        for t in CodeParser.extract_ternary_operators(body):
            scenarios.extend(
                LogicSolver.solve_full_coverage(t, self.defines, True)
                + LogicSolver.solve_full_coverage(t, self.defines, False)
            )

        if not scenarios:
            scenarios.append({})

        # [OPTIMIZATION] Deduplicate scenarios based on conditions
        unique_scenarios = []
        seen_conditions = set()
        for s in scenarios:
            # --- [NEW: EXPECTED LOGIC] Đảm bảo các scenario từ switch/ternary cũng có Base Expected
            if "_expected" not in s:
                s["_expected"] = base_expected_output.copy()
            if "_expected_return" not in s and base_expected_return is not None:
                s["_expected_return"] = base_expected_return
            # ----------------------------------------------------------------------------------

            cond_key = tuple(sorted(s.get("_conditions", [])))
            if cond_key not in seen_conditions:
                seen_conditions.add(cond_key)
                unique_scenarios.append(s)

        for i, s in enumerate(unique_scenarios):
            res = s.copy()
            if "_conditions" not in res:
                res["_conditions"] = []

            # Resolve State Dependencies [UPDATED] to handle MACRO= prefix
            for k, v in s.items():
                if k in stubs or k in ["_expected", "_expected_return"]:
                    continue

                check_val = (
                    str(v)[2:-2].strip()
                    if str(v).startswith("<<") and str(v).endswith(">>")
                    else str(v).replace("MACRO=", "").strip()
                )
                
                if k in deps:
                    # Find matching value in dependencies
                    found_conds = []
                    if check_val in deps[k]:
                        found_conds = deps[k][check_val]
                    else:
                        # Try to infer dependency from conditions (e.g. != MACRO)
                        # If we need '!= NORMAL', we can pick 'ERROR' or 'WARNING' from deps
                        must_be = set()
                        must_not_be = set()
                        
                        for cond_str in s.get("_conditions", []):
                            # Parse: "var need op target -> Set"
                            pattern = re.escape(k) + r"\s+need\s+(==|!=)\s+(.*?)\s+->\s+Set"
                            m = re.search(pattern, cond_str)
                            if m:
                                op = m.group(1)
                                target = m.group(2).strip()
                                if op == "==":
                                    must_be.add(target)
                                elif op == "!=":
                                    must_not_be.add(target)
                        
                        # Filter candidates from deps[k]
                        candidates = []
                        # [FIX] Iterate in reverse to prefer later assignments (deeper logic)
                        # This avoids picking 'INIT' states that are immediately overwritten.
                        all_vals = list(deps[k].keys())
                        for possible_val in reversed(all_vals):
                            if must_be and possible_val not in must_be:
                                continue
                            if possible_val in must_not_be:
                                continue
                            candidates.append(possible_val)
                        
                        if candidates:
                            # Pick first valid candidate (e.g. ECU_STATE_ERROR)
                            target_val_key = candidates[0]
                            found_conds = deps[k][target_val_key]
                            
                            # Update result to reflect the actual value we are forcing
                            if re.match(r"^[a-zA-Z_]\w*$", target_val_key):
                                res[k] = f"MACRO={target_val_key}"
                            else:
                                res[k] = target_val_key
                    
                    if found_conds:
                        # Use the first valid condition path found to set this state
                        cond_to_solve = found_conds[0]
                        if cond_to_solve != "1":
                            inps = LogicSolver.solve_full_coverage(cond_to_solve, self.defines, True)
                            if inps:
                                res.update(
                                    {
                                        dk: dv
                                        for dk, dv in inps[0].items()
                                        if dk != "_conditions" and dk not in res
                                    }
                                )

            self._write_test_case(
                fname, i + 1, res, params_info_list, stubs, guards, body, called_funcs, deps, local_vars
            )

    def _write_test_case(
        self, fname, case_num, scenario, params, stubs, guards, body, called_funcs, deps=None, local_vars=None
    ):
        # 1. Add Guards to scenario if missing
        for k, v in guards.items():
            if k not in scenario:
                scenario[k] = v

        desc = (
            f"[Objective] {', '.join(scenario.get('_conditions', []))}"
            if scenario.get("_conditions")
            else (
                f"[Objective] {scenario['_branch_node']}"
                if "_branch_node" in scenario
                else "[Objective] Check default execution path"
            )
        )

        self.tst_content.extend(
            [
                "",
                f"--Test Case:{fname}.{case_num:03d}",
                f"TEST.UNIT:{self.unit_name}",
                f"TEST.SUBPROGRAM:{fname}",
                "TEST.NEW",
                f"TEST.NAME:{fname}.{case_num:03d}",
            ]
        )

        branch_str = scenario.get("_branch_path_str", "")
        if branch_str or "_branch_node" in scenario:
            self.tst_content.extend(["TEST.NOTES:", desc])
            if branch_str:
                self.tst_content.extend(
                    [
                        f"[Conditions] {branch_str}",
                        "[Test method] Requirements-base test",
                        "[Test techniques] Analysis of requirements",
                    ]
                )
            self.tst_content.append("TEST.END_NOTES:")

        cur_stub = set()

        # Helper function to write Stub logic (Internal vs External)
        def write_stub_entry(func_name, ret_val):
            if func_name in self.defined_internal_funcs:
                self.tst_content.extend(
                    [
                        f"TEST.STUB:{self.unit_name}.{func_name}",
                        f"TEST.VALUE:{self.unit_name}.{func_name}.return:{ret_val}",
                    ]
                )
            else:
                # External Stub Logic
                self.tst_content.append(
                    f"TEST.VALUE:{STUB_UNIT_NAME}.{func_name}.return:{ret_val}"
                )

        # 5. Handle PARAMETERS
        for p in params:
            nm = p["name"]
            # Check if parameter is used as an array index or member base in the scenario
            val = str(scenario.get(nm, ""))

            # [FIX] Check if parameter is overwritten by a stub function based on dependencies
            # This handles cases like: if (a > 5) b = random();
            if deps and nm in deps and val and "<<" not in val:
                # Prepare evaluation context
                eval_ctx = self.defines.copy()
                for sk, sv in scenario.items():
                    # Convert VectorCAST key to C-style key for matching conditions
                    c_key = sk.replace("[0].", "->").replace(".", "->")
                    clean_sv = str(sv).replace("MACRO=", "").strip()
                    if not (
                        clean_sv.startswith("<<")
                        or clean_sv.startswith("[")
                        or clean_sv == ""
                    ):
                        eval_ctx[c_key], eval_ctx[sk] = clean_sv, clean_sv

                for rhs, conditions in deps[nm].items():
                    # Check if RHS looks like a function call
                    if "(" in rhs and ")" in rhs:
                        for cond in conditions:
                            if cond == "1" or safe_eval(cond, eval_ctx):
                                m_func = re.search(r"(\w+)\s*\(", rhs)
                                if m_func:
                                    stub_func_name = m_func.group(1)
                                    if (
                                        stub_func_name in self.defined_internal_funcs
                                        or stub_func_name in called_funcs
                                    ):
                                        if stub_func_name not in cur_stub:
                                            write_stub_entry(stub_func_name, val)
                                            cur_stub.add(stub_func_name)
                                break

            if p["is_ptr"]:
                malloc_size = 1
                for k in scenario.keys():
                    match = re.search(r"^" + re.escape(nm) + r"\[([^\]]+)\]", k)
                    if match:
                        idx_content = match.group(1).strip()
                        idx = (
                            int(idx_content)
                            if idx_content.isdigit()
                            else (
                                int(scenario.get(idx_content, 0))
                                if idx_content in scenario
                                else 0
                            )
                        )
                        if idx + 1 > malloc_size:
                            malloc_size = idx + 1
                final_v = (
                    val
                    if ("<<" in val or "MACRO=" in val)
                    else ("<<null>>" if val == "0" else f"<<malloc {malloc_size}>>")
                )
                self.tst_content.append(
                    f"TEST.VALUE:{self.unit_name}.{fname}.{nm}:{final_v}"
                )
            elif p.get("is_ref"):
                self.tst_content.append(
                    f"TEST.VALUE:{self.unit_name}.{fname}.{nm}:{val if val else '0'}"
                )
            else:
                # Check if this parameter is a struct/union and we are setting its members
                has_members = any(
                    k.startswith(nm + ".") or k.startswith(nm + "->")
                    for k in scenario.keys()
                )
                type_str = p.get("type", "")
                # If no direct value and no member setting, skip (unless it's a simple type)
                if not val and (
                    has_members or any(t in type_str for t in ["struct", "union", "_u"])
                ):
                    continue
                self.tst_content.append(
                    f"TEST.VALUE:{self.unit_name}.{fname}.{nm}:{val if val else '0'}"
                )

        # 6. Handle GLOBALS, STUBS & MEMBERS
        for k, v in scenario.items():
            if ( k in ["_conditions","_desc","_branch_node","_branch_path_str","NULL","return","_expected","_expected_return"]
                or k.isdigit()
                or any(x["name"] == k for x in params)
            ):
                continue

            # Handle array indexing in keys for VectorCAST format
            vk = re.sub(
                r"\[(.*?)\]",
                lambda m: (
                    f"[{scenario[i]}]"
                    if (i := m.group(1).strip()) in scenario
                    else (f"[{i}]" if i.isdigit() else "[0]")
                ),
                k,
            )
            clean_k = k.split("(")[0].strip() if "(" in k else k

            if k in stubs:
                # Evaluate dependencies to only stub the active function matching the execution path
                stubbed = False
                if deps and k in deps:
                    eval_ctx = self.defines.copy()
                    for sk, sv in scenario.items():
                        c_key = sk.replace("[0].", "->").replace(".", "->")
                        clean_sv = str(sv).replace("MACRO=", "").strip()
                        if not (clean_sv.startswith("<<") or clean_sv.startswith("[") or clean_sv == ""):
                            eval_ctx[c_key] = clean_sv

                    # Traverse reversed to prioritize later assignments in the code block
                    for rhs, conditions in reversed(list(deps[k].items())):
                        if "(" in rhs and ")" in rhs:
                            for cond in conditions:
                                # Check if the condition for this assignment is met in current scenario
                                if cond == "1" or safe_eval(cond, eval_ctx):
                                    m_func = re.search(r"(\w+)\s*\(", rhs)
                                    if m_func:
                                        sf = m_func.group(1)
                                        if sf not in cur_stub:
                                            write_stub_entry(sf, v)
                                            cur_stub.add(sf)
                                        stubbed = True
                                        break
                        if stubbed:
                            break
                
                # Fallback to legacy logic if dependency resolution fails
                if not stubbed:
                    for sf in stubs[k]:
                        if sf not in cur_stub:
                            write_stub_entry(sf, v)
                            cur_stub.add(sf)
            
            # Only treat as stub if it looks like a function call AND is not a local variable
            # This prevents variables named like functions (e.g. 'status') from being treated as stubs
            elif (clean_k in called_funcs or clean_k in self.defined_internal_funcs) and "(" in k:
                if clean_k not in cur_stub:
                    write_stub_entry(clean_k, v)
                    cur_stub.add(clean_k)
            else:
                if not local_vars or k not in local_vars:
                    vk = vk.replace("->", "[0].")
                    root_name = re.split(r"->|\.|\[", vk)[0].replace("*", "").strip()
                    prefix = (
                        fname
                        if any(x["name"] == root_name for x in params)
                        else "<<GLOBAL>>"
                    )
                    self.tst_content.append(
                        f"TEST.VALUE:{self.unit_name}.{prefix}.{vk}:{v}"
                    )

        # --- [NEW: EXPECTED LOGIC] Viết ra dòng TEST.EXPECTED ---
        expected_outputs = scenario.get("_expected", {})
        for exp_key, exp_val in expected_outputs.items():
            resolved_val = safe_eval(exp_val, self.defines)
            final_val = str(resolved_val) if resolved_val is not None else exp_val

            # Nếu không phải là một số (ví dụ: gán từ 1 biến local khác), xuất comment cảnh báo
            if (
                not final_val.isdigit()
                and final_val != "<<null>>"
                and not final_val.startswith("MACRO=")
            ):
                self.tst_content.append(f"-- TBD: Dynamic variable assignment detected")
                self.tst_content.append(
                    f"-- TEST.EXPECTED:{self.unit_name}.{fname}.{exp_key}:{final_val}"
                )
            else:
                self.tst_content.append(
                    f"TEST.EXPECTED:{self.unit_name}.{fname}.{exp_key}:{final_val}"
                )

        if "_expected_return" in scenario:
            exp_ret = scenario["_expected_return"]
            resolved_ret = safe_eval(exp_ret, self.defines)
            final_ret = str(resolved_ret) if resolved_ret is not None else exp_ret

            if (
                not final_ret.isdigit()
                and final_ret != "<<null>>"
                and not final_ret.startswith("MACRO=")
            ):
                self.tst_content.append(f"-- TBD: Dynamic return value detected")
                self.tst_content.append(
                    f"-- TEST.EXPECTED:{self.unit_name}.{fname}.return:{final_ret}"
                )
            else:
                self.tst_content.append(
                    f"TEST.EXPECTED:{self.unit_name}.{fname}.return:{final_ret}"
                )
        # --------------------------------------------------------

        self.tst_content.append("TEST.END")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VectorCAST Test Script Generator (AST Hybrid)"
    )
    parser.add_argument("c_file", help="Path to the source file or directory")
    parser.add_argument("--unit", required=False, help="Unit name (UUT) - required if single file")
    parser.add_argument("--env", default="TEST_SV", help="Environment name")
    parser.add_argument(
        "--output", default="Result_Final.tst", help="Output TST file name"
    )
    parser.add_argument(
        "--cpp",
        action="store_true",
        help="Enable C++ mode (parses references, classes, etc.)",
    )
    parser.add_argument(
        "--include", action="append", help="Include directories for headers"
    )

    args = parser.parse_args()

    input_path = args.c_file
    if os.path.isdir(input_path):
        c_files = []
        for root_dir, dirs, files in os.walk(input_path):
            for f in files:
                if f.endswith(('.c', '.cpp', '.cxx', '.cc')):
                    c_files.append(os.path.join(root_dir, f))
    else:
        c_files = [input_path]
        if not args.unit:
            print("Error: --unit is required when processing a single file.")
            sys.exit(1)

    for c_file in c_files:
        if os.path.isdir(input_path):
            unit_name = os.path.splitext(os.path.basename(c_file))[0]
            output_file = f"Result_{unit_name}.tst"
        else:
            unit_name = args.unit
            output_file = args.output

        generator = VectorCastGenerator(
            c_file, unit_name, args.env, args.include, args.cpp
        )
        generator.run(output_file)
