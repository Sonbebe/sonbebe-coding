import re
import os
import argparse
import logging
import sys
from typing import List, Dict, Any, Optional, Tuple, Union, Set

# ==================================================================================
# 0. LOGGING CONFIGURATION
# ==================================================================================
# Default config for CLI. UI will override handlers.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger("VCAST_GEN")

# ==================================================================================
# 1. UTILITIES & HELPER FUNCTIONS
# ==================================================================================
STUB_UNIT_NAME = "uut_prototype_stubs"

def remove_comments(text: str) -> str:
    """Remove /*...*/ and //... comments from C code."""
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'//.*?\n', '\n', text)
    return text

def find_matching_paren(text: str, start_idx: int, open_char: str = '(', close_char: str = ')') -> int:
    """Find the corresponding closing parenthesis position."""
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
    """Calculate simple expression value (supports U, L, F suffixes)."""
    if not expr:
        return None
    clean_expr = re.sub(r'(\b0x[0-9a-fA-F]+|\b\d+)[uUlLfF]+\b', r'\1', expr)
    clean_expr = clean_expr.replace('(', '').replace(')', '').strip()
    if re.match(r'^[0-9a-fA-FxX\s\+\-\*\/\.]+$', clean_expr):
        try:
            return int(eval(clean_expr))
        except Exception:
            return None
    return None

def load_defines(folder_path: str) -> Dict[str, Any]:
    """Read and parse #define and enum from header folder."""
    defines = {}
    logger.info(f"Loading headers from: {folder_path}")
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".h"):
                path = os.path.join(folder_path, filename)
                logger.debug(f"Processing header: {filename}")
                with open(path, 'r', encoding='utf-8') as f:
                    content = remove_comments(f.read())
                    
                    # 1. Parse #define
                    regex = r'^\s*#define\s+(\w+)(?:[ \t]+(.+))?$'
                    matches = re.findall(regex, content, flags=re.MULTILINE)
                    for name, val_str in matches:
                        if not val_str:
                            continue
                        val = safe_eval(val_str.strip())
                        defines[name] = val if val is not None else val_str.strip()

                    # 2. EXTRACT ENUM
                    regex_enum = r'enum\s*(?:\w+\s*)?\{([^}]+)\}'
                    matches_enum = re.findall(regex_enum, content, flags=re.DOTALL)
                    for enum_body in matches_enum:
                        items = [x.strip() for x in enum_body.split(',') if x.strip()]
                        current_val = 0
                        for item in items:
                            if '=' in item:
                                parts = item.split('=')
                                name = parts[0].strip()
                                val = safe_eval(parts[1].strip())
                                if val is not None:
                                    current_val = val
                                defines[name] = current_val
                            else:
                                defines[item.strip()] = current_val
                            current_val += 1
    else:
        logger.error(f"Header folder not found: {folder_path}")
    
    return defines

def mask_switch_blocks(body: str) -> str:
    """Mask switch content to avoid mis-parsing IF inside."""
    masked_body = list(body)
    for match in re.finditer(r'\bswitch\s*\((.*?)\)\s*\{', body):
        start = match.start()
        brace_open = match.end() - 1
        brace_close = find_matching_paren(body, brace_open, '{', '}')
        if brace_close != -1:
            for i in range(start, brace_close + 1):
                masked_body[i] = ' '
    return "".join(masked_body)

# ==================================================================================
# 2. CODE PARSER (EXTRACTION LOGIC)
# ==================================================================================

class CodeParser:
    @staticmethod
    def extract_functions(content: str) -> List[Dict[str, str]]:
        """Extract list of functions from source code."""
        header_pattern = re.compile(r'([\w\*\s]+?)\s+(\w+)\s*\((.*?)\)\s*\{', re.DOTALL)
        CONTROL_KEYWORDS = ["if", "else", "switch", "case", "default", "for", "while", "do", "return", "break", "continue", "struct"]
        functions = []
        pos = 0
        
        logger.info("Scanning functions...")
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
            body_end_index = find_matching_paren(content, start_index, '{', '}')
            
            if body_end_index != -1:
                logger.info(f"Found function: {func_name}") 
                body = content[start_index+1 : body_end_index]
                functions.append({
                    "return_type": return_type,
                    "func_name": func_name,
                    "params": params_raw,
                    "body": body
                })
                pos = body_end_index + 1
            else:
                pos = match.end()
        return functions

    @staticmethod
    def extract_if_with_body(body_content: str) -> List[Dict[str, Any]]:
        """Get IF condition and content inside."""
        if_structures = []
        for match in re.finditer(r'\bif\s*\(', body_content):
            start_paren = match.end() - 1
            end_paren = find_matching_paren(body_content, start_paren)
            if end_paren == -1:
                continue
            
            condition = body_content[start_paren+1 : end_paren]
            
            # Get body
            body_start = end_paren + 1
            while body_start < len(body_content) and body_content[body_start].isspace():
                body_start += 1
            
            code_block = ""
            struct_end_index = end_paren # Default end if no body
            
            if body_start < len(body_content):
                if body_content[body_start] == '{':
                    body_end = find_matching_paren(body_content, body_start, '{', '}')
                    if body_end != -1:
                        code_block = body_content[body_start+1 : body_end]
                        struct_end_index = body_end
                else:
                    semi = body_content.find(';', body_start)
                    if semi != -1:
                        code_block = body_content[body_start : semi+1]
                        struct_end_index = semi + 1
            
            if_structures.append({
                'condition': condition, 
                'body': code_block, 
                'start_index': match.start(),
                'end_index': struct_end_index # [NEW] Added end_index for nesting check
            })
        return if_structures

    @staticmethod
    def extract_if_conditions(body_content: str) -> List[str]:
        """Only get condition string in IF."""
        conditions = []
        for match in re.finditer(r'\bif\s*\(', body_content):
            start = match.end() - 1
            end = find_matching_paren(body_content, start)
            if end != -1:
                conditions.append(body_content[start+1 : end])
        return conditions

    @staticmethod
    def extract_loop_ranges(body: str) -> List[Dict[str, Any]]:
        """Analyze FOR, WHILE, DO-WHILE loops to find range and condition info."""
        loops = []
        
        # 1. FOR Loops
        for match in re.finditer(r'\bfor\s*\(', body):
            start = match.start()
            paren_start = match.end() - 1
            paren_end = find_matching_paren(body, paren_start)
            if paren_end == -1:
                continue

            # Parse (init; cond; step)
            for_args = body[paren_start+1 : paren_end] 
            parts = for_args.split(';')
            entry_condition = "1" 
            loop_info = {'condition': "1"}
            
            if len(parts) >= 2:
                init_part = parts[0].strip()
                cond_part = parts[1].strip()
                
                init_match = re.search(r'(?:[\w]+\s+)?(\w+)\s*=\s*([-\w\d]+)', init_part)
                if init_match:
                    loop_var, init_val = init_match.groups()
                    entry_condition = re.sub(r'\b' + re.escape(loop_var) + r'\b', init_val, cond_part)
                    logger.debug(f"[LOOP-FOR] '{cond_part}' -> Entry: '{entry_condition}'")

                    bound_match = re.search(r'\b' + re.escape(loop_var) + r'\b\s*(?:<|<=)\s*(\w+)', cond_part)
                    bound_var = bound_match.group(1) if bound_match else None
                    
                    loop_info = {
                        'condition': entry_condition,
                        'iterator': loop_var,
                        'bound_var': bound_var
                    }
                else:
                    entry_condition = cond_part
                    loop_info = {'condition': entry_condition}
                    logger.debug(f"[LOOP-FOR-FALLBACK] '{cond_part}' -> Entry: '{entry_condition}'")

            # Find Loop Body to get range
            body_start = paren_end + 1
            while body_start < len(body) and body[body_start].isspace():
                body_start += 1
            
            if body_start < len(body) and body[body_start] == '{':
                body_end = find_matching_paren(body, body_start, '{', '}')
                if body_end != -1:
                    loops.append({
                        'start': body_start,
                        'end': body_end,
                        'info': loop_info
                    })

        # 2. WHILE Loops
        for match in re.finditer(r'\bwhile\s*\(', body):
            start = match.start()
            paren_start = match.end() - 1
            paren_end = find_matching_paren(body, paren_start)
            if paren_end == -1:
                continue
            
            condition = body[paren_start+1 : paren_end].strip()
            
            # Check for body start (must be { for standard while loop block)
            body_start = paren_end + 1
            while body_start < len(body) and body[body_start].isspace():
                body_start += 1
            
            if body_start < len(body) and body[body_start] == '{':
                body_end = find_matching_paren(body, body_start, '{', '}')
                if body_end != -1:
                    logger.debug(f"[LOOP-WHILE] Condition: '{condition}'")
                    loops.append({
                        'start': body_start,
                        'end': body_end,
                        'info': {'condition': condition}
                    })

        # 3. DO-WHILE Loops
        for match in re.finditer(r'\bdo\s*\{', body):
            start = match.start()
            brace_open = match.end() - 1
            brace_close = find_matching_paren(body, brace_open, '{', '}')
            if brace_close != -1:
                # Do-while always enters, so condition is effectively 1 for entry
                logger.debug(f"[LOOP-DO-WHILE] Body found. Entry condition: 1")
                loops.append({
                    'start': brace_open,
                    'end': brace_close,
                    'info': {'condition': "1"}
                })

        return loops

    @staticmethod
    def extract_path_constraints(body: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Analyze If-Else flow to find Preconditions."""
        constraints_false = {}
        constraints_true = {}
        
        def parse_recursive(text, current_guards, parent_conds):
            idx = 0
            persistent_guards = current_guards[:]
            chain_negations = []
            
            while idx < len(text):
                match = re.search(r'\bif\s*\(', text[idx:])
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
                
                cond = text[p_open+1 : p_close].strip()
                
                if cond not in constraints_false:
                    constraints_false[cond] = []
                if cond not in constraints_true:
                    constraints_true[cond] = []
                
                # Combine guards
                total_neg = list(set(persistent_guards + chain_negations))
                constraints_false[cond].extend(total_neg)
                constraints_true[cond].extend(parent_conds)
                
                # Find Body
                rest = p_close + 1
                while rest < len(text) and text[rest].isspace():
                    rest += 1
                
                body_start = rest
                body_end = -1
                has_brace = False
                
                if rest < len(text) and text[rest] == '{':
                    has_brace = True
                    body_end = find_matching_paren(text, rest, '{', '}')
                else:
                    semi = text.find(';', rest)
                    if semi != -1:
                        body_end = semi + 1
                
                if body_end == -1:
                    idx = rest
                    continue
                
                inner_body = text[body_start+1 : body_end] if has_brace else text[body_start : body_end]
                
                # Recurse
                parse_recursive(inner_body, persistent_guards, parent_conds + [cond])
                
                # Check Guard (Break/Return)
                if re.search(r'\b(break|continue)\s*;|\breturn\b', inner_body):
                    if cond not in persistent_guards:
                        persistent_guards.append(cond)
                
                # Check Else
                next_scan = body_end + 1
                if re.match(r'\s*else', text[next_scan:]):
                    else_m = re.match(r'\s*else', text[next_scan:])
                    else_end = next_scan + else_m.end()
                    
                    if re.match(r'\s*if\s*\(', text[else_end:]):
                        chain_negations.append(cond)
                        idx = else_end
                        continue 
                    else:
                        chain_negations = [] 
                        e_open = else_end
                        while e_open < len(text) and text[e_open].isspace():
                            e_open += 1
                        if e_open < len(text) and text[e_open] == '{':
                            e_close = find_matching_paren(text, e_open, '{', '}')
                            if e_close != -1:
                                e_body = text[e_open+1 : e_close]
                                parse_recursive(e_body, persistent_guards + [cond], parent_conds)
                                idx = e_close + 1
                                continue
                        else:
                            semi = text.find(';', e_open)
                            if semi != -1:
                                e_body = text[e_open : semi+1]
                                parse_recursive(e_body, persistent_guards + [cond], parent_conds)
                                idx = semi + 1
                                continue
                else:
                    chain_negations = []
                
                idx = next_scan
        
        parse_recursive(body, [], [])
        return constraints_false, constraints_true

    @staticmethod
    def extract_switch_cases(body: str, defines: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract test cases from Switch."""
        scenarios = []
        pos = 0
        while pos < len(body):
            match = re.search(r'\bswitch\s*\((.*?)\)\s*\{', body[pos:])
            if not match:
                break
            
            var = match.group(1).strip()
            start = pos + match.start()
            brace_open = pos + match.end() - 1
            brace_close = find_matching_paren(body, brace_open, '{', '}')
            
            if brace_close != -1:
                switch_body = body[brace_open+1 : brace_close]
                iter_cases = list(re.finditer(r'\b(case\s+([a-zA-Z0-9_]+)|default)\s*:', switch_body))
                
                for i, m in enumerate(iter_cases):
                    c_head_end = m.end()
                    c_code_end = iter_cases[i+1].start() if i+1 < len(iter_cases) else len(switch_body)
                    c_block = switch_body[c_head_end : c_code_end]
                    
                    is_def = m.group(1).startswith('default')
                    val_str = "default" if is_def else m.group(2)
                    
                    real_val = 99
                    if not is_def:
                        if val_str.isdigit():
                            real_val = int(val_str)
                        elif val_str in defines:
                            v = defines[val_str]
                            real_val = v if isinstance(v, int) else 0
                        else:
                            real_val = 10

                    base = {var: real_val, '_conditions': [f"{var} == {val_str}"]}
                    # [NEW] Note for switch
                    base['_branch_path_str'] = f"[{var} == {val_str}]"
                    
                    # Nested IFs in Case
                    nested = CodeParser.extract_if_conditions(c_block)
                    if nested:
                        for n_cond in nested:
                            pos_s = LogicSolver.solve_full_coverage(n_cond, defines, True)
                            neg_s = LogicSolver.solve_full_coverage(n_cond, defines, False)
                            for s in pos_s + neg_s:
                                comb = base.copy()
                                comb.update({k:v for k,v in s.items() if k != '_conditions' and k != '_desc'})
                                comb['_conditions'] = base['_conditions'] + s.get('_conditions', [])
                                scenarios.append(comb)
                    else:
                        scenarios.append(base)
                pos = brace_close + 1
            else:
                pos = start + 1
        return scenarios

    @staticmethod
    def extract_ternary_operators(body: str) -> List[str]:
        """Extract ternary operator conditions."""
        conds = []
        regex = r'([^;{}]+?)\s*\?\s*([^;:]+)\s*:\s*([^;,\)]+)'
        for match in re.finditer(regex, body):
            raw = match.group(1).strip()
            clean = re.sub(r'^.*?(?:return|(?<!=)=(?!=)|,)\s*', '', raw).strip()
            if clean:
                conds.append(clean)
        return conds

    @staticmethod
    def extract_assignments(body: str) -> Dict[str, List[str]]:
        """Find variables assigned from function (Stub)."""
        assigns = {}
        regex = r'\b(\w+)\s*=\s*(?:\([\w\s\*]+\)\s*)?(\w+)\s*\('
        for match in re.finditer(regex, body):
            var, func = match.groups()
            if func not in ["if", "while", "for", "switch", "sizeof"]:
                if var not in assigns:
                    assigns[var] = []
                if func not in assigns[var]:
                    assigns[var].append(func)
                logger.debug(f"[STUB] '{var}' <= '{func}'")
        return assigns

    @staticmethod
    def extract_state_dependencies(body: str, defines: Dict[str, Any]) -> Dict[str, Dict[int, str]]:
        """Find State Dependency."""
        deps = {}
        for match in re.finditer(r'\b(?:if|else\s+if)\s*\(', body):
            start = match.end() - 1
            end = find_matching_paren(body, start)
            if end == -1:
                continue
            cond = body[start+1 : end].strip()
            
            rest = end + 1
            while rest < len(body) and body[rest].isspace():
                rest += 1
            blk = ""
            if rest < len(body):
                if body[rest] == '{':
                    e = find_matching_paren(body, rest, '{', '}')
                    if e != -1:
                        blk = body[rest+1:e]
                else:
                    s = body.find(';', rest)
                    if s != -1:
                        blk = body[rest:s+1]
            
            for am in re.finditer(r'([\w\->\.]+)\s*=\s*([\w\d_]+)\s*;', blk):
                var, val_str = am.groups()
                var, val_str = var.strip(), val_str.strip()
                real = 10
                if val_str.isdigit():
                    real = int(val_str)
                elif val_str in defines:
                    v = defines[val_str]
                    real = v if isinstance(v, int) else 0
                
                if var not in deps:
                    deps[var] = {}
                deps[var][real] = cond
                logger.debug(f"[DEP] {var}={val_str} needs {cond}")
        return deps

    @staticmethod
    def extract_guard_values(body: str, defines: Dict[str, Any]) -> Dict[str, Any]:
        """Find safe values to bypass Guard Clauses."""
        guards = {}
        # IF Guards
        for match in re.finditer(r'\bif\s*\(', body):
            start = match.end() - 1
            end = find_matching_paren(body, start)
            if end == -1:
                continue
            cond = body[start+1 : end]
            
            rest = end + 1
            while rest < len(body) and body[rest].isspace():
                rest += 1
            blk = ""
            if rest < len(body):
                if body[rest] == '{':
                    e = find_matching_paren(body, rest, '{', '}')
                    if e != -1:
                        blk = body[rest+1:e]
                else:
                    s = body.find(';', rest)
                    if s != -1:
                        blk = body[rest:s+1]
            
            if blk and re.search(r'return\s*[;]', blk):
                logger.debug(f"[GUARD] Found at: {cond.strip()}")
                neg = LogicSolver.solve_full_coverage(cond, defines, False)
                if neg:
                    s = neg[0]
                    guards.update({k:v for k,v in s.items() if k not in ['_conditions', '_desc']})
                    logger.debug(f"      -> Auto-Fix: {s}")
        return guards

# ==================================================================================
# 3. LOGIC SOLVER (MC/DC ENGINE)
# ==================================================================================

class LogicSolver:
    @staticmethod
    def find_split_index(text: str, operator: str) -> int:
        depth = 0
        op_len = len(operator)
        for i in range(len(text) - op_len + 1):
            if text[i] == '(':
                depth += 1
            elif text[i] == ')':
                depth -= 1
            elif depth == 0 and text[i:i+op_len] == operator:
                return i
        return -1

    @staticmethod
    def parse_atomic(condition: str, defines: Dict[str, Any], desired_state: bool = True, context_body: Optional[str] = None) -> List[Dict[str, Any]]:
        vals = {'_conditions': []}
        match = re.search(r'([\w\->\.\[\]]+)\s*(==|!=|>|<|>=|<=)\s*([\w\(\)\+\-\*\/_]+)', condition)
        if match:
            var, op, target = match.groups()
            
            # Clean target (remove parens if any)
            while target.endswith(')') and target.count('(') < target.count(')'):
                target = target[:-1]
            target = target.strip()
            
            target_val = 10; is_var = False
            
            if target == 'true': target_val = 1
            elif target == 'false': target_val = 0
            elif target == 'NULL': target_val = 0
            elif target in defines:
                v = defines[target]
                target_val = v if isinstance(v, int) else 10
            elif target.isdigit():
                target_val = int(target)
            else:
                ev = safe_eval(target)
                if ev is not None:
                    target_val = ev
                else: 
                    is_var = True
                    # [FIX] Handle LHS number (e.g. 0 < total -> total = 1)
                    lhs_is_number = var.isdigit()
                    lhs_val = int(var) if lhs_is_number else 0
                    
                    if lhs_is_number:
                        if desired_state:
                            if op == '<': target_val = lhs_val + 1
                            elif op == '<=': target_val = lhs_val
                            elif op == '>': target_val = lhs_val - 1
                            elif op == '>=': target_val = lhs_val
                            elif op == '==': target_val = lhs_val
                            elif op == '!=': target_val = lhs_val + 1
                        else:
                            if op == '<': target_val = lhs_val
                            elif op == '<=': target_val = lhs_val - 1
                            elif op == '>': target_val = lhs_val
                            elif op == '>=': target_val = lhs_val + 1
                            elif op == '==': target_val = lhs_val + 1
                            elif op == '!=': target_val = lhs_val
                    else:
                        target_val = 10
                
            final_op = op
            if not desired_state:
                inv = {'==':'!=', '!=':'==', '>':'<=', '<':'>=', '>=':'<', '<=':'>'}
                final_op = inv.get(op, op)
                
            desc = target if is_var else str(target_val)
            if target in ['NULL', 'true', 'false']: desc = target
            vals['_conditions'].append(f"{var} {final_op} {desc}")
            
            if target == 'NULL':
                vals[var] = "<<null>>" if final_op == '==' else "<<malloc 1>>"
                return [vals]
                
            # [FIX] Only assign value if LHS is not a number (e.g. 0 < count)
            if not var.isdigit():
                if final_op == '==': vals[var] = target_val
                elif final_op == '>':  vals[var] = target_val + 1
                elif final_op == '<':  vals[var] = target_val - 1
                elif final_op == '>=': vals[var] = target_val
                elif final_op == '<=': vals[var] = target_val
                elif final_op == '!=': vals[var] = target_val + 1
            
            if is_var:
                vals[target] = target_val
            return [vals]

        # Boolean / Pointer Implicit
        clean = condition.strip()
        while clean.startswith('(') and clean.endswith(')'):
            clean = clean[1:-1].strip()
        
        m_bool = re.search(r'^(!?)([\w\->\.]+)$', clean)
        if m_bool:
            is_not = (m_bool.group(1) == '!')
            var = m_bool.group(2)
            
            is_ptr = False
            if context_body:
                pat = r'(?:\*\s*{0}\b)|(?:\b{0}\s*->)|(?:\b{0}\s*\[)'.format(re.escape(var))
                if re.search(pat, context_body):
                    is_ptr = True
                
            need_true = (desired_state != is_not)
            if is_ptr:
                val = "<<malloc 1>>" if need_true else "<<null>>"
                desc = "!= NULL" if need_true else "== NULL"
            else:
                val = 1 if need_true else 0
                desc = "TRUE" if val else "FALSE"
                if is_not: desc = f"FALSE (val={val})"
                
            prefix = "!" if is_not else ""
            vals['_conditions'].append(f"{prefix}{var} is {desc}")
            vals[var] = val
            return [vals]
            
        return [vals]

    @staticmethod
    def solve_full_coverage(cond: str, defines: Dict[str, Any], desired: bool = True, body: Optional[str] = None) -> List[Dict[str, Any]]:
        cond = cond.strip()
        while cond.startswith('(') and cond.endswith(')'):
            d = 0
            valid = True
            for i in range(len(cond)-1):
                if cond[i] == '(':
                    d += 1
                elif cond[i] == ')':
                    d -= 1
                if d == 0:
                    valid = False
                    break
            if valid:
                cond = cond[1:-1].strip()
            else:
                break
            
        # OR
        idx = LogicSolver.find_split_index(cond, '||')
        if idx != -1:
            L, R = cond[:idx], cond[idx+2:]
            if desired:
                res = []
                # Case 1: Left True + Right False (Independence of L)
                tL = LogicSolver.solve_full_coverage(L, defines, True, body)
                fR = LogicSolver.solve_full_coverage(R, defines, False, body)
                for l in tL:
                    for r in fR:
                        c = l.copy(); c.update(r)
                        # [FIX] Prioritize True (L) to ensure the condition holds
                        c = r.copy(); c.update(l)
                        c['_conditions'] = l.get('_conditions',[]) + r.get('_conditions',[])
                        res.append(c)

                # Case 2: Left False + Right True (Independence of R)
                fL = LogicSolver.solve_full_coverage(L, defines, False, body)
                tR = LogicSolver.solve_full_coverage(R, defines, True, body)
                for l in fL:
                    for r in tR:
                        c = l.copy(); c.update(r)
                        c['_conditions'] = l.get('_conditions',[]) + r.get('_conditions',[])
                        res.append(c)
                
                # Case 3: Left True + Right True (Both True)
                tL2 = LogicSolver.solve_full_coverage(L, defines, True, body)
                tR2 = LogicSolver.solve_full_coverage(R, defines, True, body)
                for l in tL2:
                    for r in tR2:
                        # [FIX] Case (T || F): Prioritize True (L) as it drives the result
                        c = r.copy(); c.update(l)
                        c['_conditions'] = l.get('_conditions',[]) + r.get('_conditions',[])
                        res.append(c)

                return res
            else:
                fL = LogicSolver.solve_full_coverage(L, defines, False, body)
                fR = LogicSolver.solve_full_coverage(R, defines, False, body)
                res = []
                for l in fL:
                    for r in fR:
                        c = l.copy(); c.update(r)
                        c['_conditions'] = l.get('_conditions',[]) + r.get('_conditions',[])
                        res.append(c)
                return res
                
        # AND
        idx = LogicSolver.find_split_index(cond, '&&')
        if idx != -1:
            L, R = cond[:idx], cond[idx+2:]
            if desired:
                tL = LogicSolver.solve_full_coverage(L, defines, True, body)
                tR = LogicSolver.solve_full_coverage(R, defines, True, body)
                res = []
                for l in tL:
                    for r in tR:
                        c = l.copy(); c.update(r)
                        c['_conditions'] = l.get('_conditions',[]) + r.get('_conditions',[])
                        res.append(c)
                return res
            else:
                res = []
                tL = LogicSolver.solve_full_coverage(L, defines, True, body)
                fR = LogicSolver.solve_full_coverage(R, defines, False, body)
                for l in tL:
                    for r in fR:
                        # Prioritize False (R)
                        c = l.copy(); c.update(r)
                        c["_desc"] = " (Neg:Right)"
                        c['_conditions'] = l.get('_conditions',[]) + r.get('_conditions',[])
                        res.append(c)
                
                fL = LogicSolver.solve_full_coverage(L, defines, False, body)
                tR = LogicSolver.solve_full_coverage(R, defines, True, body)
                for l in fL:
                    for r in tR:
                        # Prioritize False (L) - Fix overwrite issue
                        c = r.copy(); c.update(l)
                        c["_desc"] = " (Neg:Left)"
                        c['_conditions'] = l.get('_conditions',[]) + r.get('_conditions',[])
                        res.append(c)

                # Case 3: Left False + Right False (Both False)
                fL2 = LogicSolver.solve_full_coverage(L, defines, False, body)
                fR2 = LogicSolver.solve_full_coverage(R, defines, False, body)
                for l in fL2:
                    for r in fR2:
                        # [FIX] Case (F && T): Prioritize False (L) as it drives the result
                        c = r.copy(); c.update(l)
                        c["_desc"] = " (Neg:Both)"
                        c['_conditions'] = l.get('_conditions',[]) + r.get('_conditions',[])
                        res.append(c)

                return res

        return LogicSolver.parse_atomic(cond, defines, desired, body)

# ==================================================================================
# 4. TEST GENERATOR (MAIN CLASS)
# ==================================================================================

class VectorCastGenerator:
    def __init__(self, c_file: str, header_folder: str, unit_name: str, env_name: str):
        self.c_file = c_file
        self.header_folder = header_folder
        self.unit_name = unit_name
        self.env_name = env_name
        self.defines = {}
        self.tst_content = []

    def run(self, output_file: str = "Result_Final.tst"):
        self.defines = load_defines(self.header_folder)
        
        if not os.path.exists(self.c_file):
            logger.error(f"C file not found: {self.c_file}")
            return

        with open(self.c_file, 'r', encoding='utf-8') as f:
            content = remove_comments(f.read())
        
        funcs = CodeParser.extract_functions(content)
        self._write_header()
        
        for func in funcs:
            if func["func_name"] == "main":
                continue
            self._process_function(func)
            
        with open(output_file, "w", encoding='utf-8') as f:
            f.write("\n".join(self.tst_content))
        logger.info(f"🚀 DONE! Generated: {output_file}")

    def _write_header(self):
        self.tst_content.extend([
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
            "--", ""
        ])

    def _process_function(self, func):
        fname = func["func_name"]
        body = func["body"]
        self.tst_content.append(f"-- Subprogram: {fname}")
        
        # 1. Extract Metadata
        stubs = CodeParser.extract_assignments(body)
        guards = CodeParser.extract_guard_values(body, self.defines)
        deps = CodeParser.extract_state_dependencies(body, self.defines)
        path_constr_false, path_constr_true = CodeParser.extract_path_constraints(body)
        loop_ranges = CodeParser.extract_loop_ranges(body)
        
        scenarios = []
        
        # 2. Process IFs
        masked = mask_switch_blocks(body)
        if_structs = CodeParser.extract_if_with_body(masked)
        
        for ifs in if_structs:
            cond = ifs['condition'].strip()
            blk = ifs['body']
            if_start_idx = ifs['start_index']
            if_end_idx = ifs['end_index']
            
            # Base Scenarios
            pos = LogicSolver.solve_full_coverage(cond, self.defines, True, body)
            neg = LogicSolver.solve_full_coverage(cond, self.defines, False, body)
            
            # [NEW] Tag scenarios with their branch logic for Notes
            for s in pos:
                s['_branch_node'] = f"[{cond}]"
            for s in neg:
                s['_branch_node'] = f"[!({cond})]"

            # Nested Ternary Expansion
            ternaries = CodeParser.extract_ternary_operators(blk)
            if ternaries:
                expanded = []
                for base in pos:
                    for t_cond in ternaries:
                        logger.debug(f"[NESTED TERNARY] '{t_cond}' inside IF('{cond}')")
                        t_pos = LogicSolver.solve_full_coverage(t_cond, self.defines, True, body)
                        t_neg = LogicSolver.solve_full_coverage(t_cond, self.defines, False, body)
                        for t in t_pos + t_neg:
                            new_s = base.copy()
                            new_s.update({k:v for k,v in t.items() if k not in ['_conditions','_desc', '_branch_node']})
                            new_s['_conditions'] = base.get('_conditions',[]) + t.get('_conditions',[])
                            expanded.append(new_s)
                if expanded: pos = expanded
            
            curr = pos + neg
            
            # Apply Constraints
            req_neg = path_constr_false.get(cond, [])
            req_pos = path_constr_true.get(cond, [])
            
            # Find loops containing this IF based on index
            req_loop = []
            for loop in loop_ranges:
                if loop['start'] < if_start_idx < loop['end']:
                    req_loop.append(loop['info'])
                    logger.debug(f"[APPLY LOOP] IF('{cond}') inside LOOP('{loop['info'].get('condition')}')")

            # [NEW] Find PARENT IFs (Nesting)
            # Find any IF block that completely encloses the current IF block
            parent_ifs = []
            for p_if in if_structs:
                if p_if == ifs: continue # Skip self
                # Check strict containment: Parent Start < Child Start AND Parent End > Child End
                if p_if['start_index'] < if_start_idx and p_if['end_index'] > if_end_idx:
                    parent_ifs.append(p_if['condition'])

            for s in curr:
                if "_desc" in s:
                    del s["_desc"]
                
                # [NEW] Build Branch Note Path based on NESTING ONLY
                # Format: [ParentLoop][ParentIF][CurrentIF]
                branch_stack = []
                
                # 1. Loops (Hierarchy)
                for l_info in req_loop:
                    c = l_info.get('condition', '')
                    branch_stack.append(f"[{c}]")
                
                # 2. Parent IFs (Hierarchy - Nested only)
                # Note: We sort them by start_index to ensure order from outermost to innermost would be correct if we tracked it, 
                # but currently just appending found parents is good enough for simple nesting.
                # To be precise, we could sort parent_ifs list if needed, but dict order usually preserves extraction order.
                for p_cond in parent_ifs:
                    branch_stack.append(f"[{p_cond}]")
                
                # 3. Current Condition
                if '_branch_node' in s:
                    branch_stack.append(s['_branch_node'])
                
                s['_branch_path_str'] = "".join(branch_stack)

                # Path Constraints application (Values) - Keep for Data Solving (Sequential)
                for p in req_neg:
                    inps = LogicSolver.solve_full_coverage(p, self.defines, False, body)
                    if inps:
                        s.update({k:v for k,v in inps[0].items() if k!='_conditions' and k not in s})
                
                for p in req_pos:
                    inps = LogicSolver.solve_full_coverage(p, self.defines, True, body)
                    if inps:
                        s.update({k:v for k,v in inps[0].items() if k!='_conditions' and k not in s})

                # Loop Constraints
                for l_info in req_loop:
                    iterator = l_info.get('iterator')
                    bound_var = l_info.get('bound_var')
                    
                    if iterator and bound_var and iterator in s:
                        required_val = s[iterator]
                        s[bound_var] = required_val + 1
                    else:
                        entry_cond = l_info.get('condition', '1')
                        inps = LogicSolver.solve_full_coverage(entry_cond, self.defines, True, body)
                        if inps:
                            s.update({k:v for k,v in inps[0].items() if k!='_conditions' and k not in s})
                        if inps: 
                            for k, v in inps[0].items():
                                if k != '_conditions':
                                    s[k] = v

                scenarios.append(s)
                
        # 3. Process Switch & Global Ternary
        scenarios.extend(CodeParser.extract_switch_cases(body, self.defines))
        for t in CodeParser.extract_ternary_operators(body):
            scenarios.extend(LogicSolver.solve_full_coverage(t, self.defines, True, body))
            scenarios.extend(LogicSolver.solve_full_coverage(t, self.defines, False, body))

        if not scenarios:
            scenarios.append({})
        
        # 4. Generate Test Cases
        params_info = self._parse_params(func["params"])
        
        for i, s in enumerate(scenarios):
            # Resolve Dependencies
            res = s.copy()
            if '_conditions' not in res:
                res['_conditions'] = []
            for k,v in s.items():
                if k in deps and v in deps[k]:
                    needed = deps[k][v]
                    inps = LogicSolver.solve_full_coverage(needed, self.defines, True, body)
                    if inps:
                        logger.debug(f"[AUTO-RESOLVE] {k}={v} -> {needed}")
                        res.update({dk:dv for dk,dv in inps[0].items() if dk!='_conditions'})

            self._write_test_case(fname, i+1, res, params_info, stubs, guards, body)

    def _parse_params(self, params_str):
        info = []
        raw = params_str.split(',') if params_str and "void" not in params_str else []
        for p in raw:
            p = p.strip()
            is_ptr = '*' in p
            parts = p.split()
            if not parts: continue
            nm = parts[-1].replace('*','').replace('[','').replace(']','').replace(';','')
            type_str = " ".join(parts[:-1])
            info.append({'name': nm, 'is_ptr': is_ptr, 'type': type_str})
        return info

    def _write_test_case(self, fname, case_num, scenario, params, stubs, guards, body):
        # [FIX] Merge Safe Defaults into scenario before processing anything
        for k, v in guards.items():
            if k not in scenario:
                scenario[k] = v
        desc = ""
        raw_c = scenario.get('_conditions', [])
        if raw_c:
            desc = f"[Expression] {', '.join(raw_c)}"
        
        cid = f"{case_num:03d}"
        self.tst_content.append("")
        self.tst_content.append(f"--Test Case:{fname}.{cid}")
        if desc: self.tst_content.append(f"-- [Test Case Description]{desc}")
        self.tst_content.append(f"TEST.UNIT:{self.unit_name}")
        self.tst_content.append(f"TEST.SUBPROGRAM:{fname}")
        self.tst_content.append("TEST.NEW")
        self.tst_content.append(f"TEST.NAME:{fname}.{cid}")

        # [NEW] ADD NOTES SECTION ----------------------------------
        branch_str = scenario.get('_branch_path_str', '')
        if branch_str:
            self.tst_content.append("TEST.NOTES:")
            self.tst_content.append(f"- Branch: {branch_str}")
            self.tst_content.append(f"- Purpose:[]")
            self.tst_content.append(f"- Test method:[]")
            self.tst_content.append(f"- Test case design techniques:[]")
            self.tst_content.append("TEST.END_NOTES:")
        # ----------------------------------------------------------
        
        cur_stub = set()
        
        # Params
        for p in params:
            nm = p['name']
            val = str(scenario.get(nm, ""))
            if p['is_ptr']:
                # [FIX] Calculate malloc size based on array access in scenario
                malloc_size = 1
                for k in scenario.keys():
                    # Match pattern: param_name[index]...
                    pattern = r'^' + re.escape(nm) + r'\[([^\]]+)\]'
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

                if "<<" in val: final_v = val
                elif val == "0": final_v = "<<null>>"
                else: final_v = f"<<malloc {malloc_size}>>"
                self.tst_content.append(f"TEST.VALUE:{self.unit_name}.{fname}.{nm}:{final_v}")
            else:
                # [FIX] Avoid assigning '0' to Struct/Union passed by value
                has_members = any(k.startswith(nm + ".") or k.startswith(nm + "->") for k in scenario.keys())
                type_str = p.get('type', '')
                is_complex = 'struct' in type_str or 'union' in type_str or type_str.endswith('_u')

                if not val and (has_members or is_complex):
                    continue

                final_v = val if val else "0"
                self.tst_content.append(f"TEST.VALUE:{self.unit_name}.{fname}.{nm}:{final_v}")

        # Vars & Stubs
        for k, v in scenario.items():
            if k in ['_conditions', '_desc', '_branch_node', '_branch_path_str']:
                continue
            is_param_exact = any(x['name']==k for x in params)
            if k in ["NULL", "return"]:
                continue
            
            vk = k
            # [FIX] Smart Array Indexing: Replace [j] with [val] if j is in scenario
            if '[' in vk and ']' in vk:
                def replace_idx(m):
                    i = m.group(1).strip()
                    return f"[{scenario[i]}]" if i in scenario else (f"[{i}]" if i.isdigit() else "[0]")
                vk = re.sub(r'\[(.*?)\]', replace_idx, vk)

            if k in stubs:
                for sf in stubs[k]:
                    if sf not in cur_stub:
                        self.tst_content.append(f"TEST.VALUE:{STUB_UNIT_NAME}.{sf}.return:{v}")
                        cur_stub.add(sf)
            elif not is_param_exact:
                # [IMPROVED] Better local variable regex detection (Supports struct, union, _t)
                type_pattern = r'(?:int|float|double|bool|char|void|short|long|[\w]+_t|struct\s+[\w]+|union\s+[\w]+|enum\s+[\w]+)'
                is_loc = re.search(r'\b(?:const\s+)?(?:unsigned\s+)?' + type_pattern + r'\s+(?:[\w\*]+\s+)?\b' + re.escape(k) + r'\b', body)
                if not is_loc:
                    # [FIX] Check if variable is a parameter to append function name
                    root_name = re.split(r'->|\.|\[', vk)[0].replace('*', '').strip()
                    is_param = any(x['name'] == root_name for x in params)
                    
                    if "->" in vk or "." in vk: vk = vk.replace("->", "[0].")
                    
                    if is_param or "->" in vk or "." in vk:
                        self.tst_content.append(f"TEST.VALUE:{self.unit_name}.{fname}.{vk}:{v}")
                    else:
                        self.tst_content.append(f"TEST.VALUE:{self.unit_name}.{vk}:{v}")

        # Safe Defaults
        for k,v in guards.items():
            if k not in scenario:
                if k in stubs:
                    for sf in stubs[k]:
                        if sf not in cur_stub:
                            self.tst_content.append(f"TEST.VALUE:{STUB_UNIT_NAME}.{sf}.return:{v}")
                            cur_stub.add(sf)
                else:
                    # [IMPROVED] Local variable regex detection for Safe Defaults
                    type_pattern = r'(?:int|float|double|bool|char|void|short|long|[\w]+_t|struct\s+[\w]+|union\s+[\w]+|enum\s+[\w]+)'
                    is_loc = re.search(r'\b(?:const\s+)?(?:unsigned\s+)?' + type_pattern + r'\s+(?:[\w\*]+\s+)?\b' + re.escape(k) + r'\b', body)
                    if not is_loc:
                        self.tst_content.append(f"TEST.VALUE:{self.unit_name}.{k}:{v}")

        self.tst_content.append("TEST.END")

# ==================================================================================
# 5. EXECUTION
# ==================================================================================
if __name__ == "__main__":
    # Check if arguments are provided. If yes -> CLI mode. If no -> UI mode.
    if len(sys.argv) > 1:
        # --- CLI MODE ---
        parser = argparse.ArgumentParser(description="VectorCAST Test Script Generator (Professional Edition)")
        
        parser.add_argument("c_file", help="Path to the C source file")
        parser.add_argument("header_folder", help="Path to the folder containing header files")
        parser.add_argument("--unit", required=True, help="Unit name (UUT)")
        parser.add_argument("--env", default="TEST_SV", help="Environment name")
        parser.add_argument("--output", default="Result_Final.tst", help="Output TST file name")
        parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")

        args = parser.parse_args()

        if args.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        if not os.path.exists(args.c_file):
            logger.error(f"Source file not found: {args.c_file}")
            exit(1)
        
        generator = VectorCastGenerator(args.c_file, args.header_folder, args.unit, args.env)
        generator.run(args.output)
    else:
        # --- UI MODE ---
        try:
            from Generate_Script_VectorCAST_UI import launch_ui
            launch_ui()
        except ImportError:
            print("Error: Could not import UI module 'Generate_Script_VectorCAST_UI.py'. Make sure it exists.")