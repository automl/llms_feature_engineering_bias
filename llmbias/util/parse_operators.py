import re


def parse_openfe_operator(s: str) -> str:
    # regex matching the operator before the brackets
    pattern = r'(\w*)\(([^)]*)\)'
    match = re.search(pattern, s)

    if match:
        word_before_bracket = match.group(1)
        inside_bracket = match.group(2)

        # Check if inside the bracket there's a +, -, *, or /
        operator_match = re.search(r'[+\-*/]', inside_bracket)
        if operator_match:
            # Return the symbol if found
            return operator_match.group(0)
        else:
            # Return the word before the bracket
            return word_before_bracket

    # this should generally not happen, but still here for safety
    return None
