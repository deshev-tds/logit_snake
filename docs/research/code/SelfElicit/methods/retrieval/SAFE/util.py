import re


def extract_first_code_block(input_string: str, ignore_language: bool = False) -> str:
    """Extracts the contents of a string between the first code block (```)."""
    if ignore_language:
        pattern = re.compile(r'`(?:\w+\n)?(.*?)`', re.DOTALL)
    else:
        pattern = re.compile(r'`(.*?)`', re.DOTALL)

    match = pattern.search(input_string)
    if not match:
        return ''
    else:
        match = match.group(1).strip('\n')
        match = match.replace('`', '')
        return match


def extract_first_square_brackets(input_string: str) -> str:
    """Extracts the contents of the FIRST string between square brackets."""
    raw_result = re.findall(r'\[.*?\]', input_string, flags=re.DOTALL)

    if raw_result:
        return raw_result[0][1:-1]
    else:
        return ''
