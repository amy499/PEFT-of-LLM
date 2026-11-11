"""Shared prompt template used by BOTH finetune.py and evaluate.py.

The text is byte-identical to the original training template in finetune.py,
including its trailing spaces and the 16-space indentation of every line after
the header. It is spelled out with explicit escapes so that editors that strip
trailing whitespace cannot silently change it.

The only behavioural change from the original is that the instruction is
stripped (math_7k instructions end in "\\n ", test-set instructions do not).
"""

_INDENT = " " * 16

# Header lines, exactly as in the original finetune.py template.
_HEADER_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request. "  # one trailing space
)
_HEADER_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.  "  # two trailing spaces
)


def generate_prompt(instruction, input=None, output=""):
    """Build the prompt. Pass output="" (default) for the inference prompt."""
    instruction = instruction.strip()
    output = output or ""
    if input:
        return (
            _HEADER_WITH_INPUT + "\n"
            "\n"
            + _INDENT + "### Instruction:\n"
            + _INDENT + instruction + "\n"
            + _INDENT + "\n"
            + _INDENT + "### Input:\n"
            + _INDENT + input + "\n"
            + _INDENT + "\n"
            + _INDENT + "### Response:\n"
            + _INDENT + output
        )
    return (
        _HEADER_NO_INPUT + "\n"
        "\n"
        + _INDENT + "### Instruction:\n"
        + _INDENT + instruction + "\n"
        + _INDENT + "\n"
        + _INDENT + "### Response:\n"
        + _INDENT + output
    )
