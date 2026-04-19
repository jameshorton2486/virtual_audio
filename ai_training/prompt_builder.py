from __future__ import annotations


def build_prompt(text: str, examples: list[dict[str, str]] | None = None) -> str:
    examples = examples or []
    example_text = "\n\n".join(
        f"Original: {item['original']}\nCorrected: {item['corrected']}"
        for item in examples[:5]
        if item.get("original") and item.get("corrected")
    )

    return f"""
You are correcting a legal deposition transcript.

STRICT RULES:
- DO NOT change meaning
- DO NOT remove filler words
- DO NOT change speaker labels
- DO NOT change Q/A structure
- DO NOT summarize
- DO NOT rewrite sentences

ONLY:
- fix punctuation
- fix capitalization
- correct obvious grammar errors

EXAMPLES:
{example_text}

TEXT:
{text}
""".strip()
