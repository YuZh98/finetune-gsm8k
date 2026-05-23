"""Answer extraction and comparison for GSM8K-style numeric responses.

Kept dependency-free (stdlib only) so the extractor can be unit-tested
without the heavy training stack. `utils.py` reexports these names for
backwards compatibility with the eval harness.

Models trained on different SFT corpora emit final answers in different
surface forms. The extractor tries each known marker in priority order
and falls back to "last number anywhere in the text" only if no marker
matched. Priority order matters: a completion that contains both a
`####` marker and an `\\boxed{...}` should resolve to the `####` value,
because `####` is the format we explicitly prompt the model for.
"""

from __future__ import annotations

import re
from typing import Optional

# Number fragment shared by all patterns: optional sign, digits with
# optional thousands separators, optional decimal part.
_NUM = r"-?[\d,]+(?:\.\d+)?"

# Ordered by signal strength. First match wins, but within a pattern we
# take the LAST occurrence (a self-correcting completion may emit the
# right answer after a wrong one).
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"####\s*\$?({_NUM})"),
    re.compile(rf"\\boxed\{{\s*\$?({_NUM})\s*\}}"),
    re.compile(
        rf"(?:answer\s*(?:is|:)|=)\s*\$?\*{{0,2}}({_NUM})",
        re.IGNORECASE,
    ),
)

# Fallback only when no marker matched.
_NUMBER_RE = re.compile(_NUM)


def _clean(raw: str) -> str:
    return raw.replace(",", "").strip()


def extract_answer(text: Optional[str]) -> Optional[str]:
    """Pull the final numeric answer out of a completion.

    Returns the number as a string (preserving sign and decimal,
    thousands separators stripped) or None if nothing numeric was found.
    """
    if not text:
        return None
    for pattern in _PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return _clean(matches[-1])
    matches = _NUMBER_RE.findall(text)
    if matches:
        return _clean(matches[-1])
    return None


def answers_match(predicted: Optional[str], gold: Optional[str]) -> bool:
    """Compare two extracted numeric answers as floats."""
    if predicted is None or gold is None:
        return False
    try:
        return abs(float(predicted) - float(gold)) < 1e-6
    except ValueError:
        return predicted.strip() == gold.strip()
