"""Tests for the answer extractor.

The eval harness must extract the final numeric answer from a model
completion regardless of which surface format the model used. The most
common formats we see in practice are:

  1. GSM8K gold marker:        "#### 18"
  2. LaTeX boxed:              "\\boxed{18}"
  3. MetaMathQA conversational: "The answer is: 18"
  4. Plain trailing number:    "...so the result is 18."

These tests pin each of those formats plus a few edge cases (negatives,
decimals, commas, currency, markdown bold, multiple numbers).
"""

from src.answer_parsing import answers_match, extract_answer


def test_gsm8k_gold_marker():
    text = "Janet has 16 eggs. 16 - 3 - 4 = 9. 9 * 2 = 18.\n#### 18"
    assert extract_answer(text) == "18"


def test_boxed_latex():
    text = "We compute step by step and arrive at $\\boxed{42}$."
    assert extract_answer(text) == "42"


def test_metamath_answer_is():
    text = "Step 1: ... Step 2: ...\nThe answer is: 18"
    assert extract_answer(text) == "18"


def test_metamath_answer_is_no_colon():
    text = "After simplification, the answer is 7."
    assert extract_answer(text) == "7"


def test_negative_number():
    text = "The temperature dropped.\n#### -5"
    assert extract_answer(text) == "-5"


def test_decimal_number():
    text = "Pi to two places.\n#### 3.14"
    assert extract_answer(text) == "3.14"


def test_strips_thousands_separator():
    text = "Revenue grew to $1,200.\nThe answer is: 1,200"
    assert extract_answer(text) == "1200"


def test_strips_currency_symbol():
    text = "She earned $50.\nThe answer is: $50"
    assert extract_answer(text) == "50"


def test_markdown_bold_in_answer_is_form():
    text = "All things considered, the answer is **23**."
    assert extract_answer(text) == "23"


def test_marker_wins_over_intermediate_numbers():
    text = "Adding 4 and 7 gives 11. Times 2 is 22.\n#### 22"
    assert extract_answer(text) == "22"


def test_last_marker_wins_when_multiple():
    text = "First pass: #### 9. Wait, recompute.\n#### 18"
    assert extract_answer(text) == "18"


def test_fallback_to_last_number_when_no_marker():
    text = "I think there are 5 birds, then 3 fly away, leaving 2."
    assert extract_answer(text) == "2"


def test_returns_none_when_no_number():
    assert extract_answer("I have no idea.") is None


def test_returns_none_on_empty_string():
    assert extract_answer("") is None


def test_gsm8k_gold_field_still_parses():
    # The eval harness applies extract_answer to dataset["answer"] too.
    gold = (
        "Natalia sold 48/2 = 24 clips in May.\n"
        "Natalia sold 48+24 = 72 clips altogether in April and May.\n"
        "#### 72"
    )
    assert extract_answer(gold) == "72"


def test_answers_match_handles_extracted_pair():
    assert answers_match("18", "18") is True
    assert answers_match("18.0", "18") is True
    assert answers_match("18", "19") is False
    assert answers_match(None, "18") is False


def test_priority_marker_beats_answer_is():
    # Marker has higher priority than the conversational pattern.
    text = "The answer is: 9.\nFinal: #### 18"
    assert extract_answer(text) == "18"


def test_priority_boxed_beats_fallback():
    text = "Working: 4 + 5 = 9. So $\\boxed{9}$ which is 9."
    assert extract_answer(text) == "9"


def test_template_echo_with_answer_is_recovers():
    # Real completion pattern observed from run1_r8 (see
    # docs/lessons/01-prompt-template-echo.md). The model echoes the prompt's
    # `#### N` marker literally (letter N, not a digit) and emits the real
    # answer via MetaMathQA's "The answer is:" pattern. Extractor must pick
    # 540, not fall through to None or scrape an intermediate number.
    text = (
        "Total meters = 540\n"
        "Therefore, James runs a total of 540 meters in a week.\n"
        "\n"
        "#### N\n"
        "\n"
        "The answer is: 540"
    )
    assert extract_answer(text) == "540"
