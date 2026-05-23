# Lesson 01 — Prompt template echo: when the model writes `#### N` literally

**Discovered:** 2026-05-23, during the first end-to-end eval of `run1_r8`.
**Status:** Workaround in place (multi-pattern extractor). Root cause unfixed
because changing it mid-experiment would invalidate the comparison with the
already-evaluated base model.

---

## What we observed

After the extractor fix shipped, the base model scored 0.812 on GSM8K, in line
with published numbers for `Qwen2.5-3B-Instruct`. Then we ran a 32-problem
smoke test on the `run1_r8` adapter and dumped five raw completions. Three of
them ended like this (truncated):

```
...
Therefore, it takes 3 bolts in total to make the robe.

#### N
#### N
```

The model is supposed to emit `#### 3`. It emits the **literal characters
`#### N`** — the letter N, not the digit. Two copies of it, even.

Other completions that day:

```
Therefore, Josh made a profit of $70,000.

#### N
#### N
```

```
Total meters = 540

Therefore, James runs a total of 540 meters in a week.

#### N

The answer is: 540
```

The model knew the answer in every case. It just refused to inject the answer
into the `####` marker, and instead pasted the template back verbatim.

## Why this happened

Look at the prompt we send (`src/config.py`, `PROMPT_PREFIX`):

```
Solve the following math problem step by step. End your answer with the
marker `#### N` where N is the final numeric answer.
```

To a human, "N is the final numeric answer" is obviously a metavariable.
To an instruction-tuned LLM trained on chat data, `#### N` inside backticks is
strong evidence of *literal output to produce*. The model has seen countless
examples of "end with the marker X" instructions where X is the literal string
to emit. The metavariable convention is borrowed from math notation, not from
NLP instruction-following conventions.

The training data we used (MetaMathQA) doesn't end its solutions with `#### N`
at all — it ends them with `The answer is: <number>`. So the model has:

1. **A weak, ambiguous prompt instruction** that's easy to misinterpret as
   "echo the literal string `#### N`".
2. **No reinforcing examples in the SFT data** showing the correct `#### 18`
   substitution behavior.
3. **A strong competing pattern from SFT** that says "end with `The answer is:
   <number>`."

Predictable result: the model emits BOTH — `The answer is: 540` (the SFT-taught
format) AND `#### N` (a clumsy attempt to obey the prompt instruction
literally).

The higher-rank adapter (`run2_r16`) imitates this MetaMathQA pattern more
aggressively, which is partly why it scored *worse* than `run1_r8` in our
original (broken-extractor) eval: more capacity → more imitation → more drift
from the `#### N` substitution → more wrong extractions.

## Why this is a teaching moment

This is a microcosm of three real fine-tuning lessons:

### 1. The prompt and the SFT data have to agree on the output format

If the prompt says "end with X" but every training example ends with Y, the
model is being given contradictory signals. Whichever signal is stronger wins,
and the stronger signal in SFT is almost always the training data itself, not
the system prompt or instructions.

**Rule of thumb:** before fine-tuning, decide on the exact output format you
want, then check that **both** the prompt template at inference time **and**
the response field of the SFT data agree on it. If they don't, you either need
to rewrite the prompt to match the data, or preprocess the data to match the
prompt (e.g. append `#### <answer>` to every MetaMathQA `response`).

This project did neither. The prompt asks for `#### N`, the SFT data ends with
`The answer is: N`, and we never bridged the gap.

### 2. Don't put metavariables in literal-formatting instructions

`#### N where N is the final numeric answer` is the offender. The fix is to
make the literal/placeholder distinction unambiguous, e.g.:

- `End with a line of the exact form: #### <number>` — angle brackets are a
  near-universal placeholder convention in technical docs.
- `End with: #### followed by the final numeric answer (a number, not the
  letter N).` — explicit.
- `End with #### then the answer. Example: #### 42` — show, don't describe.

We did not change the prompt mid-experiment, because that would invalidate the
already-completed base eval. We're shipping the broken prompt all the way
through the ablation and noting it here.

### 3. Robust postprocessing is cheap insurance

The single-regex extractor (`#### N` only) read base accuracy at 0.373.
With the multi-pattern extractor (`#### N` → `\boxed{N}` → "answer is N" →
last-number fallback), base accuracy is 0.812. **A 44-point gap, none of it
real model behavior — all of it postprocessing.**

In production this would be a critical reliability issue. In a small ablation
study like this one, it would have silently produced a meaningless leaderboard
where the "best" adapter is the one whose format drift happened to match the
extractor's blind spots least.

**Rule of thumb:** before trusting any "model X scored Y%" number, dump 5-10
raw completions, and check whether the extractor agrees with what a human
reader would say the model's answer was.

## What the extractor catches now

`src/answer_parsing.py` tries four patterns in order:

1. `#### N` (digit, not letter) — GSM8K gold format and what our prompt asks for
2. `\boxed{N}` — LaTeX, common in math
3. `answer is N` / `answer: N` / `= N` — MetaMathQA's actual format
4. Last number anywhere in the completion — fallback

For the `#### N` (literal letter N) case described above, pattern 1 fails
(`N` is not a digit), pattern 2 fails, and pattern 3 matches "the answer is
540". Disaster averted.

## What a real fix would look like

If we were starting over:

```python
# config.py
PROMPT_PREFIX = (
    "Solve the following math problem step by step. "
    "After your reasoning, write a single final line in this exact form:\n"
    "#### <final number>\n\n"
    "Example: if the answer is 42, the final line is `#### 42`.\n\n"
)
```

```python
# data.py — append the marker the prompt asks for
def _to_messages(example):
    answer = example["response"].rsplit("The answer is:", 1)[-1].strip().rstrip(".")
    response_with_marker = example["response"].rstrip() + f"\n#### {answer}"
    return {
        "messages": [
            {"role": "user", "content": example["query"]},
            {"role": "assistant", "content": response_with_marker},
        ]
    }
```

Both changes together — prompt is unambiguous, SFT data ends with the exact
form the prompt asks for. The model has nothing to be confused about.

Doing this would require retraining all 7 adapters and re-evaluating the base,
which is out of scope for this run. Tracked as a possible follow-up if this
project ever gets a v2.

## Pinning test

`tests/test_extract_answer.py::test_template_echo_with_answer_is_recovers`
encodes the exact pattern we observed in the dump — literal `#### N` plus
a trailing `The answer is: 540`. This test will fail if a future "cleanup"
of the extractor drops the conversational pattern, surfacing the regression
before it ships.

## TL;DR

- Instruction-tuned LLMs interpret `#### N` in backticks as a literal string
  to echo, not as `#### <substitute-N-here>`.
- The prompt and the SFT response format must agree on output style, or the
  model produces a confused hybrid.
- Always validate the answer extractor by dumping raw completions before
  trusting accuracy numbers. A 44-point swing from a regex fix is possible
  and was observed here.
