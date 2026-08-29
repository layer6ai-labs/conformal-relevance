"""Prompt templates for relevancy-based sentence scoring.

Key design principles:
- ICL examples and inference use identical format for better alignment
- No separate "document context" section (sentences ARE the document)
- 0-indexed sentence numbering throughout
- Contrastive framing: "selected" vs "NOT selected" (neutral labels)
- No-intent datasets omit the "Intent:" line entirely
"""

# --- Main scoring template (with ICL examples, with intent) ---

RELEVANCY_DICT_SCORING_WITH_ICL_PROMPT = """
In the examples below, some sentences were selected and others were not.
Identify what distinguishes the selected sentences from the non-selected ones.
Then apply that SAME distinction to score each sentence in the evaluation section.

Each score should be a two decimal float between 0 and 1.
A sentence that clearly matches the demonstrated distinction should score close to 1 (> 0.8).
A sentence that does not match should score close to 0 (< 0.2).
Use intermediate scores (0.3, 0.5, 0.7) for partial matches.

IMPORTANT: Output must be valid JSON format with sentence indices as keys.

{icl_examples}
Now evaluate the following:

{task_line}Intent: {intent}
Sentences to evaluate:
{sentences}

Scores (JSON format):"""


# --- Main scoring template (with ICL examples, NO intent) ---

RELEVANCY_DICT_SCORING_WITH_ICL_PROMPT_NO_INTENT = """
In the examples below, some sentences were selected and others were not.
Identify what distinguishes the selected sentences from the non-selected ones.
Then apply that SAME distinction to score each sentence in the evaluation section.

Each score should be a two decimal float between 0 and 1.
A sentence that clearly matches the demonstrated distinction should score close to 1 (> 0.8).
A sentence that does not match should score close to 0 (< 0.2).
Use intermediate scores (0.3, 0.5, 0.7) for partial matches.

IMPORTANT: Output must be valid JSON format with sentence indices as keys.

{icl_examples}
Now evaluate the following:

{task_line}Sentences to evaluate:
{sentences}

Scores (JSON format):"""


# --- Missing scores retry template (with intent) ---

RELEVANCY_MISSING_SCORES_PROMPT = """
You previously scored sentences but missed some.
In the examples below, some sentences were selected and others were not.
Identify what distinguishes the selected sentences from the non-selected ones.
Then apply that SAME distinction to score each MISSED sentence below.

Each score should be a two decimal float between 0 and 1.
A sentence that clearly matches the demonstrated distinction should score close to 1 (> 0.8).
A sentence that does not match should score close to 0 (< 0.2).
Use intermediate scores (0.3, 0.5, 0.7) for partial matches.

IMPORTANT: Output must be valid JSON format with sentence indices as keys.

{icl_examples}
Now evaluate the following:

{task_line}Intent: {intent}
Missing sentences:
{missing_sentences}

Scores (JSON format with ONLY the missing sentence numbers as keys):"""


# --- Missing scores retry template (NO intent) ---

RELEVANCY_MISSING_SCORES_PROMPT_NO_INTENT = """
You previously scored sentences but missed some.
In the examples below, some sentences were selected and others were not.
Identify what distinguishes the selected sentences from the non-selected ones.
Then apply that SAME distinction to score each MISSED sentence below.

Each score should be a two decimal float between 0 and 1.
A sentence that clearly matches the demonstrated distinction should score close to 1 (> 0.8).
A sentence that does not match should score close to 0 (< 0.2).
Use intermediate scores (0.3, 0.5, 0.7) for partial matches.

IMPORTANT: Output must be valid JSON format with sentence indices as keys.

{icl_examples}
Now evaluate the following:

{task_line}Missing sentences:
{missing_sentences}

Scores (JSON format with ONLY the missing sentence numbers as keys):"""


# --- Dataset intent configuration ---

# Whether each dataset has meaningful intent fields for ICL prompt formatting.
# False → omit "Intent: {intent}" line from both ICL examples and inference section.
# (has_intent=True datasets still embed intent in prompt; False datasets omit it entirely)
ICL_HAS_INTENT: dict[str, bool] = {
    "ectsum": False,
    "physionet": False,
    "puma": True,
    "subsume": True,
    "hotpotqa": True,
    "evidence_inference": True,
    "contractnli": True,
}
