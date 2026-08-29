"""ICL-free prompt templates for dataset-specific scoring.

Each dataset has a main scoring prompt and a retry prompt for missing scores.
The ICL_FREE_PROMPT_REGISTRY maps dataset names to their prompt configuration.
"""

# =============================================================================
# ECTSum - Importance Scoring (no intent)
# =============================================================================

IMPORTANCE_DICT_SCORING_NO_ICL_PROMPT = """
Evaluate the importance of each input sentence in the original text, based on how the information carried in the sentence is aligned with the overall message.
and provide a importance score for EACH input sentence.
Each output score should be a two decimal float number ranged between 0 and 1,
indicating how important the corresponding input sentence is in the context of the text document.
For example, if sentence 1's information is highly aligned with that of the input text,
and very likely to be included in the summary, then score 1 should be close to 1, say greater than 0.8;
if information carried in sentence 3 is trivial or only remotely related to the central message of the text,
and is not worthy of inclusion in the summary, then score 3 should be close to 0, say less than 0.2.

IMPORTANT: Output must be valid JSON format with sentence indices as keys.

Sentences to evaluate:
{sentences}

Importance scores (JSON format):"""


IMPORTANCE_MISSING_SCORES_PROMPT = """
You previously scored sentences for importance but missed some.
Evaluate the importance of each MISSED sentence listed below, based on how the information carried in the sentence is aligned with the overall message.
and provide a importance score for EACH listed sentence.
Each output score should be a two decimal float number ranged between 0 and 1,
indicating how important the corresponding input sentence is in the context of the text document.
For example, if a sentence's information is highly aligned with the overall message,
and very likely to be included in a summary, then its score should be close to 1 (greater than 0.8);
if information carried in a sentence is trivial or only remotely related to the central message,
and is not worthy of inclusion in a summary, then its score should be close to 0 (less than 0.2).

IMPORTANT: Output must be valid JSON format with sentence indices as keys.

Missing sentences:
{missing_sentences}

Importance scores (JSON format with ONLY the missing sentence numbers as keys):"""

# =============================================================================
# PUMA - Perspective-Based Scoring (has intent)
# =============================================================================

PERSPECTIVE_DICT_SCORING_NO_ICL_PROMPT = """
Evaluate each sentence based on how well it expresses or addresses the given perspective in this discussion.
Provide a score for EACH sentence.
Each score should be a two decimal float between 0 and 1.

For example, if a sentence directly addresses the {intent} perspective,
its score should be close to 1 (greater than 0.8);
if a sentence is unrelated to the {intent} perspective or addresses a different aspect,
its score should be close to 0 (less than 0.2).

IMPORTANT: Output must be valid JSON format with sentence indices as keys.

Perspective: {intent}
Sentences to evaluate:
{sentences}

Scores (JSON format):"""

PERSPECTIVE_MISSING_SCORES_NO_ICL_PROMPT = """
You previously scored sentences but missed some.
Evaluate each MISSED sentence based on how well it expresses or addresses the given perspective in this discussion.
Provide a score for EACH listed sentence.
Each score should be a two decimal float between 0 and 1.

For example, if a sentence directly addresses the {intent} perspective,
its score should be close to 1 (greater than 0.8);
if a sentence is unrelated to the {intent} perspective or addresses a different aspect,
its score should be close to 0 (less than 0.2).

IMPORTANT: Output must be valid JSON format with sentence indices as keys.

Perspective: {intent}
Missing sentences:
{missing_sentences}

Scores (JSON format with ONLY the missing sentence numbers as keys):"""

# =============================================================================
# SubSumE - Query-Based Scoring (has intent)
# =============================================================================

QUERY_DICT_SCORING_NO_ICL_PROMPT = """
Evaluate each sentence based on how well it answers or provides information relevant to the given query.
Provide a score for EACH sentence.
Each score should be a two decimal float between 0 and 1.

For example, if a sentence directly answers or supports the query,
its score should be close to 1 (greater than 0.8);
if a sentence is unrelated to the query or provides only irrelevant background,
its score should be close to 0 (less than 0.2).

IMPORTANT: Output must be valid JSON format with sentence indices as keys.

Query: {intent}
Sentences to evaluate:
{sentences}

Scores (JSON format):"""

QUERY_MISSING_SCORES_NO_ICL_PROMPT = """
You previously scored sentences but missed some.
Evaluate each MISSED sentence based on how well it answers or provides information relevant to the given query.
Provide a score for EACH listed sentence.
Each score should be a two decimal float between 0 and 1.

For example, if a sentence directly answers or supports the query,
its score should be close to 1 (greater than 0.8);
if a sentence is unrelated to the query or provides only irrelevant background,
its score should be close to 0 (less than 0.2).

IMPORTANT: Output must be valid JSON format with sentence indices as keys.

Query: {intent}
Missing sentences:
{missing_sentences}

Scores (JSON format with ONLY the missing sentence numbers as keys):"""

# =============================================================================
# PhysioNet - PHI Identification Scoring (no intent)
# =============================================================================

PHI_DICT_SCORING_NO_ICL_PROMPT = """
Evaluate each sentence for the presence of Protected Health Information (PHI).
PHI includes: patient names, doctor names, dates (birth, admission, discharge, specific times),
locations (cities, hospitals, addresses), medical record numbers, phone numbers, and ages.
Provide a PHI score for EACH sentence.
Each score should be a two decimal float between 0 and 1.

For example, if a sentence contains clear PHI such as a person's name, specific date, or location,
its score should be close to 1 (greater than 0.8);
if a sentence contains only general medical information with no personal identifiers,
its score should be close to 0 (less than 0.2).

IMPORTANT: Output must be valid JSON format with sentence indices as keys.

Sentences to evaluate:
{sentences}

PHI scores (JSON format):"""

PHI_MISSING_SCORES_NO_ICL_PROMPT = """
You previously scored sentences for PHI but missed some.
Evaluate each MISSED sentence for the presence of Protected Health Information (PHI).
PHI includes: patient names, doctor names, dates (birth, admission, discharge, specific times),
locations (cities, hospitals, addresses), medical record numbers, phone numbers, and ages.
Provide a PHI score for EACH listed sentence.
Each score should be a two decimal float between 0 and 1.

For example, if a sentence contains clear PHI such as a person's name, specific date, or location,
its score should be close to 1 (greater than 0.8);
if a sentence contains only general medical information with no personal identifiers,
its score should be close to 0 (less than 0.2).

IMPORTANT: Output must be valid JSON format with sentence indices as keys.

Missing sentences:
{missing_sentences}

PHI scores (JSON format with ONLY the missing sentence numbers as keys):"""

# =============================================================================
# HotpotQA - Question Answering Scoring (has intent)
# =============================================================================

QA_DICT_SCORING_NO_ICL_PROMPT = """
Evaluate each sentence based on how useful it is as a supporting fact for answering the given question.
Provide a score for EACH sentence.
Each score should be a two decimal float between 0 and 1.

For example, if a sentence provides information needed to answer the question,
its score should be close to 1 (greater than 0.8);
if a sentence provides unrelated information that doesn't help answer the question,
its score should be close to 0 (less than 0.2).

IMPORTANT: Output must be valid JSON format with sentence indices as keys.

Question: {intent}
Sentences to evaluate:
{sentences}

Scores (JSON format):"""

QA_MISSING_SCORES_NO_ICL_PROMPT = """
You previously scored sentences but missed some.
Evaluate each MISSED sentence based on how useful it is as a supporting fact for answering the given question.
Provide a score for EACH listed sentence.
Each score should be a two decimal float between 0 and 1.

For example, if a sentence provides information needed to answer the question,
its score should be close to 1 (greater than 0.8);
if a sentence provides unrelated information that doesn't help answer the question,
its score should be close to 0 (less than 0.2).

IMPORTANT: Output must be valid JSON format with sentence indices as keys.

Question: {intent}
Missing sentences:
{missing_sentences}

Scores (JSON format with ONLY the missing sentence numbers as keys):"""

# =============================================================================
# Evidence Inference - Evidence Scoring (has intent - user perspective)
# =============================================================================

# V8 prompt (2026-03-17). Cal MAP=0.223. V1-V7 history and ablation findings documented in
# PlanDocs/completed/evidence_inference_moe_underperformance_investigation.md § Section 5.
EVIDENCE_DICT_SCORING_NO_ICL_PROMPT = """
From {intent}'s perspective, evaluate each sentence for its value as statistical evidence of the treatment's effect on the outcome.
Evidence sentences typically contain: statistical results (p-values, confidence intervals),
outcome measurements, comparative data between treatment and control groups, hazard ratios, or effect sizes.
Provide an evidence score for EACH sentence.
Each score should be a two decimal float between 0 and 1.

For example, if a sentence reports statistical results or quantitative treatment outcomes,
its score should be close to 1 (greater than 0.8);
if a sentence contains only background information, methodology description, or non-quantitative text,
its score should be close to 0 (less than 0.2).

IMPORTANT: Output must be valid JSON with ALL sentence indices as keys. Every sentence must be scored and every index must appear.

Perspective: {intent}
Sentences to evaluate:
{sentences}

Evidence scores (JSON format):"""

EVIDENCE_MISSING_SCORES_NO_ICL_PROMPT = """
You previously scored sentences for evidence value but missed some.
From {intent}'s perspective, evaluate each MISSED sentence for its value as statistical evidence of the treatment's effect on the outcome.
Evidence sentences typically contain: statistical results (p-values, confidence intervals),
outcome measurements, comparative data between treatment and control groups, hazard ratios, or effect sizes.
Provide an evidence score for EACH listed sentence.
Each score should be a two decimal float between 0 and 1.

For example, if a sentence reports statistical results or quantitative treatment outcomes,
its score should be close to 1 (greater than 0.8);
if a sentence contains only background information, methodology description, or non-quantitative text,
its score should be close to 0 (less than 0.2).

IMPORTANT: Output must be valid JSON format. Every listed sentence index must appear as a key.

Perspective: {intent}
Missing sentences:
{missing_sentences}

Evidence scores (JSON format with ONLY the missing sentence numbers as keys):"""

# =============================================================================
# ContractNLI - Legal Contract Clause Scoring (has intent - NDA hypothesis)
# =============================================================================

CONTRACT_NLI_DICT_SCORING_NO_ICL_PROMPT = """
Evaluate each contract clause (sentence or list item) based on whether it provides evidence for or against the given hypothesis about this NDA.
A clause is evidence if it directly supports, contradicts, or qualifies the hypothesis.
Provide a relevance score for EACH clause.
Each score should be a two decimal float between 0 and 1.

For example, if a clause directly addresses the hypothesis (e.g., explicitly states an obligation or right that the hypothesis describes),
its score should be close to 1 (greater than 0.8);
if a clause is about unrelated obligations, definitions, or boilerplate unrelated to the hypothesis,
its score should be close to 0 (less than 0.2).

IMPORTANT: Output must be valid JSON format with sentence indices as keys.

Hypothesis: {intent}
Contract clauses to evaluate:
{sentences}

Relevance scores (JSON format):"""

CONTRACT_NLI_MISSING_SCORES_NO_ICL_PROMPT = """
You previously scored contract clauses but missed some.
Evaluate each MISSED clause based on whether it provides evidence for or against the given hypothesis.
Provide a relevance score for EACH listed clause.
Each score should be a two decimal float between 0 and 1.

For example, if a clause directly addresses the hypothesis,
its score should be close to 1 (greater than 0.8);
if a clause is about unrelated obligations or boilerplate,
its score should be close to 0 (less than 0.2).

IMPORTANT: Output must be valid JSON format with sentence indices as keys.

Hypothesis: {intent}
Missing clauses:
{missing_sentences}

Relevance scores (JSON format with ONLY the missing sentence numbers as keys):"""

# =============================================================================
# ICL-Free Prompt Registry
# =============================================================================

# Registry mapping dataset names to (main_prompt, retry_prompt, has_intent)
# NOTE: has_intent here means "prompt template uses {intent} placeholder"
# See Architecture Note in merge plan for the distinction with ICL strategy has_intent
ICL_FREE_PROMPT_REGISTRY: dict[str, tuple[str, str, bool]] = {
    "ectsum": (IMPORTANCE_DICT_SCORING_NO_ICL_PROMPT, IMPORTANCE_MISSING_SCORES_PROMPT, False),
    "puma": (PERSPECTIVE_DICT_SCORING_NO_ICL_PROMPT, PERSPECTIVE_MISSING_SCORES_NO_ICL_PROMPT, True),
    "subsume": (QUERY_DICT_SCORING_NO_ICL_PROMPT, QUERY_MISSING_SCORES_NO_ICL_PROMPT, True),
    "physionet": (PHI_DICT_SCORING_NO_ICL_PROMPT, PHI_MISSING_SCORES_NO_ICL_PROMPT, False),
    "hotpotqa": (QA_DICT_SCORING_NO_ICL_PROMPT, QA_MISSING_SCORES_NO_ICL_PROMPT, True),
    "evidence_inference": (EVIDENCE_DICT_SCORING_NO_ICL_PROMPT, EVIDENCE_MISSING_SCORES_NO_ICL_PROMPT, True),
    "contractnli": (CONTRACT_NLI_DICT_SCORING_NO_ICL_PROMPT, CONTRACT_NLI_MISSING_SCORES_NO_ICL_PROMPT, True),
}
