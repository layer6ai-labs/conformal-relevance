"""Dataset registry for ICL example configuration.

NOTE: multi_intent here means "is this a multi-intent dataset for STRATIFIED ICL sampling?"
This is different from the prompt registry's has_intent which means "does the prompt use {intent}?"
For example, HotpotQA has has_intent=True in prompts (question fills {intent}) but
multi_intent=False here (each question is unique, no stratified sampling).

dataset_task: optional uniform task description for all samples. Appears as "Task: {description}"
in every ICL example and inference prompt. Separate from intent (per-sample category in multi-intent
datasets). Provides high-level frame (e.g., "PHI detection", "Question answering") to orient the LLM.
"""

DATASET_REGISTRY: dict[str, dict[str, str | bool]] = {
    "subsume": {
        "pool_path": "data/SubSumE/subsume_pool.parquet",
        "cal_path": "data/SubSumE/subsume_cal.parquet",
        "test_path": "data/SubSumE/subsume_test.parquet",
        "multi_intent": True,  # 10 shared query intents
        "dataset_task": "subjective summarization",
    },
    "puma": {
        "pool_path": "data/PUMA/puma_pool.parquet",
        "cal_path": "data/PUMA/puma_cal.parquet",
        "test_path": "data/PUMA/puma_test.parquet",
        "multi_intent": True,  # 5 shared perspective intents
        "dataset_task": "healthcare answer summarization",
    },
    "ectsum": {
        "pool_path": "data/ECTSum/ectsum_pool.parquet",
        "cal_path": "data/ECTSum/ectsum_cal.parquet",
        "test_path": "data/ECTSum/ectsum_test.parquet",
        "multi_intent": False,  # No intent
        "dataset_task": "earning call transcript summarization",
    },
    "physionet": {
        "pool_path": "data/PhysionetGoldCorpus/physionet_pool.parquet",
        "cal_path": "data/PhysionetGoldCorpus/physionet_cal.parquet",
        "test_path": "data/PhysionetGoldCorpus/physionet_test.parquet",
        "multi_intent": False,  # No intent
        "dataset_task": "PHI detection: identify sentences containing names, dates, locations, identifiers, or ages",
    },
    "hotpotqa": {
        "pool_path": "data/HotpotQA/hotpotqa_pool.parquet",
        "cal_path": "data/HotpotQA/hotpotqa_cal.parquet",
        "test_path": "data/HotpotQA/hotpotqa_test.parquet",
        "multi_intent": False,  # Each question is unique - no stratification
        "dataset_task": "question answering",
    },
    "evidence_inference": {
        "pool_path": "data/EvidenceInference_v2/evidence_inference_pool.parquet",
        "cal_path": "data/EvidenceInference_v2/evidence_inference_cal.parquet",
        "test_path": "data/EvidenceInference_v2/evidence_inference_test.parquet",
        "multi_intent": True,  # 8 shared user perspectives (User0-User7)
        # [PREVIOUS VERSION — 2026-03-13] Generic label; gave LLM no task context.
        # "dataset_task": "evidence inference",
        # Replaced 2026-03-16: domain-specific description gave +17% MAP on 100-sample pilot.
        "dataset_task": "clinical trial evidence scoring for between-group outcome significance, including positive, negative, and null results",
    },
    "contractnli": {
        "pool_path": "data/ContractNLI/contractnli_pool.parquet",
        "cal_path": "data/ContractNLI/contractnli_cal.parquet",
        "test_path": "data/ContractNLI/contractnli_test.parquet",
        "multi_intent": True,  # 17 shared NDA hypotheses
        # [PREVIOUS VERSION — 2026-03-24] Polarity language + dangling "hypothesis" ref hurt MoE.
        # "dataset_task": "legal contract review: identify contract clauses that are evidence for or against the given hypothesis",
        # Replaced 2026-03-25: R2a from task hint ablation. Cal MAP 0.8265 vs 0.7748 (no hint).
        "dataset_task": "identify clauses relevant to the intent",
    },
}


def get_dataset_config(dataset_name: str) -> dict[str, str | bool]:
    """Get dataset configuration from registry."""
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[dataset_name]
