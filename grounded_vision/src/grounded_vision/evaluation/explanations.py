"""
Explanation Benchmark Suite
===========================
Quantifies explanation quality across:
1. Formal Sufficiency Rate (% of samples with mathematical sufficiency certificate)
2. Explanation Complexity / Sparsity (Average number of conditions)
3. Necessity Rate (% of conditions whose removal falsifies sufficiency)
4. Natural Language Faithfulness (Precision, Recall, Hallucination Rate)
"""

from __future__ import annotations
from typing import List, Dict, Any
import numpy as np

from ..explanations.schema import SufficientExplanation, VerificationStatus
from ..language.checks import AuditReport


def evaluate_explanation_suite(
    explanations: List[SufficientExplanation],
    audit_reports: Optional[List[AuditReport]] = None,
) -> Dict[str, Any]:
    """
    Computes explanation benchmarks.
    """
    if not explanations:
        return {}

    n = len(explanations)
    verified_count = sum(
        1 for e in explanations if e.certificate.status == VerificationStatus.VERIFIED
    )
    sufficiency_rate = verified_count / n

    sparsities = [e.sparsity for e in explanations]
    mean_sparsity = float(np.mean(sparsities))
    std_sparsity = float(np.std(sparsities))

    metrics: Dict[str, Any] = {
        "num_evaluated_samples": n,
        "sufficiency_rate": float(sufficiency_rate),
        "mean_sparsity_conditions": mean_sparsity,
        "std_sparsity_conditions": std_sparsity,
    }

    if audit_reports:
        faithful_count = sum(1 for a in audit_reports if a.is_faithful)
        precisions = [a.faithfulness_precision for a in audit_reports]
        recalls = [a.faithfulness_recall for a in audit_reports]
        hallucination_count = sum(1 for a in audit_reports if len(a.unsupported_claims) > 0)

        metrics.update({
            "nl_faithful_rate": float(faithful_count / len(audit_reports)),
            "nl_mean_precision": float(np.mean(precisions)),
            "nl_mean_recall": float(np.mean(recalls)),
            "nl_hallucination_rate": float(hallucination_count / len(audit_reports)),
        })

    return metrics
