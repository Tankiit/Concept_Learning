"""
Explanation Faithfulness Auditor
================================
Audits generated natural language explanations against formal conditions to detect:
1. Omitted claims: Verified conditions absent from the verbalized text.
2. Unsupported claims / Hallucinations: Visual attributes claimed in the text
   that are NOT backed by certified conditions or grounded facts.
3. Polarity flips: a certified ABSENCE verbalized as presence (or vice versa).

Claim mode (default) audits by BINDING, not token overlap: for each certified
condition the auditor renders both polarity phrasings with the deterministic
template and checks which one the text contains — so "a white belly color"
FAILS against a certified ``belly:belly_color=white <= 0.5`` even though
"white" is an allowed token, and "a black belly" fails against
``belly:belly_color=white`` + ``beak:bill_color=black`` because the bound
(part, attribute, value, polarity) claim never appears. Foreign/free-form text (the LLM-comparison arm) gets a
heuristic claim extractor — known value words bound to their nearest part
mention with a negation-cue polarity — cross-checked against the certified
(part, value, polarity) triples; text with no recognizable template phrasing
and no extractable claims falls back to the lexical scan. The report's
metadata names which mode ran: claim-binding | claim-heuristic | lexical.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import re

from ..explanations.evidence import VisualEvidenceCard
from ..explanations.schema import SufficientExplanation, ExplanationCondition
from ..predicates.schema import FactDatabase
from .templates import ExplanationTemplateGenerator


@dataclass
class AuditReport:
    """Detailed audit report quantifying natural language explanation faithfulness."""
    is_faithful: bool
    omitted_conditions: List[str]
    unsupported_claims: List[str]
    faithfulness_precision: float
    faithfulness_recall: float
    audit_summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_faithful": self.is_faithful,
            "omitted_conditions": self.omitted_conditions,
            "unsupported_claims": self.unsupported_claims,
            "faithfulness_precision": float(self.faithfulness_precision),
            "faithfulness_recall": float(self.faithfulness_recall),
            "audit_summary": self.audit_summary,
            "metadata": self.metadata,
        }


class ExplanationAuditor:
    """
    Auditor that scans natural language explanations against grounded conditions.
    """
    def __init__(self):
        # Known vocabulary tokens for parts and attributes. Extended to the full
        # CUB-312 value vocabulary so unsupported-claim detection also fires on
        # values outside the canonical 28.
        self.known_colors = {
            "yellow", "black", "white", "red", "brown", "blue", "grey", "gray",
            "olive", "orange", "pink", "purple", "rufous", "buff", "iridescent",
            "green", "yellowish", "buffy",
        }
        self.known_shapes = {
            "curved", "cone", "dagger", "hooked", "needle", "forked",
            "spatulate", "specialized", "all-purpose", "bulbous",
            "rounded", "notched", "fan-shaped", "pointed", "squared",
            "hooked_seabird", "curved_(up_or_down)",
        }
        self.known_parts = {
            "beak", "bill", "crown", "forehead", "breast", "belly", "wing", "tail", "throat", "back",
            "nape", "eye", "leg", "bird",
        }
        self._template = ExplanationTemplateGenerator()

    def _normalize_text(self, text: str) -> str:
        return text.lower().replace("-", " ").replace("_", " ")

    def _normalize_value(self, value: str) -> str:
        """Applies text normalization to a feature value token so that e.g.
        'curved_(up_or_down)' matches its own verbalization."""
        return value.lower().replace("-", " ").replace("_", " ").replace("(", "").replace(")", "").strip()

    # ------------------------------------------------ free-text claim extraction
    _NEGATION_CUES = {
        "no", "not", "without", "lacks", "lacking", "neither", "never", "absent",
    }

    def _part_words(self) -> List[str]:
        """Part lookup words, including compound CUB names, normalized."""
        words = set(self.known_parts)
        words |= {"left wing", "right wing", "left eye", "right eye",
                  "left leg", "right leg"}
        return sorted(words)

    def _certified_claims(
        self, sufficient_explanation: SufficientExplanation
    ) -> Set[Tuple[str, str, str]]:
        """(part, value, polarity) triples certified by the conditions.
        has_part / relational conditions carry no attribute value and are
        skipped for claim matching."""
        claims: Set[Tuple[str, str, str]] = set()
        for cond in sufficient_explanation.conditions:
            f_name = cond.feature_name
            polarity = "absent" if cond.operator == "<=" else "present"
            if ":" in f_name and "=" in f_name:
                part, attr_val = f_name.split(":", 1)
                val = attr_val.split("=", 1)[1]
                claims.add((part.replace("_", " "), self._normalize_value(val), polarity))
            elif "=" in f_name:
                val = f_name.split("=", 1)[1]
                claims.add(("", self._normalize_value(val), polarity))
        return claims

    @staticmethod
    def _parts_match(claim_part: str, certified_part: str) -> bool:
        """beak/bill are aliases; a generic mention ('wing') covers the
        laterality variants ('left wing') and vice versa."""
        alias = {"bill": "beak"}
        cp = alias.get(claim_part, claim_part)
        tp = alias.get(certified_part, certified_part)
        return cp == tp or cp in tp or tp in cp

    def extract_text_claims(self, text: str) -> List[Dict[str, str]]:
        """Heuristic (part, value, polarity) extraction from FREE text, for
        auditing foreign explanations (the LLM/VLM comparison arm).

        A claim = a known attribute-value word, the nearest part word within
        a +/-6 token window, and a polarity from negation cues within 3
        tokens before the value. This is a heuristic — the deterministic
        round-trip check remains the audit for template-family text.
        """
        norm = (
            text.lower().replace("-", " ").replace("_", " ")
            .replace("(", " ").replace(")", " ").replace(",", " ")
            .replace(".", " ").replace(";", " ")
        )
        tokens = norm.split()
        parts = self._part_words()
        value_words = {self._normalize_value(v): v for vocab in
                       (self.known_colors, self.known_shapes) for v in vocab}

        claims: List[Dict[str, str]] = []
        i = 0
        while i < len(tokens):
            matched_len = 0
            for span in (3, 2, 1):
                gram = " ".join(tokens[i:i + span])
                if gram in value_words:
                    matched_len = span
                    break
            if not matched_len:
                i += 1
                continue
            gram = " ".join(tokens[i:i + matched_len])

            part = ""
            # nearest part mention wins: scan outward from the value so the
            # adjacent part word beats a distant 'bird'/'belly' from another
            # clause
            for d in range(1, 7):
                for j in (i - d, i + d):
                    if j < 0 or j >= len(tokens):
                        continue
                    for p in parts:
                        pw = p.split()
                        if tokens[j:j + len(pw)] == pw:
                            part = p
                            break
                    if part:
                        break
                if part:
                    break

            polarity = "present"
            for cue in self._NEGATION_CUES:
                cw = cue.split()
                for j in range(max(0, i - len(cw) - 2), i):
                    if tokens[j:j + len(cw)] == cw:
                        polarity = "absent"
                        break
                if polarity == "absent":
                    break

            claims.append({"part": part, "value": gram, "polarity": polarity})
            i += matched_len
        return claims

    def _heuristic_unsupported(
        self, text: str, sufficient_explanation: SufficientExplanation
    ) -> List[str]:
        """Crosses heuristically extracted free-text claims against the
        certified (part, value, polarity) triples."""
        certified = self._certified_claims(sufficient_explanation)
        unsupported: List[str] = []
        for claim in self.extract_text_claims(text):
            ok = any(
                self._parts_match(claim["part"], cp)
                and claim["value"] == cv
                and claim["polarity"] == pol
                for cp, cv, pol in certified
            )
            if not ok:
                unsupported.append(
                    f"claim:{claim['part'] or '?'}:{claim['value']}:{claim['polarity']}"
                )
        return unsupported

    def _condition_phrasings(self, cond: ExplanationCondition) -> Tuple[str, str]:
        """Renders both polarity phrasings of a condition with the deterministic
        template (no confidence tag). Returns (present_phrase, absent_phrase),
        normalized like the audited text. The operator must live on a
        synthetic condition — verbalize_condition prefers condition.operator
        over the card's condition string."""
        def phrase(op: str) -> str:
            synthetic = ExplanationCondition(
                feature_idx=cond.feature_idx, feature_name=cond.feature_name,
                operator=op, threshold=cond.threshold,
                observed_value=cond.observed_value,
            )
            card = VisualEvidenceCard(
                condition_str=synthetic.condition_str,
                feature_name=cond.feature_name,
                part_name=None, bbox=None, confidence=None,
            )
            return self._normalize_text(
                self._template.verbalize_condition(card, synthetic)
            )

        return phrase(">"), phrase("<=")

    def _lexical_omissions(
        self, norm_text: str, sufficient_explanation: SufficientExplanation
    ) -> List[str]:
        """Token-level omission check (kept as the fallback for foreign text)."""
        omitted: List[str] = []
        for cond in sufficient_explanation.conditions:
            f_name = cond.feature_name.lower().replace("_", " ")
            if ":" in f_name and "=" in f_name:
                part, attr_val = f_name.split(":", 1)
                attr, val = attr_val.split("=", 1)
                part_mentioned = (part in norm_text) or (part == "beak" and "bill" in norm_text)
                val_mentioned = self._normalize_value(val) in norm_text
                if not (part_mentioned and val_mentioned):
                    omitted.append(cond.feature_name)
            elif "=" in f_name:
                attr, val = f_name.split("=", 1)
                if self._normalize_value(val) not in norm_text:
                    omitted.append(cond.feature_name)
        return omitted

    def _lexical_unsupported(
        self, norm_text: str, sufficient_explanation: SufficientExplanation
    ) -> List[str]:
        """Token-level unsupported-claim scan: color/shape words in the text
        that no certified condition's feature name contains."""
        allowed_tokens: Set[str] = set()
        for cond in sufficient_explanation.conditions:
            for tok in re.split(r"[:=_\s\(\),]+", cond.feature_name.lower()):
                if tok:
                    allowed_tokens.add(tok)
        if "beak" in allowed_tokens:
            allowed_tokens.add("bill")

        unsupported: List[str] = []
        words = set(re.findall(r"\b[a-z]+\b", norm_text))
        for color in self.known_colors:
            if color in words and color not in allowed_tokens:
                unsupported.append(f"color:{color}")
        for shape in self.known_shapes:
            if shape in words and shape not in allowed_tokens:
                unsupported.append(f"shape:{shape}")
        return unsupported

    def audit(
        self,
        explanation_text: str,
        sufficient_explanation: SufficientExplanation,
        db: Optional[FactDatabase] = None,
    ) -> AuditReport:
        """
        Runs an audit of explanation_text against certified conditions.
        """
        norm_text = self._normalize_text(explanation_text)
        # The target class name is not a claim about the image: CUB species
        # names carry color words ("Brown_Booby", "Red_headed_Woodpecker")
        # that the lexical scan would otherwise flag as unsupported claims.
        class_span = self._normalize_text(
            str(sufficient_explanation.target_class_name or ""))
        scannable_text = (
            norm_text.replace(class_span, " ") if class_span else norm_text
        )

        # ---- claim mode: match each condition's bound phrasing by polarity ---
        covered, flips, omissions_claim = 0, [], []
        matched_any = False
        for cond in sufficient_explanation.conditions:
            present_phrase, absent_phrase = self._condition_phrasings(cond)
            wants_absent = cond.operator == "<="
            correct = absent_phrase if wants_absent else present_phrase
            flipped = present_phrase if wants_absent else absent_phrase
            if correct and correct in norm_text:
                covered += 1
                matched_any = True
            elif flipped and flipped in norm_text:
                flips.append(f"polarity-flip:{cond.feature_name}")
                matched_any = True
            else:
                omissions_claim.append(cond.feature_name)

        unsupported: List[str] = list(flips)

        if matched_any:
            # Template-family text: bound claims found — combine the binding
            # result with the lexical scan for any extra value words.
            omitted = omissions_claim
            unsupported += self._lexical_unsupported(scannable_text, sufficient_explanation)
            mode = "claim-binding"
        else:
            # Foreign/free-form text (e.g. an LLM explainer in the comparison
            # arm): no template phrasing appears. Extract claims heuristically
            # and cross them against the certified triples; keep the lexical
            # checks as the omission net.
            omitted = self._lexical_omissions(norm_text, sufficient_explanation)
            heuristic = self._heuristic_unsupported(
                explanation_text, sufficient_explanation
            )
            unsupported += heuristic
            unsupported += self._lexical_unsupported(scannable_text, sufficient_explanation)
            mode = "claim-heuristic" if heuristic or not omitted else "lexical"

        total_conds = len(sufficient_explanation.conditions)
        covered_conds = total_conds - len(omitted)
        recall = (covered_conds / total_conds) if total_conds > 0 else 1.0

        n_claims = covered_conds + len(unsupported)
        precision = (covered_conds / n_claims) if n_claims > 0 else 1.0

        is_faithful = (len(omitted) == 0) and (len(unsupported) == 0)

        summary = (
            f"Audit {'PASSED' if is_faithful else 'FAILED'} "
            f"[{mode}]: "
            f"{len(omitted)} omitted conditions, {len(unsupported)} unsupported claims. "
            f"Precision: {precision:.2f}, Recall: {recall:.2f}"
        )

        return AuditReport(
            is_faithful=is_faithful,
            omitted_conditions=omitted,
            unsupported_claims=unsupported,
            faithfulness_precision=precision,
            faithfulness_recall=recall,
            audit_summary=summary,
            metadata={"audit_mode": mode},
        )
