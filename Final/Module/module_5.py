import json
from typing import Dict, List, Any
from Module.gpt_seq_2_seq import gpt_call


# ============================================================
#                 MODULE 5 — Verdict + Justification
# ============================================================

class Module_5:
    """
    MODULE 5 — Verdict Aggregation + LLM Justification (dùng GPT từ gpt_seq_2_seq.py)

    Input:
        results_with_stance = {
            claim1: [
                {
                    "text": "...",
                    "link": "...",
                    "stance_scores": {"support": float, "refute": float, "neutral": float},
                    "rerank_score": float,        # optional
                    "stance_score": float         # optional (for best evidence)
                },
                ...
            ],
            ...
        }

    Output:
        {
            claim1: {
                "claim": str,
                "verdict": str,
                "confidence": float,
                "stance_ratio": dict,
                "best_evidence": dict,
                "justification": str
            },
            ...
        }
    """

    def __init__(self, client, model="gpt-5-nano-2025-08-07"):
        """
        client: client GPT đã được tạo từ SDK (bạn truyền vào từ Notebook)
        model: model GPT để gọi, mặc định theo gpt_seq_2_seq.py
        """
        self.client = client
        self.model = model

    # ----------------------------------------------------------
    # 1. Verdict Aggregation (Weighted)
    # ----------------------------------------------------------
    def aggregate_verdict_weighted(self, evidence_with_stance: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not evidence_with_stance:
            return {
                "verdict": "Unknown",
                "confidence": 0.0,
                "stance_ratio": {"Support": 0.0, "Refute": 0.0, "Neutral": 1.0}
            }

        weights = {"Support": 0.0, "Refute": 0.0, "Neutral": 0.0}
        total_weight = 0.0

        for e in evidence_with_stance:
            s = e["stance_scores"]
            rel = e.get("rerank_score", 1.0)

            weights["Support"] += s["support"] * rel
            weights["Refute"] += s["refute"] * rel
            weights["Neutral"] += s["neutral"] * rel
            total_weight += rel

        ratio = {k: v / total_weight for k, v in weights.items()}

        if ratio["Support"] > 0.5:
            verdict = "True"
        elif ratio["Refute"] > 0.5:
            verdict = "False"
        else:
            verdict = "Unknown"

        confidence = max(ratio.values())

        return {
            "verdict": verdict,
            "confidence": confidence,
            "stance_ratio": ratio
        }

    # ----------------------------------------------------------
    # 2. Gọi GPT sinh justification qua gpt_call()
    # ----------------------------------------------------------
    def generate_justification(self, claim, verdict, best_evidence, stance_ratio) -> str:

        system_prompt = (
            "Bạn là trợ lý fact-check chuyên nghiệp, trung lập, không được mâu thuẫn với verdict đã cho."
        )

        input_prompt = f"""
Dưới đây là dữ liệu fact-check:

Claim: "{claim}"
System verdict: {verdict}
Stance ratio: {stance_ratio}

Best evidence:
"{best_evidence['text']}"
Nguồn: {best_evidence.get('link','(no link)')}

Yêu cầu:
- Trả lời bằng tiếng Việt.
- Output KHÔNG phải JSON.
- Cấu trúc 2 phần:
  [Nhận định]: True / False / Unknown
  [Giải thích]: 2–3 câu, rõ ràng, trung lập.
- Không được mâu thuẫn với verdict đã cho.
- Nếu best evidence ủng hộ nhưng verdict = Unknown:
  phải viết dạng 'Mặc dù..., tổng thể bằng chứng hiện nay vẫn chưa đủ...'
"""

        # Gọi LLM qua gpt_call
        result_list = gpt_call(system_prompt, input_prompt, self.client, model=self.model, flag='raw_text')

        if isinstance(result_list, list):
            return result_list[0]  # lấy phần đầu tiên

        return str(result_list)

    # ----------------------------------------------------------
    # 3. Pipeline cuối
    # ----------------------------------------------------------
    def run(self, results_with_stance: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        final = {}

        for claim, evidences in results_with_stance.items():
            if not evidences:
                continue

            agg = self.aggregate_verdict_weighted(evidences)
            best_ev = max(evidences, key=lambda e: e.get("stance_score", 0.0))

            justification = self.generate_justification(
                claim, agg["verdict"], best_ev, agg["stance_ratio"]
            )

            final[claim] = {
                "claim": claim,
                "verdict": agg["verdict"],
                "confidence": agg["confidence"],
                "stance_ratio": agg["stance_ratio"],
                "best_evidence": best_ev,
                "justification": justification
            }

        return final
