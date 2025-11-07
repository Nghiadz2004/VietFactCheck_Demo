import os
import torch
import gradio as gr
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from duckduckgo_search import DDGS
from urllib.parse import urlparse
from dotenv import load_dotenv

# ============= Load ENV =============
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# ============= Azure LLM for Preprocessing + Claim Detection =============
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

endpoint = "https://models.github.ai/inference"
model_llm = "openai/gpt-4o-mini"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(GITHUB_TOKEN),
)

# ============= Load SBERT + NLI model =============
embed_model = SentenceTransformer("keepitreal/vietnamese-sbert")

stance_model_name = "joeddav/xlm-roberta-large-xnli"
tokenizer = AutoTokenizer.from_pretrained(stance_model_name, use_fast=False)
stance_model = AutoModelForSequenceClassification.from_pretrained(stance_model_name)

TRUSTED_DOMAINS = ["vnexpress.net", "tuoitre.vn", "thanhnien.vn", "zingnews.vn", "vietnamplus.vn"]
BAD_DOMAINS = ["facebook.com", "tiktok.com", "blogspot.com", "wordpress.com"]


# ============= MODULE 1 — Preprocessing text (fix teencode, chuẩn câu) =============
def preprocess_text(text):
    prompt = """
    Ghi lại những gì người dùng nhập vào nhưng đúng chính tả và ngữ pháp.
    Teencode, từ viết tắt , không dấu => chuyển về tiếng Việt chuẩn.
    Giữ nguyên nghĩa câu.
    """
    response = client.complete(
        messages=[
            SystemMessage(prompt),
            UserMessage(text)
        ],
        model=model_llm
    )
    return response.choices[0].message.content


# ============= MODULE 2 — Claim Detection =============
def extract_claims(text):
    prompt = """
    Bạn là mô-đun Claim Detection.
    Trả về danh sách những câu chứa claim có thể kiểm chứng.
    Bỏ câu cảm thán, xã giao, câu không kiểm chứng được.
    Chỉ trả về các claim mỗi dòng một câu.
    """
    response = client.complete(
        messages=[
            SystemMessage(prompt),
            UserMessage(text)
        ],
        model=model_llm
    )
    claims = response.choices[0].message.content.strip().split("\n")
    return [c.strip(" -•") for c in claims if len(c.strip()) > 8]


# ============= MODULE 3 — Document Retrieval (DuckDuckGo) =============
def get_domain(url):
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except:
        return ""

def search_documents(query, max_docs=5):
    results = []
    with DDGS() as crawler:
        for r in crawler.text(query, max_results=max_docs):
            if r.get("href") and r.get("body"):
                results.append({
                    "url": r["href"],
                    "text": r["body"]
                })
    return results


# ============= MODULE 4 — Trust Score =============
def compute_trust_score(claim, text, url):
    sim = util.cos_sim(embed_model.encode(claim), embed_model.encode(text)).item()

    c_tokens = set(claim.lower().split())
    t_tokens = set(text.lower().split())
    overlap = len(c_tokens & t_tokens) / max(len(c_tokens), 1)

    domain = get_domain(url)
    domain_bonus = 0.25 if domain in TRUSTED_DOMAINS else -0.3 if domain in BAD_DOMAINS else 0

    lang_bonus = 0.1 if any(ch in "ăâđêôơưáàảạãéèẻẹẽóòỏọõíìỉịĩúùủụũ" for ch in text.lower()) else -0.1

    score = 0.5 * sim + 0.2 * overlap + domain_bonus + lang_bonus
    return max(0, min(1, score))


# ============= MODULE 5 — Stance (NLI) =============
def predict_stance(claim, text):
    inp = tokenizer(claim, text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = stance_model(**inp).logits
        probs = torch.softmax(logits, dim=1)[0]

    return {
        "support": float(probs[2]),   # entailment
        "refute": float(probs[0]),    # contradiction
        "neutral": float(probs[1])    # neutral
    }


# ============= MODULE 6 — Verdict Aggregation =============
def aggregate_verdict(evidences):
    if not evidences:
        return {"verdict": "Unknown", "confidence": 0.0, "stance_ratio": {}}

    weights = {"Support": 0, "Refute": 0, "Neutral": 0}
    total = 0

    for e in evidences:
        w = e["trust_score"]
        s = e["stance_scores"]
        weights["Support"] += s["support"] * w
        weights["Refute"] += s["refute"] * w
        weights["Neutral"] += s["neutral"] * w
        total += w

    ratio = {k: v / total for k, v in weights.items()}

    if ratio["Support"] > 0.5:
        verdict = "True"
    elif ratio["Refute"] > 0.5:
        verdict = "False"
    else:
        verdict = "Unknown"

    confidence = max(ratio.values())
    return {"verdict": verdict, "confidence": confidence, "stance_ratio": ratio}


# ============= FULL FACT CHECK PIPELINE =============
def fact_check_full(text):
    processed = preprocess_text(text)
    claims = extract_claims(processed)

    if not claims:
        return "Không tìm thấy câu chứa claim có thể kiểm chứng."

    results = []
    for claim in claims:
        docs = search_documents(claim, max_docs=6)

        evidences = []
        for d in docs:
            trust = compute_trust_score(claim, d["text"], d["url"])
            stance = predict_stance(claim, d["text"])
            evidences.append({
                "text": d["text"],
                "link": d["url"],
                "stance_scores": stance,
                "trust_score": trust
            })

        if not evidences:
            results.append(f"\n### Claim: **{claim}**\nKhông tìm được bằng chứng.\n")
            continue

        best_evidence = max(evidences, key=lambda e: e["trust_score"])
        agg = aggregate_verdict(evidences)

        block = f"""
### Claim: **{claim}**
Kết luận: **{agg['verdict']}** (độ tin cậy: {agg['confidence']:.2f})
Stance ratio: {agg['stance_ratio']}

**Bằng chứng mạnh nhất:**
- Nguồn: {best_evidence['link']}
- Nội dung: {best_evidence['text'][:450]}...
"""
        results.append(block)

    return "\n".join(results)


# ============= Gradio UI =============
ui = gr.Interface(
    fn=fact_check_full,
    inputs=gr.Textbox(lines=5, placeholder="Nhập 1 hoặc nhiều câu..."),
    outputs=gr.Markdown(),
    title="🇻🇳 Vietnamese Fact-Check – Full Pipeline",
    description="Nhập một đoạn văn, hệ thống sẽ chuẩn hóa, tách claim, tìm bằng chứng, tính stance + trust và kết luận."
)

if __name__ == "__main__":
    ui.launch()