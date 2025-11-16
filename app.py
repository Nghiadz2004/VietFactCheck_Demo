import os
import re
import math
import json
import time
import torch
import requests
import tldextract
import gradio as gr
from datetime import datetime
from typing import List, Tuple, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import plotly.graph_objects as go

# ============= Load ENV =============
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# ============= Azure LLM for Preprocessing + Claim Detection ============
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
embedder = SentenceTransformer("keepitreal/vietnamese-sbert")

stance_model_name = "joeddav/xlm-roberta-large-xnli"
tokenizer = AutoTokenizer.from_pretrained(stance_model_name, use_fast=False)
stance_model = AutoModelForSequenceClassification.from_pretrained(stance_model_name)

TRUSTED_DOMAINS = [
    "vnexpress.net", "tuoitre.vn", "thanhnien.vn", "nhandan.vn",
    "moh.gov.vn", "suckhoedoisong.vn", "zingnews.vn",
    "vietnamnet.vn", "baochinhphu.vn", "cafef.vn", "monre.gov.vn",
    "bbc.com", "reuters.com", "apnews.com", "theguardian.com",
    "cnn.com", "nytimes.com", "who.int", "un.org",
    "worldbank.org", "nature.com", "sciencedirect.com"
]

BAD_DOMAINS = [
    "blogspot", "wordpress", "weebly", "wixsite", "jimdo", "tumblr",
    "facebook", "twitter", "tiktok", "instagram", "youtube", "reddit", "pinterest",
    "voz.vn", "quora", "reddit", "kenh14", "webtretho", "otofun", "tinhte.vn",
    "tudientiengviet", "wiktionary", "vi.wiktionary", "tratu.soha.vn", "vdict",
    "hocmai", "loigiaihay", "olm.vn", "giaibaitap", "baitap123", "hoc24",
    "eva.vn", "2sao.vn", "afamily.vn", "tiin.vn", "yeah1", "bestie.vn", "blogtamsu",
]

# ============= MODULE 1 — Preprocessing text (fix teencode, chuẩn câu) ============
def preprocess_text(text):
    prompt = """
    Ghi lại những gì người dùng nhập vào nhưng đúng chính tả và ngữ pháp.
    Teencode, từ viết tắt , không dấu => chuyển về tiếng Việt chuẩn.
    Giữ nguyên nghĩa câu.
    """
    try:
        response = client.complete(
            messages=[
                SystemMessage(prompt),
                UserMessage(text)
            ],
            model=model_llm
        )
        return response.choices[0].message.content
    except Exception as e:
        # fallback: trả nguyên input nếu LLM lỗi
        print("Preprocess LLM error:", e)
        return text

# ============= MODULE 2 — Claim Detection ============
def extract_claims(text):
    prompt = """
    Bạn là mô-đun Claim Detection.
    Trả về danh sách những câu chứa claim có thể kiểm chứng.
    Bỏ câu cảm thán, xã giao, câu không kiểm chứng được.
    Chỉ trả về các claim mỗi dòng một câu.
    """
    try:
        response = client.complete(
            messages=[
                SystemMessage(prompt),
                UserMessage(text)
            ],
            model=model_llm
        )
        claims = response.choices[0].message.content.strip().split("\n")
        return [c.strip(" -•") for c in claims if len(c.strip()) > 8]
    except Exception as e:
        print("Claim detection LLM error:", e)
        # fallback: tách theo dấu chấm, giữ những câu đủ dài
        parts = [p.strip() for p in re.split(r"[.\n]", text) if p.strip()]
        return [p for p in parts if len(p) > 8]

# ============= MODULE 3 — Document Retrieval (Expand Query + Search) ============
MODEL = "openai/gpt-4o-mini"
client = ChatCompletionsClient(
    endpoint="https://models.github.ai/inference",
    credential=AzureKeyCredential(GITHUB_TOKEN),
)

# optional NER library
try:
    from underthesea import ner as underthesea_ner
    _HAVE_UNDER = True
except Exception:
    underthesea_ner = None
    _HAVE_UNDER = False

def _clean_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"^(paraphrase[:\-\s]+|viết lại[:\-\s]+|câu viết lại[:\-\s]+|rewrite[:\-\s]+)", "", t,
               flags=re.IGNORECASE)
    t = re.sub(r"</s>", "", t, flags=re.IGNORECASE)
    t = re.sub(r"[^0-9A-Za-zÀ-ỹ\u00C0-\u024F\u1E00-\u1EFF\.\,\?\!\:\;\-\(\)\/\%\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# For brevity: reuse expand_query implementation (same as notebook) but simplified fallback
def expand_query(claim: str) -> List[str]:
    claim = (claim or "").strip()
    if not claim:
        return []
    # Simple fallback - return claim and some trimmed variants
    return [claim, claim, claim, claim]

def serper_search(query: str, num_results: int = 5) -> list[dict]:
    if not SERPER_API_KEY:
        return []
    url = "https://google.serper.dev/search"
    payload = {"q": query, "num": num_results}
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        data = response.json()
        results = []
        for item in data.get("organic", [])[:num_results]:
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", "")
            })
        return results
    except Exception as e:
        print(f"[Serper error] {query[:40]}...: {e}")
        return []

# ============= MODULE 3 — Document Retrieval (Trust Score) ============
def compute_recency_score(pub_date_str: str, lam: float = 0.002) -> float:
    try:
        pub_date = datetime.fromisoformat(pub_date_str)
        days = (datetime.now() - pub_date).days
        return max(0, min(math.exp(-lam * days), 1))
    except Exception:
        return 0.5

def compute_trust_score(claim: str, source: dict) -> float:
    link = source.get("link", "")
    snippet = source.get("snippet", "") or ""
    domain = tldextract.extract(link).domain or ""

    try:
        emb_claim = embedder.encode(claim, convert_to_tensor=True)
        emb_snip = embedder.encode(snippet, convert_to_tensor=True)
        sim = util.cos_sim(emb_claim, emb_snip).item()
    except Exception:
        sim = 0.0

    claim_words = set(re.findall(r'\w+', claim.lower()))
    snip_words = set(re.findall(r'\w+', snippet.lower()))
    overlap = len(claim_words & snip_words) / len(claim_words) if claim_words else 0

    if any(d in domain for d in TRUSTED_DOMAINS):
        domain_score = 0.9
    elif any(d in domain for d in BAD_DOMAINS):
        domain_score = 0.2
    else:
        domain_score = 0.5

    vi_chars = re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', snippet)
    lang_bonus = 0.1 if len(vi_chars) > 5 else -0.1

    recency_score = compute_recency_score(source.get("date", ""))

    trust_score = (
        0.45 * sim +
        0.15 * overlap +
        0.2 * domain_score +
        0.1 * recency_score +
        0.1 * lang_bonus
    )
    trust_score = max(0, min(trust_score, 1))
    source["trust_reason"] = f"sim={sim:.2f}, domain={domain_score}, recency={recency_score:.2f}"
    return round(trust_score, 3)

def rerank_semantic_top50(claim: str, sources: list[dict], embedder: SentenceTransformer) -> list[dict]:
    if not sources:
        return []
    combined_texts = [f"{src.get('title','')}. {src.get('snippet','')}".strip() for src in sources]
    try:
        emb_claim = embedder.encode(claim, convert_to_tensor=True, normalize_embeddings=True)
        emb_candidates = embedder.encode(combined_texts, convert_to_tensor=True, normalize_embeddings=True, batch_size=32)
        sims = util.cos_sim(emb_claim, emb_candidates)[0]
    except Exception:
        sims = [0.0] * len(sources)

    for i, src in enumerate(sources):
        src["semantic_score"] = float(sims[i]) if len(sims) > i else 0.0

    for src in sources:
        trust = src.get("trust_score", 0)
        sem = src.get("semantic_score", 0)
        src["final_score"] = 0.7 * sem + 0.3 * trust

    ranked = sorted(sources, key=lambda x: x.get("final_score", 0), reverse=True)
    return ranked[:10]

def process_claim(claim: str) -> list[dict]:
    queries = expand_query(claim)
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(serper_search, q): q for q in queries}
        for f in as_completed(futures):
            try:
                results.extend(f.result())
                time.sleep(0.05)
            except Exception:
                pass

    filtered = []
    seen = set()
    for r in results:
        link = r.get("link")
        if not link or not link.startswith("http"):
            continue
        if link in seen:
            continue
        seen.add(link)
        domain = tldextract.extract(link).domain or ""
        if not domain:
            continue
        filtered.append(r)

    for src in filtered:
        src["trust_score"] = compute_trust_score(claim, src)

    top10 = rerank_semantic_top50(claim, filtered, embedder)
    return top10

# ============= MODULE 5 — Stance (NLI) =============
def predict_stance(claim, text):
    try:
        inp = tokenizer(claim, text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = stance_model(**inp).logits
            probs = torch.softmax(logits, dim=1)[0]
        return {
            "support": float(probs[2]),
            "refute": float(probs[0]),
            "neutral": float(probs[1])
        }
    except Exception as e:
        print("predict_stance error:", e)
        return {"support": 0.0, "refute": 0.0, "neutral": 1.0}

# ============= MODULE 6 — Verdict Aggregation ============
def aggregate_verdict(evidences):
    if not evidences:
        return {"verdict": "Unknown", "confidence": 0.0, "stance_ratio": {"Support":0.0,"Refute":0.0,"Neutral":1.0}}
    weights = {"Support": 0.0, "Refute": 0.0, "Neutral": 0.0}
    total = 0.0
    for e in evidences:
        w = e.get("trust_score", 0.0)
        s = e.get("stance_scores", {"support":0,"refute":0,"neutral":1})
        weights["Support"] += s["support"] * w
        weights["Refute"] += s["refute"] * w
        weights["Neutral"] += s["neutral"] * w
        total += w
    if total == 0:
        return {"verdict": "Unknown", "confidence": 0.0, "stance_ratio": {"Support":0.0,"Refute":0.0,"Neutral":1.0}}
    ratio = {k: v / total for k, v in weights.items()}
    if ratio["Support"] > 0.5:
        verdict = "True"
    elif ratio["Refute"] > 0.5:
        verdict = "False"
    else:
        verdict = "Unknown"
    confidence = max(ratio.values())
    return {"verdict": verdict, "confidence": confidence, "stance_ratio": ratio}

# ============= FULL FACT CHECK PIPELINE (returns markdown + stance ratio) ============
def fact_check_full(text):
    processed = preprocess_text(text)
    claims = extract_claims(processed)
    if not claims:
        md = "Không tìm thấy câu chứa claim có thể kiểm chứng."
        # return md + empty chart
        return md, {"Support":0.0,"Refute":0.0,"Neutral":1.0}

    all_blocks = []
    # We'll generate chart for the first claim's aggregation (if multiple claims)
    first_agg = None

    for claim in claims:
        docs = process_claim(claim)
        evidences = []
        for d in docs:
            snippet = d.get("snippet", "") or ""
            link = d.get("link", "") or ""
            trust = d.get("trust_score", 0) or 0
            stance = predict_stance(claim, snippet)
            evidences.append({
                "text": snippet,
                "link": link,
                "stance_scores": stance,
                "trust_score": trust
            })

        if not evidences:
            all_blocks.append(f"\n### Claim: **{claim}**\nKhông tìm được bằng chứng.\n")
            continue

        best_evidence = max(evidences, key=lambda e: e.get("trust_score", 0))
        agg = aggregate_verdict(evidences)
        if first_agg is None:
            first_agg = agg

        block = f"""
### Claim: **{claim}**
Kết luận: **{agg['verdict']}** (độ tin cậy: {agg['confidence']:.2f})  
Stance ratio: {agg['stance_ratio']}

**Bằng chứng mạnh nhất:**
- Nguồn: {best_evidence.get('link','(no link)')}
- Nội dung: {best_evidence.get('text','')[:450]}...
"""
        all_blocks.append(block)

    md_text = "\n".join(all_blocks)
    if first_agg is None:
        stance_ratio = {"Support":0.0,"Refute":0.0,"Neutral":1.0}
    else:
        stance_ratio = first_agg.get("stance_ratio", {"Support":0.0,"Refute":0.0,"Neutral":1.0})
    return md_text, stance_ratio

# ============= Visualization: Plotly pie chart ============
def render_stance_chart(ratio):
    # ensure numeric and normalized
    s = float(ratio.get("Support", 0.0))
    n = float(ratio.get("Neutral", 0.0))
    r = float(ratio.get("Refute", 0.0))
    total = s + n + r
    if total == 0:
        vals = [0.0, 1.0, 0.0]
    else:
        vals = [s/total, n/total, r/total]

    labels = ["Ủng hộ (Support)", "Trung lập (Neutral)", "Phản bác (Refute)"]
    colors = ["#2ECC71", "#9E9E9E", "#FF6B6B"]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=vals,
            marker=dict(colors=colors, line=dict(color="white", width=2)),
            hole=0.45,
            sort=False,
            textinfo="label+percent"
        )
    )
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5)
    )
    return fig

# ============= Gradio UI =============
def verify_fn(claim: str):
    claim = (claim or "").strip()
    if not claim:
        # hide chart and clear markdown
        return gr.update(value="", visible=False), gr.update(visible=False)
    md, stance_ratio = fact_check_full(claim)
    fig = render_stance_chart(stance_ratio)
    # return (markdown output, plot output)
    return gr.update(value=md, visible=True), fig

with gr.Blocks(
    title="VN Claim Verifier",
    css="""
    body { background-color: #f5f6f8; font-family: Inter, sans-serif; }
    #title-bar { font-size: 26px; font-weight:700; padding:20px 0; margin-bottom:10px; }
    #input-box textarea { font-size:18px; padding:14px; border-radius:12px; border:1px solid #d5d7da; background:white; }
    #send-btn button { height:48px; width:48px; border-radius:10px; background:#1a73e8 !important; color:white; font-size:18px; border:none; }
    #result-card { background:white; padding:20px; border-radius:14px; margin-top:12px; box-shadow:0 3px 12px rgba(0,0,0,0.06); font-size:15px; line-height:1.5; }
    """
) as ui:

    # HEADER
    gr.Markdown(
        """
        <div id="title-bar">🔎 <span style="color:#1a73e8">Verify a Claim</span></div>
        <div style="color:#777; margin-bottom:12px;">Enter a statement, headline or claim to analyze its validity.</div>
        """
    )

    # INPUT ROW
    with gr.Row():
        claim_box = gr.Textbox(
            placeholder="Nhập claim cần kiểm chứng…",
            lines=2,
            elem_id="input-box",
            scale=8
        )
        send_btn = gr.Button(
            value="➤",
            elem_id="send-btn",
            scale=1
        )

    # RESULT: Markdown + Plot
    with gr.Row():
        result_md = gr.Markdown(value="", visible=False, elem_id="result-card", scale=7)
        result_plot = gr.Plot(visible=False, elem_id="result-plot", scale=3)

    # EVENTS
    send_btn.click(fn=verify_fn, inputs=[claim_box], outputs=[result_md, result_plot])
    claim_box.submit(fn=verify_fn, inputs=[claim_box], outputs=[result_md, result_plot])

if __name__ == "__main__":
    ui.launch()