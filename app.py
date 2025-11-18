import os
import re
import math
import json
import time
import torch
import requests
import tldextract
import streamlit as st
from datetime import datetime
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import plotly.graph_objects as go

# ----------------- CONFIG -----------------
st.set_page_config(page_title="VietFactCheck Pro", layout="wide", page_icon="🛡️")
if "history" not in st.session_state:
    st.session_state["history"] = []

# CSS: dark mode + card styles (tune as needed)
st.markdown(
    """
    <style>
    .stApp { background-color: #0f1620; color: #e6eef6; font-family: Inter, sans-serif; }
    header {visibility: hidden;}
    .result-card { background-color: #141821; border: 1px solid #232936; border-radius: 14px; padding: 22px; }
    .user-bubble { background: linear-gradient(90deg,#1e88e5,#1976d2); color: white; padding:14px 18px; border-radius: 12px; display:inline-block; font-weight:600; }
    .label-small { color:#9aa6b2; font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.6px; }
    .evidence-box { background:#0e1620; border-left:4px solid #1e88e5; padding:14px; border-radius:8px; color:#cfe7ff; font-style:italic; }
    .badge { padding:6px 14px; border-radius:20px; font-weight:700; font-size:13px; display:inline-block; }
    .badge-supported { background:#e8f6ec; color:#1b5e20; border:1px solid #1b5e20; }
    .badge-refuted { background:#fdecea; color:#7b1e1e; border:1px solid #7b1e1e; }
    .badge-unknown { background:#2a2f36; color:#cdd6df; border:1px solid #3a4149; }
    .conf-container { width:100%; background:#1b2230; height:10px; border-radius:6px; overflow:hidden; }
    .conf-fill { height:100%; background:linear-gradient(90deg,#1e88e5,#42a5f5); }
    .stance-container { display:flex; height:12px; width:100%; border-radius:6px; overflow:hidden; background:#121519; }
    .st-sup { background:#29b06f; }
    .st-ref { background:#ff6b6b; }
    .st-neu { background:#7a7f86; }
    </style>
    """,
    unsafe_allow_html=True,
)

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
@st.cache_resource 
def load_models():
    embedder = SentenceTransformer("keepitreal/vietnamese-sbert")
    stance_model_name = "joeddav/xlm-roberta-large-xnli"
    tokenizer = AutoTokenizer.from_pretrained(stance_model_name, use_fast=False)
    stance_model = AutoModelForSequenceClassification.from_pretrained(stance_model_name)
    return embedder, tokenizer, stance_model

# Gọi hàm load
embedder, tokenizer, stance_model = load_models()

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
def fact_check_full(text: str):
    processed = preprocess_text(text)
    claims = extract_claims(processed)
    if not claims:
        return "Không tìm thấy claim có thể kiểm chứng.", {"Support":0.0,"Refute":0.0,"Neutral":1.0}
    blocks=[]
    first_agg=None
    for claim in claims:
        docs = process_claim(claim)
        evidences=[]
        for d in docs:
            snippet = d.get("snippet","") or ""
            link = d.get("link","") or ""
            trust = d.get("trust_score", 0.0) or 0.0
            stance = predict_stance(claim, snippet)
            evidences.append({"text": snippet, "link": link, "stance_scores": stance, "trust_score": trust})
        if not evidences:
            blocks.append(f"### Claim: **{claim}**\nKhông tìm được bằng chứng.\n")
            continue
        best = max(evidences, key=lambda e: e.get("trust_score",0))
        agg = aggregate_verdict(evidences)
        if first_agg is None:
            first_agg = agg
        block = (
            f"### Claim: **{claim}**\n"
            f"Kết luận: **{agg['verdict']}** (độ tin cậy: {agg['confidence']:.2f})  \n"
            f"Stance ratio: {agg['stance_ratio']}\n\n"
            f"**Bằng chứng mạnh nhất:**  \n- Nguồn: {best.get('link','(no link)')}  \n"
            f"- Nội dung: {best.get('text','')[:600]}...\n"
        )
        blocks.append(block)
    md_text = "\n".join(blocks)
    stance_ratio = first_agg.get("stance_ratio", {"Support":0.0,"Refute":0.0,"Neutral":1.0}) if first_agg else {"Support":0.0,"Refute":0.0,"Neutral":1.0}
    return md_text, stance_ratio

# ----------------- Visualization -----------------
def render_stance_chart(ratio: dict) -> go.Figure:
    s = float(ratio.get("Support",0.0))
    n = float(ratio.get("Neutral",0.0))
    r = float(ratio.get("Refute",0.0))
    total = max(s + n + r, 1e-9)
    vals = [s/total, n/total, r/total]
    labels = ["Ủng hộ (Support)", "Trung lập (Neutral)", "Phản bác (Refute)"]
    colors = ["#2ECC71", "#9E9E9E", "#FF6B6B"]
    fig = go.Figure(go.Pie(labels=labels, values=vals, hole=0.45, marker=dict(colors=colors, line=dict(color='rgba(0,0,0,0)', width=0)), textinfo='percent+label'))
    fig.update_layout(margin=dict(t=8,b=8,l=8,r=8), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation='h',y=-0.05))
    return fig


# ----------------- STREAMLIT UI -----------------
def strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ").replace("&quot;", '"')
    return text.strip()

def render_result_card(md_text: str, stance_ratio: dict, claim_text: str, best_evidence: dict = None):
    # parse best evidence fields
    src_link = best_evidence.get("link","") if best_evidence else ""
    src_snip = strip_html(best_evidence.get("text","")) if best_evidence else ""
    src_domain = tldextract.extract(src_link).domain if src_link else "N/A"
    agg = aggregate_verdict([best_evidence]) if best_evidence else {"verdict":"Unknown","confidence":0.0,"stance_ratio":{"Support":0.0,"Refute":0.0,"Neutral":1.0}}
    # compute values for visuals: use stance_ratio & confidence from agg when available
    confidence_pct = int(round(agg.get("confidence", 0.0) * 100))
    # fallback stance_ratio normalized
    s = stance_ratio.get("Support",0.0)*100
    ref = stance_ratio.get("Refute",0.0)*100
    neu = stance_ratio.get("Neutral",0.0)*100

    # Verdict badge mapping
    verdict_display = {"True":("Supported","badge-supported"), "False":("Refuted","badge-refuted"), "Unknown":("Unproven","badge-unknown")}
    v_label, v_class = verdict_display.get(agg.get("verdict","Unknown"), ("Unproven","badge-unknown"))

    st.markdown(f"""
    <div class="result-card">
      <div style="display:flex; justify-content:space-between; align-items:flex-end; gap:12px; margin-bottom:18px;">
        <div style="flex:1">
          <div class="label-small">KẾT LUẬN (VERDICT)</div>
          <div style="margin-top:8px;"><span class="badge {v_class}">{v_label}</span></div>
        </div>
        <div style="width:45%; text-align:right;">
          <div class="label-small">ĐỘ TIN CẬY AI (CONFIDENCE)</div>
          <div style="display:flex; align-items:center; justify-content:flex-end; gap:8px; margin-top:8px;">
            <div style="flex:1; margin-right:8px;">
              <div class="conf-container"><div class="conf-fill" style="width:{confidence_pct}%;"></div></div>
            </div>
            <div style="min-width:48px; font-weight:700;">{confidence_pct}%</div>
          </div>
        </div>
      </div>

      <div style="margin-bottom:18px;">
        <div class="label-small">TỈ LỆ QUAN ĐIỂM (STANCE RATIO)</div>
        <div style="margin-top:8px;">
          <div class="stance-container">
            <div class="st-sup" style="width:{max(0,min(100,s))}%;"></div>
            <div class="st-ref" style="width:{max(0,min(100,ref))}%;"></div>
            <div class="st-neu" style="width:{max(0,min(100,neu))}%;"></div>
          </div>
          <div style="display:flex; justify-content:space-between; color:#9aa6b2; font-size:13px; margin-top:8px;">
            <div style="color:#2ecc71;">● Supporting: {int(round(s))}%</div>
            <div style="color:#ff6b6b;">● Refuting: {int(round(ref))}%</div>
            <div style="color:#7a7f86;">● Neutral: {int(round(neu))}%</div>
          </div>
        </div>
      </div>

      <div style="border-top:1px solid #232936; padding-top:14px;">
        <div style="font-weight:700; color:#d8e9ff; margin-bottom:10px;">Bằng chứng mạnh nhất (Strongest Evidence)</div>

        <div class="label-small">NGUỒN (SOURCE)</div>
        <div style="margin-bottom:10px;">🌐 <a href="{src_link}" target="_blank" style="color:#8fcfff;"><b>{src_domain}</b></a></div>

        <div class="label-small">TRÍCH DẪN (CONTENT)</div>
        <div class="evidence-box">"{src_snip}"</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------- APP LAYOUT -----------------
st.title("🔎 Verify a Claim")
st.markdown("Enter a statement, news headline, or claim to analyze its validity.")
st.markdown("---")

# Input row: text input + send button
with st.form("claim_form", clear_on_submit=False):
    col1, col2 = st.columns([10,1])
    with col1:
        txt = st.text_area(
            "Nhập claim cần kiểm chứng...",
            height=90,
            placeholder="Ví dụ: Việt Nam sẽ cấm hoàn toàn việc sử dụng tiền mặt..."
        )
    with col2:
        send = st.form_submit_button("➤", help="Gửi để kiểm chứng")

# on-send: run pipeline
if send and txt.strip():
    st.session_state["history"].append(txt.strip())
    st.markdown(f'<div class="user-bubble">👤 "{strip_html(txt)}"</div>', unsafe_allow_html=True)
    st.write("")  # spacing

    with st.spinner("Đang phân tích và tìm bằng chứng... (có thể mất vài giây)"):
        try:
            md, ratio = fact_check_full(txt)
        except Exception as e:
            st.error("Lỗi khi chạy pipeline: " + str(e))
            md, ratio = "Lỗi nội bộ khi kiểm chứng.", {"Support":0.0,"Refute":0.0,"Neutral":1.0}

        # get best evidence from process_claim (for rendering detailed card)
        # We'll call process_claim once for the first detected claim to retrieve best evidence list
        first_claims = extract_claims(preprocess_text(txt))
        best_ev = None
        if first_claims:
            try:
                docs = process_claim(first_claims[0])
                if docs:
                    # docs already have trust_score; pick best
                    best_ev = docs[0]
                else:
                    best_ev = None
            except Exception:
                best_ev = None

        # render card + chart side-by-side
        left, right = st.columns([7,3])
        with left:
            # parse md to show details too
            render_result_card(md, ratio, txt, best_ev)
            st.markdown("")  # spacing
            # optional raw markdown summary (collapsible)
            with st.expander("Xem chi tiết kết quả (markdown)"):
                st.markdown(md)
        with right:
            fig = render_stance_chart(ratio)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.caption("Powered by your local pipeline — SBERT + XNLI + Serper/LLM (if configured).")

else:
    # initial welcome card
    st.markdown(
        """
        <div class="result-card">
            <div style="font-weight:700; font-size:18px; margin-bottom:8px;">🌟 Chào bạn — VietFactCheck Pro</div>
            <div style="color:#9aa6b2;">Nhập một tuyên bố (claim) hoặc tiêu đề tin tức bên trên để kiểm chứng. Hệ thống sẽ tìm kiếm, phân tích stance và đưa ra kết luận cùng bằng chứng mạnh nhất.</div>
        </div>
        """, unsafe_allow_html=True
    )
    st.write("")

if st.session_state["history"]:
    st.markdown("### 🕓 Lịch sử câu hỏi")
    for i, item in enumerate(st.session_state["history"][::-1], start=1):
        st.markdown(f"- **{item}**")
        
# ----------------- END -----------------