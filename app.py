import time
import streamlit as st
from logic import Fact_Checking_Pipeline
import tldextract
import re
import plotly.graph_objects as go
import asyncio

# =========================
# PAGE CONFIG + CSS UI
# =========================
st.set_page_config(page_title="VietFactCheck Pro", layout="wide", page_icon="🛡️")

if "history" not in st.session_state:
    st.session_state["history"] = []

# CSS giao diện
st.markdown(
    """
    <style>
    * { box-sizing: border-box; } 
    .stApp { background-color: #0f1620; color: #e6eef6; font-family: Inter, sans-serif; }
    header {visibility: hidden;}

    .result-card { 
        background-color: #141821; 
        border: 1px solid #232936; 
        border-radius: 14px; 
        padding: 22px; 
        width: 100%;
        max-width: 100%;
        overflow: hidden;
    }

    .user-bubble { background: linear-gradient(90deg,#1e88e5,#1976d2); color: white; padding:12px 16px; border-radius: 12px; display:inline-block; max-width: 80%; }
    .label-small { color:#9aa6b2; font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.6px; }
    
    .badge { padding:6px 14px; border-radius:20px; font-weight:700; font-size:13px; }
    .badge-supported { background:#e8f6ec; color:#1b5e20; border:1px solid #1b5e20; }
    .badge-refuted { background:#fdecea; color:#7b1e1e; border:1px solid #7b1e1e; }
    .badge-unknown { background:#2a2f36; color:#cdd6df; border:1px solid #3a4149; }

    .conf-container { width:100%; background:#1b2230; height:10px; border-radius:6px; overflow:hidden; }
    .conf-fill { height:100%; background:linear-gradient(90deg,#1e88e5,#42a5f5); }

    .stance-container { display:flex; height:12px; width:100%; border-radius:6px; overflow:hidden; background:#121519; }
    .st-sup { background:#29b06f; }
    .st-ref { background:#ff6b6b; }
    .st-neu { background:#7a7f86; }

    .evidence-box { background:#0e1620; border-left:4px solid #1e88e5; padding:14px; border-radius:8px; color:#cfe7ff; font-style:italic; margin-top: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# HELPERS
# =========================
def strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ").replace("&quot;", '"')
    return text.strip()


def render_stance_chart(ratio: dict) -> go.Figure:
    s = float(ratio.get("Support", 0))
    n = float(ratio.get("Neutral", 0))
    r = float(ratio.get("Refute", 0))

    fig = go.Figure(
        go.Pie(
            labels=["Ủng hộ", "Trung lập", "Phản bác"],
            values=[s, n, r],
            hole=0.45,
            textinfo="percent+label"
        )
    )
    fig.update_layout(
        margin=dict(t=8, b=8, l=8, r=8),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


# =========================
# RENDER RESULT CARD
# =========================
def render_result_card(
    stance_ratio: dict,
    claim_text: str,
    best_evidence: dict,
    verdict: str,
    confidence: float
):
    if not best_evidence:
        src_link = ""
        src_snip = "Không có bằng chứng."
        src_domain = "N/A"
    else:
        src_link = best_evidence.get("link", "")
        raw_snip = best_evidence.get("text", "")
        src_snip = strip_html(raw_snip)
        src_domain = tldextract.extract(src_link).domain or "N/A"

    confidence_pct = int(confidence * 100)

    s = stance_ratio.get("Support", 0) * 100
    ref = stance_ratio.get("Refute", 0) * 100
    neu = stance_ratio.get("Neutral", 0) * 100

    verdict_map = {
        "True": ("Supported", "badge-supported"),
        "False": ("Refuted", "badge-refuted"),
        "Unknown": ("Unproven", "badge-unknown"),
    }

    v_label, v_class = verdict_map.get(verdict, ("Unproven", "badge-unknown"))

    html = f"""
    <div class="result-card">
        <div style="margin-bottom:18px;">
            <div class="label-small">KẾT LUẬN</div>
            <span class="badge {v_class}">{v_label}</span>
        </div>

        <div class="label-small">ĐỘ TIN CẬY AI</div>
        <div class="conf-container"><div class="conf-fill" style="width:{confidence_pct}%;"></div></div>
        <div style="text-align:right;margin-bottom:16px;font-weight:700;">{confidence_pct}%</div>

        <div class="label-small">TỈ LỆ QUAN ĐIỂM</div>
        <div class="stance-container">
            <div class="st-sup" style="width:{s}%;"></div>
            <div class="st-ref" style="width:{ref}%;"></div>
            <div class="st-neu" style="width:{neu}%;"></div>
        </div>

        <div style="margin-top:10px;color:#9aa6b2;font-size:13px;">
            <b style="color:#2ecc71;">● Support: {int(s)}%</b> — 
            <b style="color:#ff6b6b;">● Refute: {int(ref)}%</b> — 
            <b style="color:#7a7f86;">● Neutral: {int(neu)}%</b>
        </div>

        <hr style="border-color:#232936;margin:18px 0;">

        <div class="label-small">NGUỒN</div>
        🌐 <a href="{src_link}" target="_blank" style="color:#8fcfff;"><b>{src_domain}</b></a>

        <div class="label-small" style="margin-top:10px;">TRÍCH DẪN</div>
        <div class="evidence-box">"{src_snip}"</div>
    </div>
    """

    st.markdown(re.sub(r'\s+', ' ', html).strip(), unsafe_allow_html=True)


# =========================
# MAIN UI
# =========================
st.title("🔎 Verify a Claim")
st.markdown("Nhập vào một claim để kiểm chứng bằng chứng và stance.")

with st.form("input_form"):
    user_input = st.text_area("Nội dung kiểm chứng", height=100)
    sent = st.form_submit_button("Kiểm chứng")

# =========================
# ON SUBMIT
# =========================
if sent and user_input.strip():
    with st.spinner("Đang phân tích..."):
        result = asyncio.run(Fact_Checking_Pipeline(user_input.strip()))
        
        for i, (c, r) in enumerate(result.items()):
            # --- Lấy dữ liệu ---
            ratio = r["stance_ratio"]
            best = r["best_evidence"]
            verdict = r["verdict"]
            conf = round(r["confidence"], 3)

            # Lưu lịch sử (giữ nguyên)
            st.session_state["history"].append({
                "question": r["claim"], "ratio": ratio, "best": best,
                "verdict": verdict, "confidence": conf
            })
            
            # --- GIAO DIỆN ---
            # Tạo một container cho từng result để gom nhóm
            with st.container():
                # QUAN TRỌNG: Khai báo cột BÊN TRONG vòng lặp
                col1, col2 = st.columns([2, 1]) 
                
                with col1:
                    st.markdown(f'<div class="user-bubble">👤 "{strip_html(r["claim"])}"</div>', unsafe_allow_html=True)
                    st.write("")
                    render_result_card(ratio, user_input, best, verdict, conf)

                with col2:
                    fig = render_stance_chart(ratio)
                    st.plotly_chart(fig, use_container_width=True, key=f"id_{i}")
            
            # Kẻ đường phân cách sau khi render xong 2 cột của item này
            st.divider()

# =========================
# HISTORY
# =========================
if st.session_state["history"]:
    st.subheader("Lịch sử kiểm chứng")

    for idx, item in enumerate(st.session_state["history"]):

        st.markdown(
            f'<div class="user-bubble">👤 "{strip_html(item["question"])}"</div>',
            unsafe_allow_html=True
        )

        left, right = st.columns([2, 1])

        with left:
            render_result_card(
                item["ratio"],
                item["question"],
                item["best"],
                item["verdict"],
                item["confidence"]
            )

        with right:
            fig = render_stance_chart(item["ratio"])
            st.plotly_chart(fig, use_container_width=True, key=f"hist_{idx}")

        st.markdown("---")
