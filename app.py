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


# ============= Load ENV =============
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

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
embedder = SentenceTransformer("keepitreal/vietnamese-sbert")

stance_model_name = "joeddav/xlm-roberta-large-xnli"
tokenizer = AutoTokenizer.from_pretrained(stance_model_name, use_fast=False)
stance_model = AutoModelForSequenceClassification.from_pretrained(stance_model_name)

TRUSTED_DOMAINS = [
    # Việt Nam
    "vnexpress.net", "tuoitre.vn", "thanhnien.vn", "nhandan.vn",
    "moh.gov.vn", "suckhoedoisong.vn", "zingnews.vn",
    "vietnamnet.vn", "baochinhphu.vn", "cafef.vn", "monre.gov.vn",

    # Quốc tế
    "bbc.com", "reuters.com", "apnews.com", "theguardian.com",
    "cnn.com", "nytimes.com", "who.int", "un.org",
    "worldbank.org", "nature.com", "sciencedirect.com"
]

BAD_DOMAINS = [
    # Blog cá nhân & nền tảng tạo blog miễn phí
    "blogspot", "wordpress", "weebly", "wixsite", "jimdo", "tumblr",

    # Mạng xã hội & chia sẻ video
    "facebook", "twitter", "tiktok", "instagram", "youtube", "reddit", "pinterest",

    # Diễn đàn, hỏi đáp, chia sẻ linh tinh
    "voz.vn", "quora", "reddit", "kenh14", "webtretho", "otofun", "tinhte.vn",

    # Trang từ điển / wiki không chính thống
    "tudientiengviet", "wiktionary", "vi.wiktionary", "tratu.soha.vn", "vdict",

    # Trang học sinh – chia sẻ bài làm / học liệu chưa kiểm chứng
    "hocmai", "loigiaihay", "olm.vn", "giaibaitap", "baitap123", "hoc24",

    # Nguồn tin rác / spam nội dung tổng hợp
    "eva.vn", "2sao.vn", "afamily.vn", "tiin.vn", "yeah1", "bestie.vn", "blogtamsu",
]


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


# ============= MODULE 3 — Document Retrieval (Expand Query) =============
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


# ---- Text cleaning ----
def _clean_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"^(paraphrase[:\-\s]+|viết lại[:\-\s]+|câu viết lại[:\-\s]+|rewrite[:\-\s]+)", "", t,
               flags=re.IGNORECASE)
    t = re.sub(r"</s>", "", t, flags=re.IGNORECASE)
    # keep Vietnamese letters and punctuation
    t = re.sub(r"[^0-9A-Za-zÀ-ỹ\u00C0-\u024F\u1E00-\u1EFF\.\,\?\!\:\;\-\(\)\/\%\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_entities_vn(text: str) -> List[Tuple[str, str]]:
    """
    Trả về danh sách (entity_text, label) từ câu tiếng Việt.
    Cố gắng dùng underthesea nếu có, nếu không fallback sang heuristics.
    Labels: PERSON, ORG, LOCATION, DATE, NUMBER, MISC
    """

    # try underthesea first
    try:
        if _HAVE_UNDER and underthesea_ner is not None:
            ents = underthesea_ner(text)
            # reconstruct contiguous non-'O' spans
            spans = []
            cur_tokens = []
            for item in ents:
                token = None
                tag = None
                if isinstance(item, (list, tuple)):
                    # pick first string-like as token
                    for e in item:
                        if isinstance(e, str) and token is None:
                            token = e
                    # find a tag-like element
                    for e in reversed(item):
                        if isinstance(e, str) and (e == 'O' or re.match(r'^[BIE]-', e) or (e.isupper() and len(e) <= 6)):
                            tag = e
                            break
                else:
                    token = str(item)
                    tag = 'O'

                if token is None:
                    continue

                if tag != 'O':
                    lab = 'MISC'
                    if 'PER' in tag or 'PERSON' in str(tag).upper():
                        lab = 'PERSON'
                    elif 'LOC' in tag or 'GPE' in str(tag):
                        lab = 'LOCATION'
                    elif 'ORG' in str(tag):
                        lab = 'ORG'
                    cur_tokens.append((token, lab))
                else:
                    if cur_tokens:
                        ent_text = ' '.join(t for t, _ in cur_tokens)
                        ent_label = cur_tokens[0][1]
                        spans.append((ent_text.strip(), ent_label))
                        cur_tokens = []
            if cur_tokens:
                ent_text = ' '.join(t for t, _ in cur_tokens)
                ent_label = cur_tokens[0][1]
                spans.append((ent_text.strip(), ent_label))

            # dedupe preserving order
            seen = set()
            out = []
            for e, l in spans:
                if e and e not in seen:
                    seen.add(e)
                    out.append((e, l))
            if out:
                return out
    except Exception:
        # ignore underthesea errors, fall through
        pass

    # If user has a custom ner() in scope, try calling it (older code used ner(text))
    try:
        if 'ner' in globals() and callable(globals()['ner']):
            ents = globals()['ner'](text)
            parsed = []
            for ent in ents:
                token = None
                tag = None
                if isinstance(ent, (list, tuple)):
                    if len(ent) >= 1 and isinstance(ent[0], str):
                        token = ent[0].strip()
                    if len(ent) > 3 and isinstance(ent[3], str):
                        tag = ent[3]
                    else:
                        for e in ent:
                            if isinstance(e, str) and (e == 'O' or re.match(r'^[BIE]-', e) or (e.isupper() and len(e) <= 6)):
                                tag = e
                                break
                else:
                    token = str(ent).strip()
                    tag = 'O'

                if token and token != '' and len(token) > 1 and tag and tag != 'O':
                    lab = 'MISC'
                    if 'PER' in str(tag) or 'PERSON' in str(tag).upper():
                        lab = 'PERSON'
                    elif 'LOC' in str(tag) or 'GPE' in str(tag):
                        lab = 'LOCATION'
                    elif 'ORG' in str(tag):
                        lab = 'ORG'
                    parsed.append((token, lab))

            # merge contiguous tokens with same label
            merged = []
            cur_tok = []
            cur_lab = None
            for tok, lab in parsed:
                if cur_lab is None or lab == cur_lab:
                    cur_tok.append(tok)
                    cur_lab = lab
                else:
                    merged.append((" ".join(cur_tok).strip(), cur_lab))
                    cur_tok = [tok]
                    cur_lab = lab
            if cur_tok:
                merged.append((" ".join(cur_tok).strip(), cur_lab))

            seen = set()
            out = []
            for e, l in merged:
                if e and e not in seen:
                    seen.add(e)
                    out.append((e, l))
            if out:
                return out
    except Exception:
        pass

    # final fallback heuristics: dates, numbers, capitalized sequences (heuristic for names)
    heur = []
    for m in re.finditer(r"\d{4}|\d{2}/\d{2}/\d{4}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}", text):
        heur.append((m.group(0), 'DATE'))
    for m in re.finditer(r"\d+[\.,]?\d*", text):
        heur.append((m.group(0), 'NUMBER'))
    for m in re.finditer(r"(?:[A-ZÀ-Ỹ][\wÀ-ỹ'\-]+(?:\s+|$)){1,4}", text):
        val = m.group(0).strip()
        if len(val) > 1 and val.lower() != val:
            heur.append((val, 'PERSON'))

    seen = set()
    out = []
    for e, l in heur:
        if e not in seen:
            seen.add(e)
            out.append((e, l))
    return out


# ---- Helper: mask entities from an entity list (preserve mapping order) ----
def mask_entities_from_list(text: str, entities: List[Tuple[str, str]]) -> Tuple[str, Dict[str, str]]:
    """
    Replace the first occurrence of each entity (in the order provided) with a placeholder [ENT_i].
    Returns (masked_text, mapping) where mapping maps placeholder -> original entity string.
    """
    masked = text
    mapping: Dict[str, str] = {}
    idx = 0
    for ent, _label in entities:
        if not ent or len(ent.strip()) == 0:
            continue
        ent_clean = ent.strip()
        ph = f"[ENT_{idx}]"
        pattern = re.escape(ent_clean)
        # try case-sensitive first, then case-insensitive
        new_masked, n = re.subn(pattern, ph, masked, count=1)
        if n == 0:
            new_masked, n = re.subn(pattern, ph, masked, count=1, flags=re.IGNORECASE)
        if n > 0:
            mapping[ph] = ent_clean
            masked = new_masked
            idx += 1
        else:
            # if exact match not found, skip that entity (do not create placeholder)
            continue
    return masked, mapping


# ---- Helper: mask entities from raw text using underthesea / heuristics ----
def mask_entities(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Mask entities by detecting them in-text. Returns (masked_text, mapping).
    Uses underthesea if available; else falls back to heuristics (numbers, capitalized sequences).
    """
    mapping: Dict[str, str] = {}
    masked = text

    if _HAVE_UNDER and underthesea_ner is not None:
        try:
            ents = underthesea_ner(text)
            spans = []
            cur_tokens = []
            cur_tag = None
            for item in ents:
                token = None
                tag = None
                if isinstance(item, (list, tuple)):
                    for e in item:
                        if isinstance(e, str) and token is None:
                            token = e
                    for e in reversed(item):
                        if isinstance(e, str) and (e == 'O' or re.match(r'^[BIE]-', e) or (e.isupper() and len(e) <= 6)):
                            tag = e
                            break
                else:
                    token = str(item)
                    tag = 'O'

                if token is None:
                    continue

                if tag != 'O':
                    if cur_tag == tag or cur_tag is None:
                        cur_tokens.append(token)
                        cur_tag = tag
                    else:
                        spans.append((cur_tag, " ".join(cur_tokens)))
                        cur_tokens = [token]
                        cur_tag = tag
                else:
                    if cur_tokens:
                        spans.append((cur_tag, " ".join(cur_tokens)))
                        cur_tokens = []
                        cur_tag = None
            if cur_tokens:
                spans.append((cur_tag, " ".join(cur_tokens)))

            cnt = 0
            for tag, value in spans:
                ph = f"[ENT_{cnt}]"
                # replace first occurrence
                masked = masked.replace(value, ph, 1)
                mapping[ph] = value
                cnt += 1
            return masked, mapping
        except Exception:
            pass

    # heuristics fallback
    # mask long digit sequences and capitalized sequences
    cnt = 0
    for m in re.finditer(r"\d{2,}", text):
        ph = f"[ENT_{cnt}]"
        mapping[ph] = m.group(0)
        masked = masked.replace(m.group(0), ph, 1)
        cnt += 1
    for m in re.finditer(r"(?:[A-ZÀ-Ỹ][\wÀ-ỹ'\-]+(?:\s+|$)){1,3}", masked):
        val = m.group(0).strip()
        if len(val) > 1 and val.lower() != val:
            ph = f"[ENT_{cnt}]"
            if ph not in mapping:
                mapping[ph] = val
                masked = masked.replace(val, ph, 1)
                cnt += 1
    return masked, mapping


# ---- Helper: unmask entities ----
def unmask_entities(text: str, mapping: Dict[str, str]) -> str:
    out = text
    for ph, val in mapping.items():
        out = out.replace(ph, val)
    return out



# ---------------- typed/guided expand_query using GPT (your function) ----------------
def expand_query(
    claim: str,
    temperature: float = 0.7,
    retries: int = 1
) -> List[str]:
    """
    Trả về list gồm 5 phần:
    [claim_chinh, paraphrase_1, paraphrase_2, headline_1, headline_2]
    - Sử dụng NER để mask entity trước khi gọi GPT, rồi unmask và kiểm tra.
    - KHÔNG thêm site:... tự động.
    """
    claim = (claim or "").strip()
    if not claim:
        return []

    # 1) extract & mask
    try:
        entities = extract_entities_vn(claim) if 'extract_entities_vn' in globals() else []
    except Exception:
        entities = []
    if entities:
        masked_claim, mapping = mask_entities_from_list(claim, entities)
        # if mask_entities_from_list didn't produce mapping (no exact matches), fallback to auto mask
        if not mapping:
            masked_claim, mapping = mask_entities(claim)
    else:
        try:
            masked_claim, mapping = mask_entities(claim)
        except Exception:
            masked_claim, mapping = claim, {}

    # 2) prepare prompt: ask GPT to return EXACTLY a JSON array of 4 strings:
    few_shot = (
        "Ví dụ (mask dùng [ENT_i]):\n"
        "Input (masked): \"[ENT_0] giành giải thưởng về giáo dục\"\n"
        "Output: [\"[ENT_0] thắng giải về giáo dục\", \"[ENT_0] được trao giải dự án giáo dục\", "
        "\"[ENT_0] gây chú ý vì chiến thắng!\", \"[ENT_0] đã chiến thắng?\"]\n\n"
    )

    instruction = (
        f"SINH 4 biến thể NGẮN cho truy vấn (trong dạng JSON array):\n"
        f"- 2 biến thể đầu: paraphrase/tìm kiếm (ngắn, cùng ý, phù hợp để tìm kiếm)\n"
        f"- 2 biến thể sau: dạng tiêu đề báo (câu hỏi hoặc cảm thán), tự nhiên, không giật tít\n"
        f"- Giữ NGUYÊN placeholder như [ENT_0], [ENT_1] nếu có; không thay thế chúng.\n"
        f"- Trả **CHỈ** một JSON array duy nhất, ví dụ: [\"p1\", \"p2\", \"h1\", \"h2\"]\n\n"
        f"Input (masked): \"{masked_claim}\"\n"
    )

    prompt_base = few_shot + instruction

    def call_gpt(prompt_text: str) -> str:
        try:
            resp = client.complete(
                messages=[
                    SystemMessage("Bạn chỉ trả về JSON array, KHÔNG có giải thích."),
                    UserMessage(prompt_text)
                ],
                model=MODEL,
                temperature=temperature
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print("expand_query: client error:", e)
            return ""

    def parse_json_array(text: str):
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        m = re.search(r"\[.*\]", text, re.S)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return None

    # 3) call GPT with retry
    parsed = None
    attempt_prompt = prompt_base
    raw = ""
    for attempt in range(retries + 1):
        raw = call_gpt(attempt_prompt)
        parsed = parse_json_array(raw)
        if parsed and len(parsed) >= 4:
            break
        # tighten prompt for retry
        attempt_prompt = "PHẢI CHỈ TRẢ VỀ 1 JSON ARRAY GỒM 4 ITEMS. " + instruction

    # fallback if GPT didn't return usable JSON
    if not parsed or len(parsed) < 4:
        # Best-effort: try to split raw lines if any
        if raw:
            lines = [ln.strip() for ln in re.split(r"\n|;|•|-|\u2022|\||\.", raw) if ln.strip()]
            parsed = lines[:4]
        else:
            # final fallback: simple deterministic paraphrases (very basic)
            parsed = [
                masked_claim,
                masked_claim,
                masked_claim,
                masked_claim
            ]
    # ensure length 4
    parsed = parsed[:4] + [masked_claim] * max(0, 4 - len(parsed))

    # 4) unmask and clean
    def unmask_and_clean(s: str) -> str:
        t = s
        for ph, orig in mapping.items():
            if ph in t:
                t = t.replace(ph, orig)
        return _clean_text(t)

    unmasked = [unmask_and_clean(s) for s in parsed]

    # 5) ensure entity preservation: prefer items that contain all mapping values
    def preserves_all(s: str) -> bool:
        if not mapping:
            return True
        return all(orig in s for orig in mapping.values())

    preserved = [s for s in unmasked if preserves_all(s)]
    final_variants = preserved if len(preserved) >= 4 else unmasked

    # 6) dedupe while preserving order, but we need exactly: [claim, p1, p2, h1, h2]
    deduped = []
    seen = set()
    for s in final_variants:
        if s and s not in seen:
            deduped.append(s)
            seen.add(s)
        if len(deduped) >= 4:
            break

    # If still <4, fill with simple transforms
    while len(deduped) < 4:
        # simple fallback: short rephrasings (use claim as fallback)
        deduped.append(claim)

    # 7) assemble final list with original claim first
    result = [claim] + deduped[:4]

    # Final safety: truncate whitespace and return
    return [r.strip() for r in result]



# ============= MODULE 3 — Document Retrieval (Serper Search) =============
def serper_search(query: str, num_results: int = 5) -> list[dict]:
    url = "https://google.serper.dev/search"
    payload = {"q": query, "num": num_results}
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
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
    
    

# ============= MODULE 3 — Document Retrieval (Trust Score) =============
def compute_recency_score(pub_date_str: str, lam: float = 0.002) -> float:
    """Tính điểm thời gian bằng hàm suy giảm mũ."""
    try:
        pub_date = datetime.fromisoformat(pub_date_str)
        days = (datetime.now() - pub_date).days
        return max(0, min(math.exp(-lam * days), 1))
    except Exception:
        return 0.5  # nếu không có ngày đăng

def compute_trust_score(claim: str, source: dict) -> float:
    """Tính trust_score chuẩn hóa dựa trên ngữ nghĩa, domain, thời gian, ngôn ngữ."""
    link = source.get("link", "")
    snippet = source.get("snippet", "")
    domain = tldextract.extract(link).top_domain_under_public_suffix

    # 1️⃣ Semantic similarity
    try:
        emb_claim = embedder.encode(claim, convert_to_tensor=True)
        emb_snip = embedder.encode(snippet, convert_to_tensor=True)
        sim = util.cos_sim(emb_claim, emb_snip).item()
    except Exception:
        sim = 0.0

    # 2️⃣ Lexical overlap
    claim_words = set(re.findall(r'\w+', claim.lower()))
    snip_words = set(re.findall(r'\w+', snippet.lower()))
    overlap = len(claim_words & snip_words) / len(claim_words) if claim_words else 0

    # 3️⃣ Domain reliability
    if any(d in domain for d in TRUSTED_DOMAINS):
        domain_score = 0.9
    elif any(d in domain for d in BAD_DOMAINS):
        domain_score = 0.2
    else:
        domain_score = 0.5

    # 4️⃣ Language bonus
    vi_chars = re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', snippet)
    lang_bonus = 0.1 if len(vi_chars) > 5 else -0.1

    # 5️⃣ Recency
    recency_score = compute_recency_score(source.get("date", ""))

    # 6️⃣ Tổng hợp (chuẩn hóa theo IR/fact-checking)
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
    """
    Re-rank theo ngữ nghĩa (title + snippet) và độ tin cậy.
    Trả về top 50 kết quả có độ tương đồng ngữ nghĩa cao nhất,
    có thể kết hợp với trust_score nếu có.
    """
    if not sources:
        return []

    # Kết hợp title + snippet để embedding toàn diện hơn
    combined_texts = [
        f"{src.get('title', '')}. {src.get('snippet', '')}".strip()
        for src in sources
    ]

    # Encode claim và candidate batch (tăng tốc)
    emb_claim = embedder.encode(claim, convert_to_tensor=True, normalize_embeddings=True)
    emb_candidates = embedder.encode(combined_texts, convert_to_tensor=True, normalize_embeddings=True, batch_size=32)

    # Tính cosine similarity
    sims = util.cos_sim(emb_claim, emb_candidates)[0]

    # Gán điểm semantic_score cho từng nguồn
    for i, src in enumerate(sources):
        src["semantic_score"] = float(sims[i])

    # Kết hợp trust_score
    for src in sources:
        trust = src.get("trust_score", 0)
        sem = src["semantic_score"]
        # công thức kết hợp chuẩn theo thực tiễn IR/fact-checking
        src["final_score"] = 0.7 * sem + 0.3 * trust

    # Sắp xếp theo final_score giảm dần
    ranked = sorted(sources, key=lambda x: x["final_score"], reverse=True)

    # Lấy top 50 kết quả tốt nhất
    return ranked[:10]


# ============= MODULE 3 — Document Retrieval (Pipeline) =============
def process_claim(claim: str) -> list[dict]:
    queries = expand_query(claim)
    print(f"\n=== Claim: {claim}")
    print(f"Sinh {len(queries)} truy vấn mở rộng.\n")

    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(serper_search, q): q for q in queries}
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"Tìm kiếm '{claim[:40]}...'"):
            results.extend(f.result())
            time.sleep(0.1)

    # Lọc trùng link và loại bỏ link không hợp lệ
    filtered = []
    seen = set()
    for r in results:
        link = r.get("link")
        if not link or not link.startswith("http"):
            continue
        if link in seen:
            continue
        seen.add(link)

        domain = tldextract.extract(link).top_domain_under_public_suffix

        if not domain:  # bỏ link rác
            continue

        filtered.append(r)

    # Tính trust score
    for src in filtered:
        src["trust_score"] = compute_trust_score(claim, src)

    # Re-rank theo ngữ nghĩa + trust_score
    top10 = rerank_semantic_top50(claim, filtered, embedder)
    return top10


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
        docs = process_claim(claim)

        evidences = []
        for d in docs:
            snippet = d.get("snippet", "")
            link = d.get("link", "")
            trust = d.get("trust_score", 0)

            stance = predict_stance(claim, snippet)

            evidences.append({
                "text": snippet,
                "link": link,
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
def chat_fn(history, message):
    if not message.strip():
        return history, ""

    reply = fact_check_full(message)
    history.append((message, reply))
    return history, ""

def init_chat():
    return [
        (
            None,
            "🌟 **Chào bạn! Đây là hệ thống kiểm chứng thông tin tiếng Việt.**\n\n"
            "Bạn có thể nhập bất kỳ claim nào để kiểm chứng.\n"
        )
    ]


with gr.Blocks(
    title="Vietnamese Fact-Check – Chat",
    css="""
    #send-btn button {
        background: none !important;
        border: none !important;
        padding: 0 !important;
    }
    """
) as ui:

    gr.Markdown(
        """
        # Vietnamese Fact-Check Chat
        *Một chatbot kiểm chứng thông tin Tiếng Việt*
        """
    )

    chat = gr.Chatbot(show_label=False)
    chat.value = init_chat()

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Nhập claim cần kiểm chứng...",
            show_label=False,
            lines=1,
            container=False,
            scale=8
        )

        # 🟢 Nút gửi = icon
        send_btn = gr.Button(
            value="",
            icon="assets/send.png",
            elem_id="send-btn",
            scale=1
        )

    clear = gr.Button("🧹 Xoá cuộc hội thoại")

    # Gửi bằng Enter
    msg.submit(chat_fn, [chat, msg], [chat, msg])

    # Gửi bằng nút icon
    send_btn.click(chat_fn, [chat, msg], [chat, msg])

    # Reset + giữ lời chào
    clear.click(fn=lambda: init_chat(), outputs=chat)

if __name__ == "__main__":
    ui.launch()