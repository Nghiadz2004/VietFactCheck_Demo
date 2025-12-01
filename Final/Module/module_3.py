import json
import os
import re
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import tldextract
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
import math
from domainscorer import DomainScorer

class Module3:
    """
    Module 3: Document Retrieval

    - Nhận vào một claim
    - Sinh query mở rộng bằng expand_query()
    - Gọi SERPER
    - Tính trust_score + semantic_re_rank
    - Trả về top nguồn tin tốt nhất
    """

    def __init__(
        self,
        embed_model: str = "keepitreal/vietnamese-sbert",
        openai_api_key: str = None,
        serper_api_key: str = None,
        trusted_domains_file: str = "trusted_domains.json",
        bad_domains_file: str = "bad_domains.json"
    ):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        trusted_path = os.path.join(current_dir, trusted_domains_file)
        bad_path = os.path.join(current_dir, bad_domains_file)
        """
        Khởi tạo Module 3.

        Args:
            embed_model (str): Tên model Sentence-BERT để embed.
            serper_api_key (str): API key Serper (Google Search API).
            trusted_domains (List[str]): Danh sách domain uy tín.
            bad_domains (List[str]): Domain rác / độ tin cậy thấp.
        """
        self.embedder = SentenceTransformer(embed_model)
        self.domain_scorer = DomainScorer()
        
        self.openai_api_key = openai_api_key
        self.SERPER_API_KEY = serper_api_key

        # Load trusted domains
        if os.path.exists(trusted_path):
            with open(trusted_path, "r", encoding="utf-8") as f:
                self.TRUSTED_DOMAINS = json.load(f)
        else:
            raise FileNotFoundError(f"Không tìm thấy file {trusted_domains_file}")

        # Load bad domains
        if os.path.exists(bad_path):
            with open(bad_path, "r", encoding="utf-8") as f:
                self.BAD_DOMAINS = json.load(f)
        else:
            raise FileNotFoundError(f"Không tìm thấy file {bad_domains_file}")

        # Load NER
        try:
            from underthesea import ner as underthesea_ner
            self.underthesea_ner = underthesea_ner
            self.have_under = True
        except:
            self.underthesea_ner = None
            self.have_under = False
        
    # ====================================
    #       EXPAND QUERY
    # ====================================
    def expand_query(self, claim: str, temperature: float = 0.4, retries: int = 1) -> List[str]:
        """
        Sinh 3 truy vấn mở rộng:
        - 2 paraphrase ngắn (semantic → search)
        - 1 headline nhẹ nhàng (không clickbait)
        - Giữ nguyên placeholders [ENT_i] đã mask.

        Sau đó:
        - Trả về 5 phần tử:
            [ claim_goc,
              claim_with_quotes,
              paraphrase1_with_quotes,
              paraphrase2_with_quotes,
              headline_with_quotes ]

        Trong đó mọi tên riêng sẽ được đặt trong cặp dấu nháy kép.
        """
        from openai import OpenAI
        client = OpenAI(api_key=self.openai_api_key)

        claim = (claim or "").strip()
        if not claim:
            return []

        # ----------------------------------------
        # 1) NER + mask entity
        # ----------------------------------------
        try:
            entities = self.extract_entities_vn(claim)
        except Exception:
            entities = []

        if entities:
            masked_claim, mapping = self.mask_entities_from_list(claim, entities)
            if not mapping:  # fallback nếu không match được trong text
                masked_claim, mapping = self.mask_entities(claim)
        else:
            masked_claim, mapping = self.mask_entities(claim)

        # Tạo phiên bản claim có tên riêng trong dấu nháy kép
        claim_with_quotes = masked_claim
        for ph, orig in mapping.items():
            claim_with_quotes = claim_with_quotes.replace(ph, f"\"{orig}\"")
        claim_with_quotes = self._clean_text(claim_with_quotes)

        # ----------------------------------------
        # 2) Prompt GPT: sinh 2 paraphrase + 1 headline
        # ----------------------------------------
        few_shot = (
            "Ví dụ (có dùng placeholder [ENT_i]):\n"
            "Input: \"[ENT_0] giành giải thưởng về giáo dục\"\n"
            "Output: [\"[ENT_0] thắng giải giáo dục\", "
            "\"[ENT_0] được trao giải giáo dục\", "
            "\"[ENT_0] gây chú ý vì chiến thắng giải thưởng giáo dục?\"]\n\n"
        )

        instruction = (
            "Sinh DUY NHẤT 1 JSON array gồm 3 chuỗi:\n"
            "- 2 câu đầu: paraphrase ngắn, cùng nghĩa.\n"
            "- Câu thứ 3: dạng tiêu đề báo, tự nhiên, không giật tít.\n"
            "- Giữ nguyên placeholder [ENT_i].\n"
            f"Input: \"{masked_claim}\""
        )

        system_prompt = (
            "Bạn là hệ thống sinh truy vấn tìm kiếm. "
            "Hãy CHỈ TRẢ VỀ JSON array 3 phần tử, không thêm giải thích."
        )

        def call_gpt(prompt_text: str) -> str:
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_text}
                    ]
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print("⚠️ expand_query GPT error:", e)
                return ""

        def parse_json_array(text: str):
            """Thử parse JSON array → nếu fail, dùng regex tìm mảng."""
            if not text:
                return None
            try:
                arr = json.loads(text)
                if isinstance(arr, list):
                    return arr
            except json.JSONDecodeError:
                pass

            m = re.search(r"\[.*\]", text, re.S)
            if m:
                try:
                    arr = json.loads(m.group(0))
                    if isinstance(arr, list):
                        return arr
                except json.JSONDecodeError:
                    pass
            return None

        # ----------------------------------------
        # 3) Gọi GPT + Retry
        # ----------------------------------------
        prompt_base = few_shot + instruction
        raw = ""
        parsed = None

        for _ in range(retries + 1):
            raw = call_gpt(prompt_base)
            parsed = parse_json_array(raw)
            if parsed and len(parsed) >= 3:
                break
            prompt_base = (
                "CHỈ TRẢ VỀ JSON ARRAY 3 PHẦN TỬ. Không thêm chữ khác.\n" +
                instruction
            )

        if not parsed or len(parsed) < 3:
            # fallback: dùng lại masked_claim nếu GPT hỏng format
            parsed = [masked_claim, masked_claim, masked_claim]
        parsed = parsed[:3]

        # ----------------------------------------
        # 4) Unmask + thêm dấu nháy kép cho tên riêng
        # ----------------------------------------
        def unmask_with_quotes(s: str) -> str:
            for ph, orig in mapping.items():
                s = s.replace(ph, f"\"{orig}\"")
            return self._clean_text(s)

        paraphrase1 = unmask_with_quotes(parsed[0])
        paraphrase2 = unmask_with_quotes(parsed[1])
        headline = unmask_with_quotes(parsed[2])

        # ----------------------------------------
        # 5) Ghép list 5 phần:
        #    [claim_goc, claim_quoted, para1, para2, headline]
        # ----------------------------------------
        return [claim, claim_with_quotes, paraphrase1, paraphrase2, headline]

    # ====================================
    #             Utilities
    # ====================================
    def _clean_text(self, text: str) -> str:
        """
        Làm sạch text đơn giản: xoá </s>, chuẩn hoá khoảng trắng.

        Args:
            text (str): Văn bản input.

        Returns:
            str: Văn bản đã được làm sạch.
        """
        t = text.strip()
        t = re.sub(r"</s>", "", t)
        t = re.sub(r"\s+", " ", t)
        return t.strip()
    
    def extract_entities_vn(self, text: str):
        """
        Trích xuất 4 loại entity:
        - PERSON
        - LOCATION
        - ORG
        - NUMBER (bao gồm số liệu, %, tiền tệ, date)
        """
        # ================================
        # 1) UnderTheSea NER
        # ================================
        if self.have_under and self.underthesea_ner:
            try:
                raw = self.underthesea_ner(text)  # (tok, pos, _, tag)
                buf = []
                cur_tag = None
                out = []

                for tok, pos, _, tag in raw:
                    if tag != "O":
                        if cur_tag == tag:
                            buf.append(tok)
                        else:
                            if buf:
                                out.append((" ".join(buf), cur_tag))
                            buf = [tok]
                            cur_tag = tag
                    else:
                        if buf:
                            out.append((" ".join(buf), cur_tag))
                            buf = []
                            cur_tag = None

                if buf:
                    out.append((" ".join(buf), cur_tag))

                mapped = []
                for span, tag in out:
                    if "PER" in tag:
                        mapped.append((span, "PERSON"))
                    elif "LOC" in tag:
                        mapped.append((span, "LOCATION"))
                    elif "ORG" in tag:
                        mapped.append((span, "ORG"))

                if mapped:
                    return mapped  # Ưu tiên UnderTheSea
            except:
                pass

        # ================================
        # 2) FALLBACK (Nếu Underthesea không nhận ra)
        # ================================
        fallback = []

        # --- SỐ LIỆU: 100, 3.5, 5.000, 5 tỷ, 20%, 2025, 12/3/2024 ---
        number_patterns = [
            r"\b\d[\d\.,]*\b",
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            r"\b\d+(\.\d+)?\s*(%|tỷ|triệu|nghìn|k|đ|usd|eur)\b"
        ]
        for pat in number_patterns:
            for m in re.findall(pat, text, flags=re.IGNORECASE):
                span = m if isinstance(m, str) else m[0]
                fallback.append((span, "NUMBER"))

        # --- TÊN NGƯỜI 2–4 từ viết hoa ---
        person_pat = r"\b[A-ZÀ-Ỹ][a-zà-ỹ']+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ']+){1,3}\b"
        for m in re.findall(person_pat, text):
            fallback.append((m, "PERSON"))

        # (ORG / LOC để regex cũng dễ bị sai → chỉ rely Underthesea)
        # nhưng vẫn quét các tổ chức kiểu 2–5 từ viết hoa
        org_loc_pat = r"\b[A-ZÀ-Ỹ][A-Za-zÀ-ỹ']+(?:\s+[A-ZÀ-Ỹ][A-Za-zÀ-ỹ']+){1,4}\b"
        for m in re.findall(org_loc_pat, text):
            if len(m.split()) >= 2:  # tránh mask từ đầu câu
                fallback.append((m, "ORG"))

        return fallback


    # ---- Helper: mask entities from an entity list (preserve mapping order) ----
    def mask_entities_from_list(self, text: str, entities):
        """
        Mask entity theo thứ tự xuất hiện trong entities list.
        """
        masked = text
        mapping = {}
        idx = 0

        import re
        for ent, _ in entities:
            ent = ent.strip()
            ph = f"[ENT_{idx}]"
            new_masked, n = re.subn(re.escape(ent), ph, masked, count=1)
            if n > 0:
                mapping[ph] = ent
                masked = new_masked
                idx += 1

        return masked, mapping


    # ---- Helper: mask entities from raw text using underthesea / heuristics ----
    def mask_entities(self, text: str):
        """
        Mask 4 loại entity:
        - PERSON
        - LOCATION
        - ORG
        - NUMBER
        """
        masked = text
        mapping = {}
        idx = 0

        entities = self.extract_entities_vn(text)

        for ent, ent_type in entities:

            # tìm đúng vị trí match
            for m in re.finditer(re.escape(ent), text):
                ph = f"[ENT_{idx}]"
                masked = masked.replace(ent, ph, 1)
                mapping[ph] = ent
                idx += 1
                break  # mỗi entity chỉ mask 1 lần

        return masked, mapping

    # ====================================
    #        VIDEO URL FILTER
    # ====================================
    def is_video_url(self, url: str) -> bool:
        """
        Kiểm tra URL có phải là trang video / mạng xã hội video hay không.
        Dùng để loại khỏi kết quả retrieval.
        """
        if not url or not isinstance(url, str):
            return False

        u = url.lower()

        # ==========================================
        # 1) Domain video / mạng xã hội
        # ==========================================
        video_domains = {
            "tiktok.com", "youtube.com", "youtu.be",
            "facebook.com", "fb.watch", "instagram.com",
            "dailymotion.com", "vimeo.com", "bilibili.com",
            "twitter.com", "x.com",
            "tv.zing.vn", "zingmp3.vn"
        }

        domain = tldextract.extract(u).top_domain_under_public_suffix
        if domain in video_domains:
            return True

        # ==========================================
        # 2) Đường dẫn dạng video
        # ==========================================
        video_patterns = [
            "watch?", "watch/", "shorts/",
            "/video", "/videos/", "/live/", "/livestream",
            "clip", "/tv/", "/media/", "/embed/",
            "reels/", "reel/", "stories/"
        ]
        if any(p in u for p in video_patterns):
            return True

        # ==========================================
        # 3) File video
        # ==========================================
        video_extensions = (
            ".mp4", ".mov", ".avi", ".webm", ".mkv", ".m3u8", ".flv"
        )
        if u.endswith(video_extensions):
            return True

        return False

    # ====================================
    #       SERPER SEARCH
    # ====================================
    def serper_search(self, query: str, num_results: int = 8) -> list[dict]:
        """
        Gửi query tới Serper API để lấy kết quả Google Search.

        Args:
            query (str): Chuỗi truy vấn tìm kiếm.
            num_results (int): Số kết quả cần lấy.

        Returns:
            list[dict]: Danh sách kết quả gồm {title, link, snippet}.
        """
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {"q": query, "num": num_results}

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
                
            # --- LỌC VIDEO / SOCIAL ---
            cleaned = []
            for item in results:
                link = item.get("link", "")
                if not self.is_video_url(link):
                    cleaned.append(item)

            return cleaned

        except Exception as e:
            print(f"[Serper error] Query='{query[:40]}' ERR={e}")
            return []

    # ====================================
    #              TRUST SCORE
    # ====================================
    def compute_recency_score(self, pub_date_str: str, lam: float = 0.002) -> float:
        """
        Tính điểm thời gian bằng hàm decay mũ.

        Args:
            pub_date_str (str): Chuỗi ngày đăng (ISO format).
            lam (float): Hệ số suy giảm.

        Returns:
            float: Điểm recency nằm [0, 1].
        """
        try:
            pub_date = datetime.fromisoformat(pub_date_str)
            days = (datetime.now() - pub_date).days
            return max(0, min(math.exp(-lam * days), 1))
        except:
            return 0.5

    def compute_trust_score(self, source: dict) -> float:
        link = source.get("link", "")
        snippet = source.get("snippet", "")

        # domain-based score (đã cache theo root domain)
        domain_info = self.domain_scorer.get_domain_score(link)
        domain_score = domain_info["score"]

        # recency
        recency = self.compute_recency_score(source.get("date", ""))

        # vietnamese language check
        vi_chars = re.findall(
            r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩ'
            r'òóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữ'
            r'ỳýỵỷỹđ]', snippet.lower()
        )
        lang_bonus = 0.1 if len(vi_chars) > 5 else -0.05

        score = 0.7 * domain_score + 0.2 * recency + 0.1 * lang_bonus
        score = max(0, min(score, 1))
        score = round(score, 3)

        # ghi trust_reason
        source["trust_reason"] = {
            "domain": domain_info,
            "recency": recency,
            "lang_bonus": lang_bonus
        }

        return score

    # ====================================
    #         SEMANTIC RE-RANK
    # ====================================
    def rerank(self, claim: str, sources: list[dict]) -> list[dict]:
        """
        Re-rank theo:
        1. Semantic similarity (batch encode)
        2. Chỉ tính trust_score cho top K kết quả theo semantic (lazy scoring)
        3. Ưu tiên domain uy tín
        """
        if not sources:
            return []

        # 0. Pre-filter: loại BAD_DOMAINS để giảm noise
        filtered_sources = []
        for s in sources:
            link = s.get("link", "")
            if not link or not link.startswith("http"):
                continue
            domain = tldextract.extract(link).top_domain_under_public_suffix
            if domain in self.BAD_DOMAINS:
                continue
            filtered_sources.append(s)

        if not filtered_sources:
            filtered_sources = sources  # fallback nếu lọc hết

        # 1. Batch embedding: encode [claim] + tất cả nguồn
        texts = [f"{s.get('title','')}. {s.get('snippet','')}" for s in filtered_sources]
        all_embeddings = self.embedder.encode(
            [claim] + texts,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        emb_claim = all_embeddings[0].unsqueeze(0)
        emb_src = all_embeddings[1:]

        sims = util.cos_sim(emb_claim, emb_src)[0]

        # 2. Gán semantic_score & lọc theo threshold
        semantic_threshold = 0.45
        for i, src in enumerate(filtered_sources):
            semantic = float(sims[i])
            src["semantic_score"] = semantic

        semantic_filtered = [s for s in filtered_sources if s["semantic_score"] >= semantic_threshold]

        # Nếu không có gì qua ngưỡng, dùng tất cả nhưng vẫn ưu tiên semantic cao
        candidate_list = semantic_filtered if semantic_filtered else filtered_sources

        # 3. Sắp xếp theo semantic score giảm dần
        candidate_list = sorted(
            candidate_list,
            key=lambda x: x.get("semantic_score", 0.0),
            reverse=True
        )

        # 4. Lazy scoring: chỉ tính trust_score cho top K nguồn
        MAX_CANDIDATES = 15
        candidate_list = candidate_list[:MAX_CANDIDATES]

        for s in candidate_list:
            s["trust_score"] = self.compute_trust_score(s)

        # 5. Ưu tiên trusted domains
        trusted = []
        neutral = []

        for s in candidate_list:
            domain = tldextract.extract(s.get("link", "")).top_domain_under_public_suffix
            if domain in self.TRUSTED_DOMAINS:
                trusted.append(s)
            else:
                neutral.append(s)

        trusted_sorted = sorted(trusted, key=lambda x: x["trust_score"], reverse=True)
        neutral_sorted = sorted(neutral, key=lambda x: x["trust_score"], reverse=True)

        # 6. Lấy top 10 cuối
        return (trusted_sorted[:5] + neutral_sorted[:5])[:10]

    # ====================================
    #         PROCESS 1 CLAIM
    # ====================================
    def process_claim(self, claim: str) -> list[dict]:
        """
        Xử lý một claim duy nhất: expand → search → filter → score → rerank.

        Args:
            claim (str): Câu claim cần kiểm tra.

        Returns:
            list[dict]: Danh sách top nguồn tin liên quan.
        """
        queries = self.expand_query(claim)
        for i in range(len(queries)):
            print(f"  [Query {i+1}] {queries[i]}")
        print("-"*80)

        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self.serper_search, q): q for q in queries}
            for f in as_completed(futures):
                results.extend(f.result())
                time.sleep(0.1)

        # Loại trùng link
        seen = set()
        filtered = []
        for r in results:
            link = r.get("link")
            if link and link.startswith("http") and link not in seen:
                seen.add(link)
                filtered.append(r)

        return self.rerank(claim, filtered)

    # ====================================
    #         MAIN PIPELINE (N claims)
    # ====================================
    def run(self, claims: List[str], output: str = "document_retrieval_results.json"):
        """
        Chạy pipeline cho nhiều claim liên tiếp.

        Args:
            claims (List[str]): Danh sách claim.
            output (str): File JSON output.

        Returns:
            list[dict]: Kết quả toàn bộ pipeline.
        """
        out = []
        for c in claims:
            if not c.strip():
                continue
            sources = self.process_claim(c)
            out.append({
                "claim": c,
                "sources": sources
            })

        with open(output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        print(f"\n✔ Kết quả được lưu tại: {output}")
        return out
    
# # Cách dùng module 3 để test
# !pip install tldextract --quiet
# from module_3 import Module3
# from dotenv import load_dotenv
# import os

# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# # Khởi tạo Module 3
# m3 = Module3(
#     openai_api_key=OPENAI_API_KEY,
#     serper_api_key=SERPER_API_KEY,
# )

# # Claim test
# claim = "Richard Nixon là tổng thống Mỹ duy nhất từ chức."

# print(f"CLAIM: {claim}\n")

# print(">>> Đang chạy pipeline Document Retrieval...\n")
# results = m3.process_claim(claim)

# print("\n============================")
# print("       KẾT QUẢ TOP 10")
# print("============================\n")

# for i, src in enumerate(results, start=1):
#     print(f"{i}. {src['title']}")
#     print(f"   Link: {src['link']}")
#     print(f"   Trust score: {src.get('trust_score')}")
#     print(f"   Semantic score: {src.get('semantic_score')}")
#     print(f"   Snippet: {src.get('snippet')}")
#     print(f"   Trust reason: {src.get('trust_reason')}\n")