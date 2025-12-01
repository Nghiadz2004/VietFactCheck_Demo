import io
import re
import json
import asyncio
import aiohttp
import torch
import py_vncorenlp
import trafilatura
import os
import shutil
import logging
from sentence_transformers.cross_encoder import CrossEncoder
from pdfminer.high_level import extract_text as extract_pdf_text
from playwright.async_api import async_playwright
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from py_vncorenlp import VnCoreNLP

if "GLOBAL_VNCORENLP" not in globals():
    GLOBAL_VNCORENLP = None

# Cấu hình logging để dễ theo dõi tiến trình
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from playwright_stealth import stealth_async
    from playwright_stealth import Stealth
except ImportError:
    pass

class Module_4:
    def __init__(self):
        """
        Khởi tạo pipeline. 
        Tự động tải VnCoreNLP (kiểm tra kỹ file hệ thống), Bi-Encoder, Cross-Encoder, NLI.
        """
        # 1. Cấu hình thiết bị
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 Initializing Module_4 on device: {self.device}")

        # 2. Cấu hình & Tải VnCoreNLP (Word Segmenter)
        # Sử dụng đường dẫn tuyệt đối để tránh lỗi relative path
        self.vncorenlp_dir = os.path.join(os.getcwd(), "vncorenlp_models")
        
        # Kiểm tra file quan trọng: .jar và folder models. Nếu thiếu 1 trong 2 -> Tải lại
        jar_path = os.path.join(self.vncorenlp_dir, "VnCoreNLP-1.2.jar")
        models_path = os.path.join(self.vncorenlp_dir, "models")
        
        if not os.path.exists(jar_path) or not os.path.exists(models_path):
            logger.info("⚠️ VnCoreNLP models missing or incomplete. Cleaning and Downloading...")
            
            # Nếu thư mục tồn tại nhưng thiếu file, xóa đi để tải mới sạch sẽ
            if os.path.exists(self.vncorenlp_dir):
                shutil.rmtree(self.vncorenlp_dir)
            
            os.makedirs(self.vncorenlp_dir, exist_ok=True)
            py_vncorenlp.download_model(save_dir=self.vncorenlp_dir)
            logger.info("✅ VnCoreNLP Downloaded successfully.")
        else:
            logger.info("✅ VnCoreNLP models found.")

        logger.info("Loading VnCoreNLP segmenter...")
        # Load VnCoreNLP
        global GLOBAL_VNCORENLP

        logger.info("Loading VnCoreNLP segmenter...")
        
        if GLOBAL_VNCORENLP is None:
            try:
                GLOBAL_VNCORENLP = VnCoreNLP(
                    annotators=["wseg"],
                    save_dir=self.vncorenlp_dir
                )
                logger.info("VnCoreNLP loaded (new instance).")
            except Exception as e:
                logger.warning(f"VnCoreNLP init failed: {e}. Using fallback tokenizer.")
                GLOBAL_VNCORENLP = None
        
        self.rdrsegmenter = GLOBAL_VNCORENLP
        
        # fallback nếu GLOBAL_VNCORENLP không load được
        if self.rdrsegmenter is None:
            class _FallbackSegmenter:
                @staticmethod
                def word_segment(text: str):
                    text = re.sub(r'([.,!?;:()\"“”«»])', r' \1 ', text)
                    return [t for t in re.split(r"\s+", text) if t]
        
            self.rdrsegmenter = _FallbackSegmenter()

        # 3. Tải Bi-Encoder (Retrieval)
        bi_encoder_name = 'bkai-foundation-models/vietnamese-bi-encoder'
        logger.info(f"Loading Bi-Encoder: {bi_encoder_name}...")
        self.bi_encoder = SentenceTransformer(bi_encoder_name, device=self.device)

        # 4. Tải Cross-Encoder (Reranking)
        cross_encoder_name = 'itdainb/PhoRanker'
        logger.info(f"Loading Cross-Encoder: {cross_encoder_name}...")
        self.cross_encoder = CrossEncoder(cross_encoder_name, max_length=256, device=self.device)
        
        if self.device == "cuda":
            try:
                self.cross_encoder.model.half()
                logger.info("✅ Converted Cross-Encoder to FP16.")
            except Exception as e:
                logger.warning(f"Unable to convert CrossEncoder to FP16: {e}")

        # 5. Tải NLI Model (Stance Detection)
        nli_model_name = "vicgalle/xlm-roberta-large-xnli-anli"
        logger.info(f"Loading NLI Model: {nli_model_name}...")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        self.nli_model.to(self.device)
        
        logger.info("Module_4 Initialized Successfully!")

    # SECTION 1: UTILS (TEXT PROCESSING)
    
    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        # 1. Loại bỏ URL và Email
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # 2. Chuẩn hóa khoảng trắng đặc biệt (non-breaking space)
        text = text.replace('\xa0', ' ')
        
        # 3. QUAN TRỌNG: Giữ lại cấu trúc đoạn văn
        # Thay thế nhiều dòng trống liên tiếp thành 2 dòng (\n\n) để đánh dấu hết đoạn
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # 4. Xóa khoảng trắng thừa ở đầu/cuối mỗi dòng
        # (cờ re.MULTILINE giúp xử lý từng dòng một)
        text = re.sub(r'^[ \t]+|[ \t]+$', '', text, flags=re.MULTILINE)
        
        # 5. Thay thế khoảng trắng thừa trong dòng (tab, nhiều space) thành 1 space
        # Lưu ý: Không đưa \n vào trong ngoặc vuông [] để tránh xóa mất dấu xuống dòng
        text = re.sub(r'[ \t]{2,}', ' ', text)
        
        return text.strip()

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 512, min_remaining_ratio: float = 0.3) -> list[str]:
        """
        Chia văn bản thành các chunks với logic ngữ nghĩa.
        
        Args:
            text: Văn bản cần chia
            chunk_size: Kích thước tối đa của mỗi chunk (tính theo ký tự)
            min_remaining_ratio: Tỷ lệ tối thiểu của phần còn lại so với chunk_size
                                để quyết định có nên lấy hết đoạn văn hay không
        
        Returns:
            Danh sách các chunks
        """
        if not text:
            return []
        
        # Tách văn bản thành các đoạn (paragraphs)
        paragraphs = re.split(r'\n\n+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para_idx, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue
            
            # Tách đoạn thành các câu
            sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', para)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            for sent_idx, sentence in enumerate(sentences):
                sentence_len = len(sentence)
                
                # Tính toán độ dài phần còn lại của đoạn văn
                remaining_sentences = sentences[sent_idx + 1:]
                remaining_length = sum(len(s) for s in remaining_sentences)
                if remaining_sentences:
                    remaining_length += len(remaining_sentences)  # Khoảng trắng giữa các câu
                
                # Kiểm tra xem thêm câu này có vượt quá chunk_size không
                space_needed = 1 if current_chunk else 0  # Khoảng trắng trước câu
                total_if_added = current_length + space_needed + sentence_len
                
                if total_if_added > chunk_size and current_chunk:
                    # Chunk sẽ vượt quá kích thước
                    
                    # Kiểm tra: nếu phần còn lại quá ngắn, ưu tiên lấy hết đoạn
                    threshold = chunk_size * min_remaining_ratio
                    
                    if remaining_length > 0 and remaining_length < threshold:
                        # Phần còn lại quá ngắn -> lấy hết câu này vào chunk hiện tại
                        current_chunk.append(sentence)
                        current_length = total_if_added
                    else:
                        # Phần còn lại đủ dài -> đóng chunk hiện tại
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [sentence]
                        current_length = sentence_len
                else:
                    # Thêm câu vào chunk hiện tại
                    current_chunk.append(sentence)
                    current_length = total_if_added
            
            # Kết thúc đoạn văn
            # Kiểm tra: nếu đây không phải đoạn cuối và chunk hiện tại có nội dung
            if current_chunk and para_idx < len(paragraphs) - 1:
                # Kiểm tra đoạn kế tiếp
                next_para_idx = para_idx + 1
                while next_para_idx < len(paragraphs) and not paragraphs[next_para_idx].strip():
                    next_para_idx += 1
                
                if next_para_idx < len(paragraphs):
                    next_para = paragraphs[next_para_idx].strip()
                    next_sentences = re.split(r'(?<=[.!?])\s+', next_para)
                    next_sentences = [s.strip() for s in next_sentences if s.strip()]
                    
                    if next_sentences:
                        first_sentence_next = next_sentences[0]
                        first_sentence_len = len(first_sentence_next)
                        
                        # Nếu thêm câu đầu của đoạn kế tiếp vẫn không vượt chunk_size
                        if current_length + 1 + first_sentence_len <= chunk_size:
                            # Vẫn có thể tiếp tục, không đóng chunk
                            continue
                
                # Đóng chunk vì đoạn mới bắt đầu và không thể thêm vào
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
        
        # Thêm chunk cuối cùng nếu còn
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    # SECTION 2: CRAWLING & EXTRACTION (ASYNC)

    async def extract_text_from_pdf(self, url: str, session: aiohttp.ClientSession) -> str | None:
        logger.info(f"  -> Detect PDF link. Handle by pdfminer.six: {url}")
        try:
            async with session.get(url, timeout=60) as response:
                if response.status != 200:
                    logger.error(f"Failed to download PDF ({response.status})")
                    return None
                data = await response.read()
            
            # Chạy pdfminer trong executor để tránh chặn event loop
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, extract_pdf_text, io.BytesIO(data))
            
            logger.info("SUCCESS! Extracted text using pdfminer.six")
            return text.strip()
        except Exception as e:
            logger.error(f"Error when open file PDF {url}: {e}")
            return None

    async def extract_text_from_web(self, url: str, session: aiohttp.ClientSession) -> str | None:
        logger.info(f"  -> Detect web link. Handle by Trafilatura: {url}")
        try:
            async with session.get(url, timeout=30) as resp:
                if resp.status != 200:
                    raise ValueError(f"HTTP {resp.status}")
                html_content = await resp.text()

            text = trafilatura.extract(html_content)
            if text:
                logger.info("SUCCESS! Extracted text using Trafilatura")
                return text
            raise ValueError("Trafilatura extraction return None")

        except Exception as e:
            logger.warning(f"Error using Trafilatura for URL {url}: {e}")
            logger.info(f"Falling back to Playwright extraction ...")
            return await self._extract_with_playwright(url)

    async def _extract_with_playwright(self, url: str) -> str | None:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                try:
                    await page.goto(url, timeout=15000, wait_until="domcontentloaded")
                    html_content = await page.content()
                    
                    # Check Cloudflare
                    if "Cloudflare Ray ID" in html_content:
                        logger.warning("Detected Cloudflare! Retrying with Stealth...")
                        try:
                            async with Stealth().use_async(async_playwright()) as p2:
                                browser2 = await p2.chromium.launch(headless=True)
                                page2 = await browser2.new_page()
                                await page2.goto(url, timeout=20000)
                                html_content = await page2.content()
                                await browser2.close()
                        except NameError:
                             logger.error("Stealth module not found. Skipping stealth retry.")

                    await browser.close()
                    if not html_content: return None
                    
                    main_text = trafilatura.extract(html_content, include_comments=False)
                    logger.info(f"SUCCESS! Extracted text using Playwright")
                    return main_text
                
                except Exception as inner_e:
                    await browser.close()
                    raise inner_e
                
        except Exception as e1:
            logger.error(f"Error using Playwright extraction for URL {url}: {e1}")
            return None

    async def fetch_content_from_url(self, url: str, session: aiohttp.ClientSession) -> str | None:
      if url.lower().endswith('.pdf'):
        return await self.extract_text_from_pdf(url, session)
      else:
        return await self.extract_text_from_web(url, session)

    async def process_claims_crawling(self, retrieved_data: dict) -> dict:
      """
      Bước 1: Crawl dữ liệu từ các link đã retrieve được.
      """
      evidence_by_claim = {}
      claims = list(retrieved_data.keys())

      async with aiohttp.ClientSession() as session:
        for claim in claims:
          print(f"\n{'='*50}\nHandle claim: '{claim}'")
          documents = retrieved_data[claim]
          all_chunks_for_this_claim = []
          
          urls = [doc['link'] for doc in documents]
          print(f"  -> Crawling {len(urls)} links in parallel...")
          
          tasks = [self.fetch_content_from_url(u, session) for u in urls]
          full_contents = await asyncio.gather(*tasks)

          for doc, full_content in zip(documents, full_contents):
            content_to_process = ""
            if full_content and len(full_content) > 100:
              cleaned = self.clean_text(full_content)
              content_to_process = f"{doc.get('title', '')}. {cleaned}"
            else:
              print(f"FAIL/EMPTY!! Using snippet for {doc['link']}")
              cleaned = self.clean_text(doc.get('snippet', ''))
              content_to_process = f"{doc.get('title', '')}. {cleaned}"

            chunks = self.chunk_text(content_to_process)
            for chunk_part in chunks:
              all_chunks_for_this_claim.append({
                  "text": chunk_part,
                  "link": doc['link']
              })

          evidence_by_claim[claim] = all_chunks_for_this_claim
          print(f"==> Finish for claim '{claim}'. Total: {len(all_chunks_for_this_claim)} chunks.")
  
      return evidence_by_claim

    # SECTION 3: RETRIEVAL & RANKING (BI-ENCODER)

    def process_bi_encoder(self, evidence_by_claim: dict) -> dict:
        retrieved_candidates_by_claim = {}

        for claim, chunks_for_claim in evidence_by_claim.items():
            print(f"\n--- Finding top-k chunk related to claim: '{claim}' ---")
            if not chunks_for_claim:
                print("     No evidence chunk for this claim")
                retrieved_candidates_by_claim[claim] = []
                continue

            claim_embedding = self.bi_encoder.encode(claim, convert_to_tensor=True)
            chunk_texts = [chunk['text'] for chunk in chunks_for_claim]
            chunk_embeddings = self.bi_encoder.encode(chunk_texts, convert_to_tensor=True)

            cosine_scores = util.cos_sim(claim_embedding, chunk_embeddings)[0]

            top_k = 20
            actual_top_k = min(top_k, len(chunks_for_claim))
            top_results_indices = cosine_scores.argsort(descending=True)[:actual_top_k]

            top_chunks = []
            print(f"    Top {actual_top_k} chunk best related for claim '{claim}':")
            for idx in top_results_indices:
                candidate = chunks_for_claim[idx]
                top_chunks.append(candidate)

            retrieved_candidates_by_claim[claim] = top_chunks

        return retrieved_candidates_by_claim

    # SECTION 4: RERANKING (CROSS-ENCODER)

    def process_cross_encoder(self, retrieved_candidates_by_claim: dict) -> dict:
        final_evidence_by_claim = {}

        for claim, candidates in retrieved_candidates_by_claim.items():
            print(f"\n--- Reranking evidences for claim: '{claim}' ---")
            if not candidates:
                final_evidence_by_claim[claim] = []
                continue

            candidate_texts = [candidate['text'] for candidate in candidates]

            # Tokenize bằng RdrSegmenter (VnCoreNLP)
            tokenized_claim = " ".join(self.rdrsegmenter.word_segment(claim))
            tokenized_candidates = [" ".join(self.rdrsegmenter.word_segment(text)) for text in candidate_texts]
            tokenized_pairs = [[tokenized_claim, sent] for sent in tokenized_candidates]

            cross_scores = self.cross_encoder.predict(tokenized_pairs)

            reranked_results = list(zip(cross_scores, candidates))
            reranked_results.sort(key=lambda x: x[0], reverse=True)

            final_top_k = 10
            actual_final_top_k = min(final_top_k, len(candidates))
            final_evidence = reranked_results[:actual_final_top_k]

            final_evidence_by_claim[claim] = final_evidence
            print(f"    Top {actual_final_top_k} evidences after reranking done.")

        return final_evidence_by_claim

    # SECTION 5: STANCE DETECTION (NLI)

    def classify_stance(self, claim: str, evidence_text: str) -> dict:
        tokenized_input = self.nli_tokenizer(
            evidence_text, claim, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.nli_model(**tokenized_input)

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
        
        labels = ['Refute', 'Neutral', 'Support']
        predicted_index = torch.argmax(logits, dim=1).item()

        return {
            "stance": labels[predicted_index],
            "score": probabilities[predicted_index],
            "scores_all": {
                "refute": probabilities[0],
                "neutral": probabilities[1],
                "support": probabilities[2]
            }
        }

    def process_stance(self, final_evidence_by_claim: dict) -> dict:
        results_with_stance = {}
        for claim, top_evidence in final_evidence_by_claim.items():
            evidence_with_stance = []
            print(f"\n--- Phân loại lập trường cho claim: '{claim}' ---")

            for score, chunk in top_evidence:
                stance_result = self.classify_stance(claim, chunk['text'])
                evidence_with_stance.append({
                    'text': chunk['text'],
                    'link': chunk['link'],
                    'rerank_score': float(score),
                    'stance': stance_result['stance'],
                    'stance_score': stance_result['score'],
                    'stance_scores': stance_result['scores_all']
                })
                print(f"  -> {stance_result['stance']} | Text: {chunk['text'][:50]}...")

            results_with_stance[claim] = evidence_with_stance
        return results_with_stance

    # MAIN ENTRY POINT

    async def run(self, retrieved_data: dict) -> dict:
        """
        Hàm chạy toàn bộ quy trình từ dữ liệu đầu vào (claims + links)
        đến kết quả cuối cùng có nhãn stance.
        """
        # 1. Crawl và Chunking
        evidence_by_claim = await self.process_claims_crawling(retrieved_data)
        
        # 2. Retrieve (Bi-Encoder)
        candidates = self.process_bi_encoder(evidence_by_claim)
        
        # 3. Rerank (Cross-Encoder)
        reranked = self.process_cross_encoder(candidates)
        
        # 4. Stance Detection
        final_results = self.process_stance(reranked)
        
        return final_results