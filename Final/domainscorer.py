import json
import os
import requests
import tldextract
from datetime import datetime

COMMONCRAWL_INDEXES = [
    ("2024", "CC-MAIN-2024-18-index"),
    ("2021", "CC-MAIN-2021-17-index"),
    ("2018", "CC-MAIN-2018-22-index"),
    ("2015", "CC-MAIN-2015-11-index"),
    ("2013", "CC-MAIN-2013-20-index"),
]

class DomainScorer:
    """
    Automatic domain trustworthiness evaluator.
    - Uses free signals only (GDELT, CommonCrawl, TLD, HTTPS, spam heuristics).
    - Caches domain results to avoid repeated computation (per ROOT DOMAIN).
    """

    def __init__(self, cache_file= "domain_cache.json", ttl_days=30):
        self.cache_file = cache_file
        self.ttl = ttl_days

        # Load cache if exists
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

        # Spam keyword list (expandable)
        self.spam_keywords = [
            "24h", "hot", "funny", "sex", "live", "phim", "tintuc",
            "blog", "truc-tiep", "clip", "game", "hotgirl",
            "giaitrivn", "docdao"
        ]

        # Good TLDs vs Suspicious TLDs
        self.tld_good = ["gov", "gov.vn", "edu", "edu.vn", "org"]
        self.tld_bad = ["xyz", "top", "icu", "click", "buzz"]

    # -----------------------------
    # 1) PUBLIC API
    # -----------------------------
    def get_domain_score(self, url: str):
        """
        Dùng URL để:
        - Check https
        - Extract root domain
        - Cache theo root domain
        """
        ext = tldextract.extract(url)
        root_domain = f"{ext.domain}.{ext.suffix}".lower()

        today = datetime.now().date()

        # Nếu domain đã cache → trả ra nhưng vẫn giữ https_score riêng
        if root_domain in self.cache:
            data = self.cache[root_domain]
            last = datetime.fromisoformat(data["last_updated"]).date()
            if (today - last).days < self.ttl:
                # BỔ SUNG: thêm https_score từ URL hiện tại
                data["reason"]["https"] = url.startswith("https")
                return data

        # Chưa có cache → tính mới
        score_data = self.compute_domain_score_auto(url, root_domain)

        # Cache theo root domain
        self.cache[root_domain] = score_data
        self._write_cache()

        return score_data

    # -----------------------------
    # 2) CORE LOGIC: AUTO SCORING
    # -----------------------------
    def compute_domain_score_auto(self, url: str, root_domain: str):
        ext = tldextract.extract(root_domain)

        # 1. TLD score
        tld = ext.suffix.lower()
        if tld in self.tld_good:
            tld_score = 1.0
        elif tld in self.tld_bad:
            tld_score = 0.3
        else:
            tld_score = 0.6

        # 2. HTTPS check (GIỜ CHÍNH XÁC)
        https_score = 1.0 if url.startswith("https") else 0.0

        # 3. Spam penalty
        spam_penalty = -0.6 if any(k in root_domain for k in self.spam_keywords) else 0

        # 4. CommonCrawl (giữ lại)
        cc_presence = self._check_commoncrawl(root_domain)
        
        # 5. Domain age score
        age_years = self.estimate_domain_age(root_domain)
        age_score = self.domain_age_score(age_years)

        score = (
            0.20 * cc_presence +
            0.20 * tld_score +
            0.20 * https_score +
            0.20 * age_score +
            0.20 * (1 + spam_penalty)
        )
        score = max(0, min(score, 1))

        return {
            "domain": root_domain,
            "score": round(score, 3),
            "reason": {
                "commoncrawl_presence": cc_presence,
                "tld_score": tld_score,
                "https": bool(https_score),
                "spam_penalty": spam_penalty
            },
            "last_updated": datetime.now().date().isoformat()
        }

    # -----------------------------
    # 3) SUPPORTING SIGNALS
    # -----------------------------
    def _check_commoncrawl(self, domain):
        """Return presence score using CommonCrawl (free)."""
        api = f"https://index.commoncrawl.org/CC-MAIN-2023-14-index?url={domain}&output=json"
        try:
            resp = requests.get(api, timeout=1.5)  # giảm timeout
            lines = resp.text.strip().split("\n")

            if len(lines) == 0:
                return 0.0

            # More appearances = more trustworthy
            return min(len(lines) / 50, 1.0)
        except Exception:
            return 0.0
        
    def estimate_domain_age(domain, timeout=1.0):
        for label, idx in COMMONCRAWL_INDEXES:
            url = f"https://index.commoncrawl.org/{idx}?url={domain}&matchType=domain&output=json"
            try:
                r = requests.get(url, timeout=timeout)
                if r.text.strip():
                    # xuất hiện lần đầu ở năm label
                    year = int(label)
                    return datetime.now().year - year
            except:
                continue

        return 0  # chưa từng xuất hiện → rất mới
    
    def domain_age_score(self, years):
        if years >= 10:
            return 1.0
        elif years >= 5:
            return 0.8
        elif years >= 2:
            return 0.5
        elif years >= 1:
            return 0.3
        else:
            return 0.0

    # -----------------------------
    # 4) SAVE CACHE
    # -----------------------------
    def _write_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception:
            # Không để lỗi ghi cache làm crash hệ thống
            pass