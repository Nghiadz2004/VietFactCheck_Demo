# Sử dụng Python 3.10 trên nền Debian Bookworm (Bản ổn định, thay vì Slim mặc định đang là Trixie/Testing)
FROM python:3.10-slim-bookworm

# ==========================================
# 1. CÀI ĐẶT HỆ THỐNG (Java + Tiện ích)
# ==========================================
# Thay đổi: Dùng 'default-jdk-headless' thay vì 'openjdk-11...'
RUN apt-get update && \
    apt-get install -y \
    default-jdk-headless \
    wget \
    curl \
    unzip \
    git \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Thay đổi: JAVA_HOME trỏ về thư mục default-java (tự động link tới bản Java cài được)
ENV JAVA_HOME=/usr/lib/jvm/default-java

# ==========================================
# 2. CHUẨN BỊ MÔI TRƯỜNG PYTHON
# ==========================================
WORKDIR /app

# Copy file requirements trước để tận dụng Docker cache
COPY requirements.txt .

# Cài đặt các thư viện Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==========================================
# 3. CÀI ĐẶT RIÊNG CHO PLAYWRIGHT & VNCORENLP
# ==========================================

# A. Cài đặt trình duyệt cho Playwright (Bắt buộc)
RUN pip install playwright && \
    playwright install chromium --with-deps

# B. Tải trước model VnCoreNLP vào /app/vncorenlp
RUN mkdir -p /app/vncorenlp
RUN python -c "import py_vncorenlp; py_vncorenlp.download_model(save_dir='/app/vncorenlp')"

# ==========================================
# 4. THIẾT LẬP USER & QUYỀN (Cho Hugging Face)
# ==========================================
RUN useradd -m -u 1000 user
RUN chown -R user:user /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# ==========================================
# 5. CHẠY APP
# ==========================================
COPY --chown=user:user . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]