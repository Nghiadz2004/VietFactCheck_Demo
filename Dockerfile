# Sử dụng Python 3.9
FROM python:3.9-slim

# 1. Cài đặt Java (Bắt buộc cho VnCoreNLP) và các công cụ cần thiết
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk-headless wget unzip && \
    apt-get clean;

# Thiết lập biến môi trường JAVA_HOME (để code của bạn tìm thấy Java)
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# 2. Thiết lập thư mục làm việc
WORKDIR /app

# 3. Copy file requirements và cài đặt thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Tải trước model VnCoreNLP (Bước quan trọng!)
# Chúng ta tạo một thư mục vncorenlp và tải model về đó để lúc chạy app không phải tải lại
RUN mkdir -p /app/vncorenlp
RUN python -c "import py_vncorenlp; py_vncorenlp.download_model(save_dir='/app/vncorenlp')"

# 5. Copy toàn bộ code vào
COPY . .

# 6. Tạo user non-root để chạy an toàn trên Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 7. Mở port cho Streamlit
EXPOSE 8501

# 8. Chạy App
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]