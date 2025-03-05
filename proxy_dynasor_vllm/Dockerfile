FROM python:3.12-slim

# Cài đặt các gói hệ thống cần thiết (git, build-essential, curl, jq)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    jq \
 && rm -rf /var/lib/apt/lists/*

# Clone repository Dynasor và cài đặt nó qua pip
RUN git clone https://github.com/hao-ai-lab/Dynasor.git && \
    cd Dynasor && pip install . && cd -

# Copy các file cần thiết vào image
COPY entrypoint.sh /entrypoint.sh
COPY patch_dynasor.py /patch_dynasor.py
COPY start_dynasor.py /start_dynasor.py
RUN chmod +x /entrypoint.sh

# Cài đặt thêm uvicorn và fastapi nếu cần
RUN pip install uvicorn fastapi 

# Expose cổng nếu Dynasor cần (ví dụ: 8080)
EXPOSE 8080

# Sử dụng entrypoint script thay vì CMD cố định
ENTRYPOINT ["/entrypoint.sh"]
