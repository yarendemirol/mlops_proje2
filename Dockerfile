# Python imajını çek
FROM python:3.9-slim

# Çalışma dizinini ayarla
WORKDIR /app

# Gereksinimleri kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tüm kodları içeri kopyala
COPY . .

# API portu
EXPOSE 8000

# Uygulamayı başlat
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]