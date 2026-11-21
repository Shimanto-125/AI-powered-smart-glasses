# Deployment
## Hugging Face Spaces
1. Create a Space. Choose Streamlit SDK if available; otherwise Docker.
2. Upload app.py, config.py, src/, requirements.txt, packages.txt and trained models/best.pt.
3. Free CPU is suitable for a showcase demo only; inference may be slow. Train on Colab/local GPU.
4. Add large weights with Git LFS if needed. Never upload API secrets.
5. Free Spaces may sleep and have resource limits.

Dockerfile for a Docker Space:
```
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-ben libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["streamlit","run","app.py","--server.port=7860","--server.address=0.0.0.0"]
```
## Streamlit Community Cloud
Push to GitHub, connect repository, choose app.py. System Tesseract availability may vary; use Docker/HF for reliable OCR dependencies.
