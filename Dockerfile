# PriceMatch demo container: UI + full rule-agent stack, fully offline.
# ML extras (neural retrieval, FLAN-T5, CLIP photo search) are optional;
# add them with: pip install -r requirements-ml.txt inside the image.
#
#   docker build -t pricematch .
#   docker run -p 8501:8501 pricematch
# then open http://localhost:8501
FROM python:3.11-slim

WORKDIR /app

# install deps first for layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# the demo self-seeds its SQLite catalog on first run (no network needed)
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "pricematch/demo/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
