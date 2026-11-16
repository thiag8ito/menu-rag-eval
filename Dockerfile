FROM python:3.12-slim
WORKDIR /app
# EMBEDDER=tfidf-char roda sem baixar modelos; para densos: --build-arg EMBEDDER=minilm-multi --build-arg DENSE=1
ARG EMBEDDER=tfidf-char
ARG DENSE=0
ENV EMBEDDER=$EMBEDDER CHUNK_SIZE=1024 CHUNK_STRATEGY=structured CONTEXTUAL_HEADER=1 HYBRID=1 ENTITY_LINK=1
COPY requirements.txt requirements-dense.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ "$DENSE" = "1" ]; then pip install --no-cache-dir -r requirements-dense.txt; fi
COPY rag ./rag
COPY evaluation ./evaluation
RUN python -m rag.generate_menus --n 150 --seed 42 && python -m rag.build_index
EXPOSE 8000
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
