.PHONY: install data dataset smoke experiments experiments-rerank index serve e2e test

install:
	pip install -r requirements-dev.txt          # núcleo + testes (+ requirements-dense.txt p/ modelos densos)

data:
	python -m rag.generate_menus --n 150 --seed 42

dataset: data
	python -m evaluation.build_dataset --n 300 --seed 7

test:
	pytest -q

# ---- OFFLINE (só scikit-learn): tfidf, tfidf-char, lsa x chunking x header x híbrido ~4 min ----
smoke: dataset
	python -m evaluation.run_experiments --out results

# ---- COM MODELOS DENSOS (pip install -r requirements-dense.txt; baixa do HuggingFace) ----
experiments: dataset
	python -m evaluation.run_experiments \
	  --embedders tfidf tfidf-char lsa minilm-multi e5-small bge-m3 \
	  --chunk-sizes 512 1024 2048 --strategies fixed structured --out results/full

# reranker é caro (um cross-encoder por candidato): avalie só uma shortlist
experiments-rerank: dataset
	python -m evaluation.run_experiments \
	  --embedders tfidf-char minilm-multi e5-small bge-m3 --chunk-sizes 512 1024 \
	  --strategies structured --headers 1 --hybrids 0 1 \
	  --rerankers none mmarco bge-reranker --out results/rerank

index:
	python -m rag.build_index

serve:
	uvicorn rag.api:app --reload --port 8000

# geração + LLM-as-judge (precisa de ANTHROPIC_API_KEY; ~3 chamadas por pergunta)
e2e:
	python -m evaluation.run_e2e --n 100 --variant clean --out results/e2e_clean.json
	python -m evaluation.run_e2e --n 100 --variant noisy --out results/e2e_noisy.json
