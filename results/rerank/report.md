# Avaliação de retrieval

Seleção no **dev** (média do MRR@10 clean/noisy); resultado final no **test**. `clean` = perguntas bem escritas; `noisy` = coloquial + typos + sem acento.

## 1. Top-10 configs elegíveis (dev, contexto ≤ 5000 chars)

| config | chunks | clean hit@5 | clean MRR@10 | noisy hit@5 | noisy MRR@10 | ctx chars@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| tfidf-char|structured-1024|hdr1|hyb1|ent1|rr-none | 358 | 0.969 | 0.885 | 0.945 | 0.795 | 4145 | 3.9 | 4.3 |
| tfidf-char|structured-1024|hdr1|hyb0|ent1|rr-mmarco | 358 | 1.000 | 0.877 | 0.961 | 0.793 | 4508 | 1291.0 | 1560.1 |
| tfidf-char|structured-1024|hdr1|hyb0|ent1|rr-none | 358 | 0.961 | 0.870 | 0.922 | 0.800 | 3975 | 3.3 | 3.5 |
| minilm-multi|structured-1024|hdr1|hyb1|ent1|rr-mmarco | 358 | 1.000 | 0.875 | 0.945 | 0.792 | 4463 | 1252.7 | 1415.4 |
| e5-small|structured-1024|hdr1|hyb1|ent1|rr-mmarco | 358 | 1.000 | 0.864 | 0.953 | 0.801 | 4516 | 1268.9 | 1442.8 |
| tfidf-char|structured-1024|hdr1|hyb1|ent1|rr-mmarco | 358 | 1.000 | 0.871 | 0.969 | 0.793 | 4495 | 1344.8 | 1464.7 |
| e5-small|structured-1024|hdr1|hyb0|ent1|rr-mmarco | 358 | 1.000 | 0.874 | 0.922 | 0.773 | 4506 | 1296.0 | 1483.9 |
| e5-small|structured-1024|hdr1|hyb1|ent1|rr-none | 358 | 1.000 | 0.866 | 0.898 | 0.757 | 4201 | 13.3 | 14.3 |
| tfidf-char|structured-512|hdr1|hyb0|ent1|rr-mmarco | 724 | 0.977 | 0.854 | 0.938 | 0.754 | 2238 | 637.0 | 862.5 |
| minilm-multi|structured-512|hdr1|hyb1|ent1|rr-mmarco | 724 | 0.969 | 0.873 | 0.906 | 0.729 | 2236 | 616.3 | 720.3 |

**Sem restrição de contexto**, o vencedor seria `tfidf-char|structured-1024|hdr1|hyb1|ent1|rr-none` (4145 chars): com chunks desse tamanho o top-5 já é quase o menu inteiro de 5 restaurantes. É recuperação de documento, não de passagem, e custa muito mais tokens.

### Fronteira de Pareto (melhor score de dev para cada nível de contexto)

| config | chunks | clean hit@5 | clean MRR@10 | noisy hit@5 | noisy MRR@10 | ctx chars@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| tfidf-char|structured-512|hdr1|hyb1|ent1|rr-none | 724 | 0.867 | 0.736 | 0.852 | 0.653 | 2149 | 4.6 | 4.8 |
| e5-small|structured-512|hdr1|hyb1|ent1|rr-none | 724 | 0.969 | 0.818 | 0.898 | 0.722 | 2172 | 13.2 | 13.8 |
| e5-small|structured-512|hdr1|hyb0|ent1|rr-none | 724 | 0.961 | 0.833 | 0.891 | 0.735 | 2185 | 12.7 | 13.5 |
| minilm-multi|structured-512|hdr1|hyb1|ent1|rr-mmarco | 724 | 0.969 | 0.873 | 0.906 | 0.729 | 2236 | 616.3 | 720.3 |
| tfidf-char|structured-512|hdr1|hyb0|ent1|rr-mmarco | 724 | 0.977 | 0.854 | 0.938 | 0.754 | 2238 | 637.0 | 862.5 |
| tfidf-char|structured-1024|hdr1|hyb0|ent1|rr-none | 358 | 0.961 | 0.870 | 0.922 | 0.800 | 3975 | 3.3 | 3.5 |
| tfidf-char|structured-1024|hdr1|hyb1|ent1|rr-none | 358 | 0.969 | 0.885 | 0.945 | 0.795 | 4145 | 3.9 | 4.3 |

## 2. Config escolhida no dev, avaliada no test: `tfidf-char|structured-1024|hdr1|hyb1|ent1|rr-none`

| variante | n | hit@1 | hit@5 [IC95%] | MRR@10 [IC95%] | recall@5 | hit@cand (teto reranker) |
|---|---|---|---|---|---|---|
| clean | 136 | 0.824 | 0.949 [0.904, 0.985] | 0.882 [0.835, 0.922] | 0.908 | 0.993 |
| noisy | 136 | 0.706 | 0.904 [0.853, 0.949] | 0.798 [0.743, 0.850] | 0.860 | 0.985 |

## 4. Efeito marginal de cada fator (média sobre as demais configs, dev, MRR@10)

| fator | nível | clean | noisy | n configs |
|---|---|---|---|---|
| embedder | e5-small | 0.851 | 0.748 | 8 |
| embedder | minilm-multi | 0.776 | 0.638 | 8 |
| embedder | tfidf-char | 0.835 | 0.747 | 8 |
| strategy | structured | 0.821 | 0.711 | 24 |
| chunk_size | 1024 | 0.842 | 0.747 | 12 |
| chunk_size | 512 | 0.799 | 0.676 | 12 |
| header | 1 | 0.821 | 0.711 | 24 |
| hybrid | 0 | 0.810 | 0.696 | 12 |
| hybrid | 1 | 0.831 | 0.726 | 12 |
| entity | 1 | 0.821 | 0.711 | 24 |
| reranker | mmarco | 0.864 | 0.752 | 12 |
| reranker | none | 0.778 | 0.671 | 12 |

## 5. Por tipo de pergunta (config escolhida, test)

| tipo | n | clean hit@5 | clean MRR | noisy hit@5 | noisy MRR |
|---|---|---|---|---|---|
| dietary | 25 | 1.000 | 0.933 | 0.920 | 0.858 |
| discovery | 30 | 0.833 | 0.735 | 0.800 | 0.694 |
| ingredients | 32 | 0.969 | 0.906 | 0.938 | 0.799 |
| price | 49 | 0.980 | 0.929 | 0.939 | 0.830 |

## Como ler

- `hit@cand`: algum candidato relevante entre os top-N do 1º estágio = teto do reranker.
- `ctx chars@5`: caracteres enviados ao LLM. Chunks maiores sobem hit@k quase de graça (o menu inteiro cabe num chunk); leia hit@k junto com este custo.
- 24 configs avaliadas e a melhor foi escolhida no dev: o dev é otimista por construção; o test é o número que vale.