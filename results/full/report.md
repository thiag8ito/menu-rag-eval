# Avaliação de retrieval

Seleção no **dev** (média do MRR@10 clean/noisy); resultado final no **test**. `clean` = perguntas bem escritas; `noisy` = coloquial + typos + sem acento.

## 1. Top-10 configs elegíveis (dev, contexto ≤ 5000 chars)

| config | chunks | clean hit@5 | clean MRR@10 | noisy hit@5 | noisy MRR@10 | ctx chars@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| tfidf-char|structured-1024|hdr1|hyb1|ent1|rr-none | 358 | 0.969 | 0.885 | 0.945 | 0.795 | 4145 | 6.5 | 10.0 |
| bge-m3|structured-1024|hdr1|hyb1|ent1|rr-none | 358 | 0.977 | 0.879 | 0.938 | 0.793 | 4260 | 258.7 | 354.4 |
| tfidf-char|structured-1024|hdr1|hyb0|ent1|rr-none | 358 | 0.961 | 0.870 | 0.922 | 0.800 | 3975 | 4.6 | 5.1 |
| bge-m3|fixed-1024|hdr1|hyb1|ent1|rr-none | 362 | 0.977 | 0.865 | 0.945 | 0.787 | 4385 | 380.1 | 785.0 |
| tfidf-char|fixed-1024|hdr1|hyb1|ent1|rr-none | 362 | 0.977 | 0.847 | 0.945 | 0.779 | 4298 | 5.0 | 5.4 |
| e5-small|structured-1024|hdr1|hyb1|ent1|rr-none | 358 | 1.000 | 0.866 | 0.898 | 0.757 | 4201 | 27.9 | 29.9 |
| tfidf-char|fixed-1024|hdr1|hyb0|ent1|rr-none | 362 | 0.969 | 0.839 | 0.938 | 0.784 | 4129 | 5.1 | 5.6 |
| bge-m3|fixed-1024|hdr1|hyb0|ent1|rr-none | 362 | 0.977 | 0.859 | 0.891 | 0.760 | 4384 | 275.9 | 425.4 |
| bge-m3|structured-1024|hdr1|hyb0|ent1|rr-none | 358 | 0.969 | 0.861 | 0.906 | 0.748 | 4291 | 261.9 | 445.2 |
| tfidf-char|structured-1024|hdr0|hyb0|ent1|rr-none | 358 | 0.922 | 0.822 | 0.875 | 0.765 | 4196 | 4.8 | 5.6 |

**Sem restrição de contexto**, o vencedor seria `tfidf-char|structured-2048|hdr0|hyb0|ent1|rr-none` (8223 chars): com chunks desse tamanho o top-5 já é quase o menu inteiro de 5 restaurantes. É recuperação de documento, não de passagem, e custa muito mais tokens.

### Fronteira de Pareto (melhor score de dev para cada nível de contexto)

| config | chunks | clean hit@5 | clean MRR@10 | noisy hit@5 | noisy MRR@10 | ctx chars@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| tfidf|structured-512|hdr1|hyb0|ent0|rr-none | 724 | 0.859 | 0.553 | 0.664 | 0.362 | 2062 | 1.2 | 1.4 |
| tfidf|structured-512|hdr1|hyb1|ent0|rr-none | 724 | 0.883 | 0.601 | 0.711 | 0.427 | 2100 | 2.4 | 2.9 |
| tfidf-char|structured-512|hdr1|hyb1|ent0|rr-none | 724 | 0.812 | 0.587 | 0.766 | 0.494 | 2117 | 4.1 | 5.2 |
| tfidf|structured-512|hdr1|hyb0|ent1|rr-none | 724 | 0.875 | 0.624 | 0.781 | 0.530 | 2136 | 2.5 | 3.0 |
| tfidf|structured-512|hdr1|hyb1|ent1|rr-none | 724 | 0.891 | 0.637 | 0.820 | 0.548 | 2144 | 3.1 | 4.4 |
| tfidf-char|structured-512|hdr1|hyb1|ent1|rr-none | 724 | 0.867 | 0.736 | 0.852 | 0.653 | 2149 | 6.7 | 7.8 |
| e5-small|structured-512|hdr1|hyb1|ent1|rr-none | 724 | 0.969 | 0.818 | 0.898 | 0.722 | 2172 | 28.4 | 31.3 |
| e5-small|structured-512|hdr1|hyb0|ent1|rr-none | 724 | 0.961 | 0.833 | 0.891 | 0.735 | 2185 | 34.6 | 47.3 |
| tfidf-char|structured-1024|hdr1|hyb0|ent1|rr-none | 358 | 0.961 | 0.870 | 0.922 | 0.800 | 3975 | 4.6 | 5.1 |
| tfidf-char|structured-1024|hdr1|hyb1|ent1|rr-none | 358 | 0.969 | 0.885 | 0.945 | 0.795 | 4145 | 6.5 | 10.0 |
| e5-small|structured-2048|hdr0|hyb0|ent1|rr-none | 171 | 0.930 | 0.889 | 0.852 | 0.806 | 7556 | 31.0 | 33.8 |
| tfidf|structured-2048|hdr1|hyb0|ent1|rr-none | 171 | 1.000 | 0.911 | 0.922 | 0.792 | 7595 | 2.3 | 4.7 |
| e5-small|fixed-2048|hdr0|hyb0|ent1|rr-none | 171 | 0.945 | 0.905 | 0.883 | 0.823 | 7701 | 38.0 | 46.7 |
| tfidf-char|structured-2048|hdr1|hyb0|ent1|rr-none | 171 | 0.992 | 0.922 | 0.977 | 0.864 | 7869 | 3.8 | 4.5 |
| tfidf-char|fixed-2048|hdr1|hyb0|ent1|rr-none | 171 | 0.992 | 0.933 | 0.984 | 0.877 | 8149 | 4.2 | 4.8 |
| tfidf-char|structured-2048|hdr0|hyb0|ent1|rr-none | 171 | 0.992 | 0.976 | 0.977 | 0.904 | 8223 | 3.5 | 5.1 |

## 2. Config escolhida no dev, avaliada no test: `tfidf-char|structured-1024|hdr1|hyb1|ent1|rr-none`

| variante | n | hit@1 | hit@5 [IC95%] | MRR@10 [IC95%] | recall@5 | hit@cand (teto reranker) |
|---|---|---|---|---|---|---|
| clean | 136 | 0.824 | 0.949 [0.904, 0.985] | 0.882 [0.835, 0.922] | 0.908 | 0.993 |
| noisy | 136 | 0.706 | 0.904 [0.853, 0.949] | 0.798 [0.743, 0.850] | 0.860 | 0.985 |

## 3. Selecionada vs baseline ingênuo (`tfidf|fixed-512|hdr0|hyb0|ent0|rr-none`), test, bootstrap pareado

| variante | Δ MRR@10 | IC95% | P(Δ≤0) |
|---|---|---|---|
| clean | +0.658 | [+0.591, +0.728] | 0.000 |
| noisy | +0.662 | [+0.597, +0.723] | 0.000 |

## 4. Efeito marginal de cada fator (média sobre as demais configs, dev, MRR@10)

| fator | nível | clean | noisy | n configs |
|---|---|---|---|---|
| embedder | bge-m3 | 0.759 | 0.652 | 48 |
| embedder | e5-small | 0.696 | 0.606 | 48 |
| embedder | lsa | 0.720 | 0.591 | 48 |
| embedder | minilm-multi | 0.540 | 0.409 | 48 |
| embedder | tfidf | 0.723 | 0.588 | 48 |
| embedder | tfidf-char | 0.709 | 0.650 | 48 |
| strategy | fixed | 0.677 | 0.572 | 144 |
| strategy | structured | 0.705 | 0.593 | 144 |
| chunk_size | 1024 | 0.709 | 0.599 | 96 |
| chunk_size | 2048 | 0.855 | 0.727 | 96 |
| chunk_size | 512 | 0.509 | 0.422 | 96 |
| header | 0 | 0.652 | 0.554 | 144 |
| header | 1 | 0.730 | 0.611 | 144 |
| hybrid | 0 | 0.661 | 0.555 | 144 |
| hybrid | 1 | 0.722 | 0.610 | 144 |
| entity | 0 | 0.603 | 0.476 | 144 |
| entity | 1 | 0.779 | 0.689 | 144 |
| reranker | none | 0.691 | 0.582 | 288 |

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
- 288 configs avaliadas e a melhor foi escolhida no dev: o dev é otimista por construção; o test é o número que vale.