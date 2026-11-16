# Avaliação de retrieval

Seleção no **dev** (média do MRR@10 clean/noisy); resultado final no **test**. `clean` = perguntas bem escritas; `noisy` = coloquial + typos + sem acento.

## 1. Top-10 configs elegíveis (dev, contexto ≤ 5000 chars)

| config | chunks | clean hit@5 | clean MRR@10 | noisy hit@5 | noisy MRR@10 | ctx chars@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| tfidf-char|structured-1024|hdr1|hyb1|ent1|rr-none | 358 | 0.969 | 0.885 | 0.945 | 0.795 | 4145 | 5.4 | 6.4 |
| tfidf-char|structured-1024|hdr1|hyb0|ent1|rr-none | 358 | 0.961 | 0.870 | 0.922 | 0.800 | 3975 | 4.4 | 5.7 |
| tfidf-char|fixed-1024|hdr1|hyb1|ent1|rr-none | 362 | 0.977 | 0.847 | 0.945 | 0.779 | 4298 | 5.3 | 6.5 |
| tfidf-char|fixed-1024|hdr1|hyb0|ent1|rr-none | 362 | 0.969 | 0.839 | 0.938 | 0.784 | 4129 | 4.4 | 6.1 |
| tfidf-char|structured-1024|hdr0|hyb0|ent1|rr-none | 358 | 0.922 | 0.822 | 0.875 | 0.765 | 4196 | 4.6 | 5.9 |
| tfidf-char|fixed-1024|hdr0|hyb0|ent1|rr-none | 362 | 0.930 | 0.816 | 0.883 | 0.770 | 4335 | 4.5 | 5.7 |
| tfidf-char|fixed-1024|hdr0|hyb1|ent1|rr-none | 362 | 0.930 | 0.813 | 0.883 | 0.755 | 4492 | 4.9 | 6.6 |
| tfidf|structured-1024|hdr1|hyb1|ent1|rr-none | 358 | 0.977 | 0.827 | 0.898 | 0.737 | 4087 | 2.5 | 3.7 |
| tfidf-char|structured-1024|hdr0|hyb1|ent1|rr-none | 358 | 0.922 | 0.814 | 0.875 | 0.746 | 4324 | 4.7 | 5.7 |
| lsa|structured-1024|hdr0|hyb1|ent1|rr-none | 358 | 0.922 | 0.820 | 0.844 | 0.729 | 4298 | 3.4 | 4.2 |

**Sem restrição de contexto**, o vencedor seria `tfidf-char|structured-2048|hdr0|hyb0|ent1|rr-none` (8223 chars): com chunks desse tamanho o top-5 já é quase o menu inteiro de 5 restaurantes. É recuperação de documento, não de passagem, e custa muito mais tokens.

### Fronteira de Pareto (melhor score de dev para cada nível de contexto)

| config | chunks | clean hit@5 | clean MRR@10 | noisy hit@5 | noisy MRR@10 | ctx chars@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| tfidf|structured-256|hdr0|hyb0|ent0|rr-none | 1919 | 0.094 | 0.039 | 0.070 | 0.035 | 816 | 1.1 | 2.0 |
| tfidf|structured-256|hdr0|hyb1|ent0|rr-none | 1919 | 0.086 | 0.047 | 0.078 | 0.044 | 845 | 2.1 | 2.7 |
| tfidf-char|structured-256|hdr0|hyb0|ent0|rr-none | 1919 | 0.125 | 0.067 | 0.094 | 0.061 | 864 | 4.7 | 7.3 |
| tfidf-char|structured-256|hdr0|hyb1|ent0|rr-none | 1919 | 0.109 | 0.063 | 0.125 | 0.066 | 868 | 4.3 | 5.9 |
| tfidf|structured-256|hdr1|hyb0|ent1|rr-none | 1919 | 0.781 | 0.511 | 0.688 | 0.394 | 886 | 2.3 | 3.7 |
| tfidf|structured-256|hdr1|hyb1|ent1|rr-none | 1919 | 0.789 | 0.523 | 0.688 | 0.410 | 901 | 3.2 | 4.3 |
| tfidf-char|structured-256|hdr1|hyb0|ent1|rr-none | 1919 | 0.820 | 0.619 | 0.828 | 0.555 | 908 | 7.6 | 16.0 |
| tfidf-char|structured-256|hdr1|hyb1|ent1|rr-none | 1919 | 0.844 | 0.627 | 0.836 | 0.579 | 920 | 11.1 | 27.5 |
| tfidf-char|structured-512|hdr1|hyb1|ent1|rr-none | 724 | 0.867 | 0.736 | 0.852 | 0.653 | 2149 | 7.3 | 8.5 |
| tfidf|structured-1024|hdr1|hyb0|ent1|rr-none | 358 | 0.969 | 0.801 | 0.867 | 0.712 | 3877 | 1.9 | 2.9 |
| tfidf-char|structured-1024|hdr1|hyb0|ent1|rr-none | 358 | 0.961 | 0.870 | 0.922 | 0.800 | 3975 | 4.4 | 5.7 |
| tfidf-char|structured-1024|hdr1|hyb1|ent1|rr-none | 358 | 0.969 | 0.885 | 0.945 | 0.795 | 4145 | 5.4 | 6.4 |
| tfidf|structured-2048|hdr1|hyb0|ent1|rr-none | 171 | 1.000 | 0.911 | 0.922 | 0.792 | 7595 | 1.7 | 2.7 |
| tfidf-char|structured-2048|hdr1|hyb0|ent1|rr-none | 171 | 0.992 | 0.922 | 0.977 | 0.864 | 7869 | 3.7 | 4.5 |
| tfidf-char|fixed-2048|hdr1|hyb0|ent1|rr-none | 171 | 0.992 | 0.933 | 0.984 | 0.877 | 8149 | 3.6 | 5.0 |
| tfidf-char|structured-2048|hdr0|hyb0|ent1|rr-none | 171 | 0.992 | 0.976 | 0.977 | 0.904 | 8223 | 3.6 | 4.5 |

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
| embedder | lsa | 0.633 | 0.515 | 64 |
| embedder | tfidf | 0.609 | 0.494 | 64 |
| embedder | tfidf-char | 0.615 | 0.561 | 64 |
| strategy | fixed | 0.588 | 0.501 | 96 |
| strategy | structured | 0.650 | 0.545 | 96 |
| chunk_size | 1024 | 0.753 | 0.642 | 48 |
| chunk_size | 2048 | 0.904 | 0.766 | 48 |
| chunk_size | 256 | 0.324 | 0.264 | 48 |
| chunk_size | 512 | 0.494 | 0.421 | 48 |
| header | 0 | 0.570 | 0.488 | 96 |
| header | 1 | 0.667 | 0.558 | 96 |
| hybrid | 0 | 0.607 | 0.513 | 96 |
| hybrid | 1 | 0.631 | 0.533 | 96 |
| entity | 0 | 0.545 | 0.430 | 96 |
| entity | 1 | 0.693 | 0.616 | 96 |
| reranker | none | 0.619 | 0.523 | 192 |

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
- 192 configs avaliadas e a melhor foi escolhida no dev: o dev é otimista por construção; o test é o número que vale.