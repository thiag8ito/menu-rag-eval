# Menu RAG — busca semântica em cardápios, com avaliação rigorosa

RAG sobre cardápios sintéticos de restaurantes de delivery, com API em FastAPI. O foco não é "fazer um RAG
funcionar", e sim **medir se ele funciona e decidir com dados**: dataset de teste com gabarito exato,
hit@k/MRR no retrieval, LLM-as-judge para fidelidade, e comparação de chunking, embeddings, busca híbrida e reranker,
com protocolo dev/test, intervalos de confiança e custo de contexto.

```
menus.jsonl ─► chunking (fixed | structured, ±header contextual) ─► embedder ─► VectorIndex (cosseno)
                                                                                    │
query ─► embed ─► top-N densos ─┐                                                  │
                                ├─ RRF ─► [entity link: restaurante citado primeiro] ─► [reranker] ─► top-k
query ─► BM25 ─► top-N ─────────┘                                                                     │
                                                      LLM (só com o contexto) ─► resposta + fontes
                                                      API: POST /search  POST /ask  GET /health
```

## Rodando

```bash
make install            # núcleo + testes. Modelos densos: pip install -r requirements-dense.txt
make test
make smoke              # ~4 min, 100% offline (scikit-learn): gera corpus, dataset e results/report.md
make index serve        # API em :8000  (export ANTHROPIC_API_KEY para /ask)
make e2e                # geração + LLM-as-judge (usa a API da Anthropic)
```

```bash
curl -X POST localhost:8000/search -H 'content-type: application/json' \
  -d '{"query":"opção vegana de pizza no Centro","k":3}'
```

Docker: `docker build -t menu-rag .` (índice offline com `tfidf-char`; densos via `--build-arg EMBEDDER=minilm-multi --build-arg DENSE=1`).

## O corpus

150 restaurantes, ~1.500 pratos, 8 culinárias (`rag/generate_menus.py`, determinístico). Decisões de design que tornam
a avaliação difícil de propósito:

- **Os mesmos pratos aparecem em vários restaurantes**, com preços diferentes. "Quanto custa a Margherita no X?" só é
  respondida certo se o retriever distingue o restaurante, não só o prato (hard negatives reais).
- Cada documento tem seção "Sobre" e **avaliações de clientes** que citam pratos do mesmo restaurante sem trazer preço
  nem ingredientes: distratores *dentro* do documento, que fazem o chunking importar.

## O dataset de avaliação

300 perguntas (`evaluation/build_dataset.py`), gabarito derivado dos dados estruturados (sem ruído de anotação), 5 tipos:

| tipo | exemplo | gabarito |
|---|---|---|
| price / ingredients | "Quanto custa X no Y?" | 1 prato |
| dietary | "O Y tem opção vegana?" | lista de pratos |
| discovery | "opção fit de hambúrguer no bairro Z, cidade W?" (sem nome do restaurante) | pratos/restaurantes |
| unanswerable | "Qual o telefone do Y?" | o sistema deve se abster |

Cada pergunta existe em duas versões do **mesmo pedido**: `clean` (template) e `noisy` (reformulação coloquial + typos de
teclado, sem acento, abreviações de chat: `"qnt e wafflle c/ sorvete do sorvete fresh"`, ver `evaluation/noise.py`).
Split dev/test 50/50. Há testes de invariantes do gabarito (`tests/test_dataset.py`): bebidas fora do gold de discovery,
prato ausente de fato nas perguntas sem resposta, concordância de gênero, ruído preserva a identidade do pedido.

**Relevância no nível de evidência:** um chunk é relevante se contém a linha *completa* de um prato-gold
(`Chunk.dish_ids`). Isso torna tamanhos de chunk comparáveis e penaliza cortar um prato no meio.

## Protocolo de avaliação

1. Cada config (embedder × chunking × tamanho × header × híbrido × entity linking × reranker) é avaliada em clean e noisy, em dev e test.
2. A config é **escolhida no dev** (média do MRR@10 clean/noisy) **dentro de um orçamento de contexto**
   (caracteres do top-5 enviados ao LLM). Sem o orçamento, o vencedor degenera para chunks de 2048 caracteres, ou seja,
   recuperar o documento inteiro, o que sobe hit@k mandando ~2x mais tokens (ver fronteira de Pareto em `results/report.md`).
3. O número final é o da config escolhida no **test**, com IC95% por bootstrap, comparado por bootstrap pareado com um
   **baseline ingênuo** (fixed-512, sem header, sem híbrido), não com a pior config (isso seria cherry-picking).
4. Métricas: hit@1/3/5/10, MRR@10, recall@k (listas), nDCG@10, `hit@cand` (teto do reranker), latência p50/p95, custo de contexto.

**Geração (LLM-as-judge)**, `evaluation/judge.py` e `run_e2e.py`: fidelidade estilo RAGAS (decompõe a resposta em
afirmações e verifica cada uma contra o contexto), correção com rubrica por modo (exata / lista com precisão / abstenção),
taxa de falsa abstenção, e classificação das falhas em `retrieval_miss`, `false_abstention`, `generation_error`,
`hallucination`. Juiz com temperatura 0, JSON, modelo maior e diferente do gerador (reduz viés de autopreferência),
sanity check juiz × regra determinística de abstenção, e calibração com rotulagem humana **cega** (`calibrate_judge.py`, kappa).

## Resultados medidos (offline: TF-IDF palavras, TF-IDF n-gramas de caracteres, LSA; 192 configs)

Config escolhida no dev: **`tfidf-char | structured-1024 | header | híbrido (BM25+RRF) | entity linking`**
(~4,2k caracteres de contexto no top-5, p50 3,2 ms). No **test** (136 perguntas):

| variante | hit@1 | hit@5 [IC95%] | MRR@10 [IC95%] |
|---|---|---|---|
| clean | 0,824 | 0,949 [0,904; 0,985] | 0,882 [0,835; 0,922] |
| noisy | 0,706 | 0,904 [0,853; 0,949] | 0,798 [0,743; 0,850] |

Contra o baseline ingênuo: Δ MRR@10 = **+0,66** (IC95% [+0,59; +0,73]) em clean e **+0,66** ([+0,60; +0,72]) em noisy.

### Análise de erros → correção → ganho

A primeira versão (sem entity linking) tinha hit@1 = 0,66 no test. Classificando o top-1 das perguntas de preço e
ingredientes (161, config 1024 + header + híbrido, clean):

| top-1 | n | % |
|---|---|---|
| correto | 91 | 57% |
| **restaurante errado, MESMO prato** | **59** | **37%** |
| restaurante certo, chunk errado | 11 | 7% |

O gargalo não era chunking nem embedding: era *desambiguar qual restaurante o usuário citou* (o mesmo prato existe em
muitos cardápios, e o nome do restaurante se dilui num chunk de ~1000 caracteres). Correção: `rag/entity.py`, um linker
por containment de trigramas + gate por distância de edição que prioriza os chunks do restaurante citado. Medido no dev
(limiares escolhidos só nele): **precisão 1,00**, recall 0,94 (clean) e 0,81 (noisy), 0 vínculos espúrios em discovery.
Efeito marginal no dev: MRR@10 **0,55 → 0,69** (clean) e **0,43 → 0,62** (noisy). No test: hit@1 0,66 → 0,82 (clean) e
0,53 → 0,71 (noisy).

Um teste de invariantes (`tests/test_entity.py`) pegou um falso positivo real do linker na primeira versão ("Tem alguma
opção … comida brasileira" vinculava a "Tempero Brasil" por trigramas coincidentes). A varredura de limiares no dev
*não* o detectava; a correção foi trocar o gate por distância de edição, com teste de regressão.

### O que mais os experimentos mostram (efeitos marginais no dev, MRR@10 em clean; `results/report.md`, seção 4)

- **Header contextual:** repetir nome/culinária/bairro no texto embedado, 0,57 → 0,67. Sem ele, perguntas de discovery
  (sem nome do restaurante) não têm de onde tirar o contexto.
- **Chunk size é um trade-off de custo:** 256 → 0,32; 512 → 0,49; 1024 → 0,75; 2048 → 0,90, mas 2048 manda ~8,2k
  caracteres ao LLM contra ~4,2k. Chunks pequenos cortam a evidência; grandes viram recuperação de documento. Por isso
  a seleção usa orçamento de contexto.
- **Ruído e tipo de representação.** Sem entity linking, ruído derruba o TF-IDF de palavras (MRR 0,81 → 0,60, clean →
  noisy) bem mais que o de n-gramas de caracteres (0,76 → 0,65), e as posições se invertem entre clean e noisy. Com
  entity linking, n-gramas de caracteres ganham nas duas (0,89 / 0,79 contra 0,83 / 0,74). Avaliar só com perguntas limpas
  teria dado um ranking diferente.
- **Híbrido (+0,02) e chunking estruturado (+0,06) ajudam, pouco.**
- **Discovery é o tipo mais difícil** (hit@5 0,83 clean / 0,80 noisy no test), justamente o que não cita restaurante.

## Pendente: rodar localmente (o código está pronto; eu não tive rede para baixar modelos nem chave de API)

- [ ] `make experiments` (embeddings densos `minilm-multi`, `e5-small`, `bge-m3`) e `make experiments-rerank`
      (`mmarco`, `bge-reranker`). Pergunta a responder: algum denso supera n-gramas de caracteres em noisy? O reranker
      compensa a latência p95? (o teto é `hit@cand`)
- [ ] `make e2e` (clean e noisy): fidelidade, correção, abstenção, quebra de falhas.
- [ ] Rotular 40 respostas com `calibrate_judge` e reportar o kappa do juiz.
- [ ] Colar os números aqui na seção de resultados.

## Limitações

1. Corpus e perguntas sintéticos: o ruído também é sintético. O ideal é complementar com perguntas reais anonimizadas
   e uma amostra revisada à mão.
2. Corpus pequeno (~360 chunks em 1024): diferenças entre embeddings podem não generalizar para milhões de itens.
3. "LSA" aqui é um baseline semântico offline, não um embedding neural: não substitui os modelos densos pendentes acima.
4. Índice por força bruta em memória. Em produção: FAISS/pgvector e filtros por metadados (bairro/cidade) em vez de
   depender do texto, que é o que o header contextual aproxima.
5. A seleção de config usa dev e é otimista por construção (192 configs); por isso o número que vale é o do test.
6. **O entity linker nasceu de uma análise de erros feita sobre todas as perguntas (dev + test)**; só os limiares foram
   escolhidos no dev. O ganho no test é, portanto, um pouco otimista: a *ideia* veio de olhar o conjunto inteiro. Além
   disso, 82% das perguntas citam o restaurante pelo nome (todas menos discovery); em tráfego real essa fração é menor e
   o ganho tende a ser menor também.

## Estrutura

```
rag/         chunking, embeddings (tfidf/char/lsa/ST), bm25+RRF, entity linker, index, reranker, retriever, generator, API
evaluation/  build_dataset, noise, metrics (+bootstrap), run_experiments, judge, run_e2e, calibrate_judge
tests/       métricas, chunking, invariantes do dataset/ruído, retrieval/híbrido/persistência, entity linker, lógica do e2e (LLM mockado), API
.github/     CI: testes + gate de regressão do retrieval
```
