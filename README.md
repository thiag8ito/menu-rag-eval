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

Sem `make` (Windows): cada alvo do `Makefile` é um `python -m ...`; copie o comando de lá.

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
   recuperar o documento inteiro, o que sobe hit@k mandando ~2× mais contexto ao LLM (ver fronteira de Pareto em `results/full/report.md`).
3. O número final é o da config escolhida no **test**, com IC95% por bootstrap, comparado por bootstrap pareado com um
   **baseline ingênuo** (fixed-512, sem header, sem híbrido), não com a pior config (isso seria cherry-picking).
4. Métricas: hit@1/3/5/10, MRR@10, recall@k (listas), nDCG@10, `hit@cand` (teto do reranker), latência p50/p95, custo de contexto.

**Geração (LLM-as-judge)**, `evaluation/judge.py` e `run_e2e.py`: fidelidade estilo RAGAS (decompõe a resposta em
afirmações e verifica cada uma contra o contexto), correção com rubrica por modo (exata / lista com precisão / abstenção),
taxa de falsa abstenção, e classificação das falhas em `retrieval_miss`, `false_abstention`, `generation_error`,
`hallucination`. Juiz com temperatura 0 (quando o SDK/modelo aceita o parâmetro; senão roda com o padrão e o log do e2e deve ser lido sabendo disso), JSON, modelo maior e diferente do gerador (reduz viés de autopreferência),
sanity check juiz × regra determinística de abstenção, e calibração com rotulagem humana **cega** (`calibrate_judge.py`, kappa).

## Resultados medidos

Todos gerados por `evaluation/run_experiments.py` (escolha no dev, número final no test):

- `results/full/report.md`: grid completo, **288 configs** (6 embedders × fixed/structured × chunk 512/1024/2048 × header × híbrido × entity linking).
- `results/rerank/report.md`: **24 configs** com e sem reranker `mmarco` (3 embedders × chunk 512/1024).
- `results/report.md`: grid offline (TF-IDF, n-gramas, LSA), inclui chunk de 256.

Config escolhida no dev: **`tfidf-char | structured-1024 | header | híbrido (BM25+RRF) | entity linking`**, sem reranker
(~4,1k caracteres de contexto no top-5, p50 de 4 a 7 ms). No **test** (136 perguntas):

| variante | hit@1 | hit@5 [IC95%] | MRR@10 [IC95%] |
|---|---|---|---|
| clean | 0,824 | 0,949 [0,904; 0,985] | 0,882 [0,835; 0,922] |
| noisy | 0,706 | 0,904 [0,853; 0,949] | 0,798 [0,743; 0,850] |

Contra o baseline ingênuo: Δ MRR@10 = **+0,66** (IC95% [+0,59; +0,73]) em clean e **+0,66** ([+0,60; +0,72]) em noisy.
Como o IC95% do MRR no test é de ±0,04 a ±0,05, diferenças menores que ~0,03 entre configs devem ser lidas como empate.

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
No grid completo, o efeito marginal no dev é MRR@10 **0,60 → 0,78** (clean) e **0,48 → 0,69** (noisy). No test:
hit@1 0,66 → 0,82 (clean) e 0,53 → 0,71 (noisy).

Um teste de invariantes (`tests/test_entity.py`) pegou um falso positivo real do linker na primeira versão ("Tem alguma
opção … comida brasileira" vinculava a "Tempero Brasil" por trigramas coincidentes). A varredura de limiares no dev
*não* o detectava; a correção foi trocar o gate por distância de edição, com teste de regressão.

### Embeddings: um baseline lexical empata com o melhor denso

MRR@10 no dev, média sobre as 48 configs de cada embedder (`results/full/report.md`, seção 4):

| embedder | clean | noisy | queda com ruído |
|---|---|---|---|
| bge-m3 | 0,759 | 0,652 | −0,107 |
| tfidf (palavras) | 0,723 | 0,588 | −0,135 |
| lsa | 0,720 | 0,591 | −0,129 |
| tfidf-char (n-gramas de caracteres) | 0,709 | 0,650 | **−0,059** |
| e5-small | 0,696 | 0,606 | −0,090 |
| minilm-multi | 0,540 | 0,409 | −0,131 |

- **Na config escolhida, `tfidf-char` e `bge-m3` empatam:** MRR@10 0,885 vs 0,879 (clean) e 0,795 vs 0,793 (noisy), dentro
  do ruído. O `bge-m3` custa ~40× mais por consulta (p50 259 ms contra 6,5 ms; `e5-small`: 28 ms).
- Ele ganha na média (0,76 vs 0,71 em clean), o que sugere que ajuda em configs mais fracas (chunk pequeno, sem header),
  mas isso é hipótese: o que os dados mostram é o empate na melhor config.
- **n-gramas de caracteres são o mais robustos a ruído** (−0,06, menos que qualquer denso), e as posições mudam entre
  clean e noisy: avaliar só com perguntas limpas, o TF-IDF de palavras pareceria melhor que o de caracteres (0,723 contra 0,709), e a ordem se inverte em noisy (0,588 contra 0,650).
- `minilm-multi` é o mais fraco. Não isolei a causa (treinado para paráfrases, não para busca; e, até onde sei, trunca em
  ~128 tokens).
- **Ressalva importante:** o corpus é lexical por construção (as perguntas citam o nome exato do restaurante e do prato),
  o que favorece TF-IDF. Com perguntas parafraseadas de usuários reais, é de esperar que os densos se destaquem; não
  testei isso.

### Reranker (`mmarco`): ajuda quando o 1º estágio é fraco, não quando é forte

No relatório de rerank, a média sobre 12 configs com reranker contra 12 sem (mesmos embedders, tamanhos e híbrido) é
**+0,086 (clean) e +0,081 (noisy)**. A média esconde o padrão, que aparece nos pares (Δ MRR@10 no dev):

| embedder | chunk | Δ clean | Δ noisy |
|---|---|---|---|
| minilm-multi | 512 | +0,16 a +0,22 | +0,19 a +0,20 |
| tfidf-char | 512 | +0,13 | +0,09 a +0,11 |
| e5-small | 512 | +0,03 a +0,04 | −0,01 a +0,01 |
| tfidf-char | 1024 | −0,01 a +0,01 | −0,01 a 0,00 |
| e5-small | 1024 (híbrido) | 0,00 | +0,04 |

(faixas = com e sem híbrido, quando os dois pares existem). A latência p50 sobe de ~4 ms para ~0,6 a 1,0 s em chunk 512 e ~1,3 a 1,7 s em 1024 (>150× a >300×; CPU, sem GPU).

- Com chunk 1024, a melhor config **sem** reranker já é a melhor do grid: o reranker não agrega, e `hit@cand` ≈ 0,99
  mostra que não faltam candidatos, o 1º estágio já ordena bem.
- Com chunk 512, o reranker compensa um 1º estágio fraco e chega perto do 1024 em clean (0,84 a 0,87 contra 0,885) usando
  54% do contexto (2,2k contra 4,1k caracteres), mas fica abaixo em noisy (0,66 a 0,75 contra 0,80) e custa 0,6 a 1,0 s por consulta.
- O reranker `bge-reranker` não foi avaliado.

### Chunk size e contexto

MRR@10 médio no dev (clean / noisy): 512 → 0,51 / 0,42; 1024 → 0,71 / 0,60; 2048 → 0,86 / 0,73, com contexto de ~2,1k,
~4,1k e ~8,2k caracteres. A melhor config de 2048 (`tfidf-char`, sem header, sem híbrido) chega a 0,976 / 0,904 no dev
contra 0,885 / 0,795 da escolhida: **+0,09 / +0,11 de MRR por 2× o contexto**. É o dilema que o e2e precisa responder:
mais contexto traz mais evidência, mas pode diluir a resposta (medir fidelidade e correção com 1024 e com 2048). Em 2048
o header não ajuda e até piora (0,976 sem contra 0,922 com, mesma config), plausivelmente porque o chunk já contém o título
do restaurante; é hipótese, não testei.

### Demais fatores (efeito marginal no dev, MRR@10, 288 configs)

| fator | clean | noisy |
|---|---|---|
| entity linking | 0,60 → 0,78 (+0,18) | 0,48 → 0,69 (+0,21) |
| header contextual | 0,65 → 0,73 (+0,08) | 0,55 → 0,61 (+0,06) |
| híbrido (BM25 + RRF) | 0,66 → 0,72 (+0,06) | 0,56 → 0,61 (≈ +0,06) |
| chunking estruturado vs fixo | 0,68 → 0,71 (+0,03) | 0,57 → 0,59 (+0,02) |

**Discovery é o tipo mais difícil** (hit@5 0,83 clean / 0,80 noisy no test), justamente o que não cita restaurante.

## Próximos passos

- [x] Embeddings densos (`minilm-multi`, `e5-small`, `bge-m3`) e reranker `mmarco`.
- [ ] `make e2e` (clean e noisy) com a config escolhida **e** com a de 2048: fidelidade, correção, abstenção, falhas.
- [ ] Rotular 40 respostas com `calibrate_judge` e reportar o kappa do juiz.
- [ ] Avaliar o `bge-reranker` (mais pesado) e salvar os resultados por pergunta das top-K configs, para testar
      diferenças entre elas por bootstrap pareado (hoje só a config escolhida tem isso).

## Limitações

1. Corpus e perguntas sintéticos: o ruído também é sintético, e o corpus é lexical por construção (ver ressalva em
   Embeddings). O ideal é complementar com perguntas reais anonimizadas e uma amostra revisada à mão.
2. Corpus pequeno (~360 chunks em 1024): diferenças entre embeddings podem não generalizar para milhões de itens.
3. "LSA" é um baseline semântico offline, não um embedding neural.
4. Índice por força bruta em memória. Em produção: FAISS/pgvector e filtros por metadados (bairro/cidade) em vez de
   depender do texto, que é o que o header contextual aproxima.
5. A seleção de config usa dev e é otimista por construção (288 configs); por isso o número que vale é o do test.
6. **O entity linker nasceu de uma análise de erros feita sobre todas as perguntas (dev + test)**; só os limiares foram
   escolhidos no dev. O ganho no test é, portanto, um pouco otimista: a *ideia* veio de olhar o conjunto inteiro. Além
   disso, 82% das perguntas citam o restaurante pelo nome (todas menos discovery); em tráfego real essa fração é menor e
   o ganho tende a ser menor também.
7. Só a config escolhida tem resultados por pergunta salvos; as comparações entre configs vizinhas (ex.: `tfidf-char`
   contra `bge-m3`) usam médias e IC, não teste pareado.
8. O reranker foi avaliado só com chunk 512/1024, header, entity linking e 3 embedders (24 configs).

## Estrutura

```
rag/         chunking, embeddings (tfidf/char/lsa/ST), bm25+RRF, entity linker, index, reranker, retriever, generator, API
evaluation/  build_dataset, noise, metrics (+bootstrap), run_experiments, judge, run_e2e, calibrate_judge
tests/       métricas, chunking, invariantes do dataset/ruído, retrieval/híbrido/persistência, entity linker, lógica do e2e (LLM mockado), API
.github/     CI: testes + gate de regressão do retrieval
```
