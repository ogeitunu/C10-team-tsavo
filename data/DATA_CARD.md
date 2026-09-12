# Data Card: TRI AI Agricultural Extension RAG Benchmark

## Benchmark Overview
- **Domain:** Sub-Saharan African Smallholder Agriculture (crop diseases, pests, soil fertility, climate, fertilizer guidance).
- **Task:** Information Retrieval (IR) & Ranking for Retrieval-Augmented Generation (RAG). Given a farmer's natural query, retrieve and rank the top 5 most relevant documents from a fixed knowledge base.
- **Evaluation Metric:** Normalized Discounted Cumulative Gain at 5 (**nDCG@5**).
- **Baseline Score:** TF-IDF Cosine Similarity baseline achieves **nDCG@5 = 0.551**.

---

## Corpus & Query Dataset Specifications

| File Name | Row Count | Description |
| :--- | :--- | :--- |
| **`documents.csv`** | 695 | The complete knowledge base to retrieve from. |
| **`train_queries.csv`**| 308 | Training set queries with positive document helpers. |
| **`qrels_train.csv`** | 4,194 | Graded relevance labels ($0, 1, 2, 3$) for training queries. |
| **`test_queries.csv`** | 200 | Hidden test set queries (`QueryId` $1001 \rightarrow 1200$). |
| **`sample_submission.csv`** | 1,000 | Output format benchmark ($200 \text{ queries} \times 5 \text{ rows}$). |

---

## Graded Relevance Schema

- **3 (Perfect):** Directly answers the question (e.g., exact identification of nitrogen deficiency in maize across agro-zones).
- **2 (Relevant):** A complementary facet of the topic (e.g., symptom management or complementary treatments).
- **1 (Marginal):** The same issue described on a different crop type.
- **0 (Not Relevant / Hard Negative):** Lexically similar text answering the wrong intent (e.g., prevention vs. treatment or confusable crop symptoms).

---

## Feature Schemas

### `documents.csv`
- `document_id` *(str)*: Unique document primary key.
- `title` *(str)*: Extension document title.
- `text` *(str)*: Main factsheet content.
- `source` *(str)*: Attributed publishing body (e.g., FAO, CGIAR, Plantwise, IITA).
- `crop` *(str)*: Target primary crop or general soil/climate category.
- `country` *(str)*: Country context.
- `origin` *(str)*: Grounded (`llm_grounded`) or synthetic (`synthetic`).

### `test_queries.csv` & Submissions
- `QueryId` *(str)*: Test query ID ($1001 \rightarrow 1200$).
- `DocumentId` *(str)*: Retrieved top-5 document ID (ordered by rank 1 through 5).

---

## License & Attribution
- Grounded documents retain **CC-BY / CC0** attribution from CGIAR / CGSpace open-access materials.
- Synthetic documents released under **CC0 (Public Domain)**.
