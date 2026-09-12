# C10-team-tsavo
Tri AI ML/AI training program in conjunction with Kaggle Competition
# High-Precision Hybrid Agricultural RAG System (Team Tsavo)

**Repository Name:** `C10-team-tsavo`  
Cohort: TRI AI Saturdays Cohort 10  
Competition Outcome: Top-10 Leaderboard Finish (9th / 25 Teams) | Final Evaluation Score: 0.86972 MRR@10  
**Team Members:**
Fagbamigbe kehinde (active)
Chiemerie Chinedu Favour (Active)
Ogechuwu itunu Aina (Active)
Abolaji David Adeoye	
Abdulrahman Buhari Auta	 (Active)
Adeboye Olugbenga Abiodun	
Joseph Edet	
Sadiya Muhammad Kilgori
Seun Ajayi- mentor (active)

## Dataset

The dataset originates from the Kaggle TRI AI Agricultural Extension RAG Competition. It consists of agronomic extension queries paired with a multi-field document corpus containing agricultural guides, pest management protocols, soil science manuals, and crop management strategies.

### Selection & Preprocessing
Metadata Fusion: Raw document body text was enriched by fusing structural metadata fields into a unified schema: Title | Category | Section Heading | Body Text. This eliminated context loss during semantic vector search.
Lexical Cleaning: Applied custom tokenization to clean text while strictly preserving chemical formulations, NPK fertilizer ratios, crop names, and metric units.
Index Integrity: Document primary keys were mapped directly to global dataset IDs to ensure zero index-shifting during evaluation exports.

## Training & Retrieval Pipeline

Our architecture implements a multi-stage hybrid retrieval system combining sparse lexical search, dense vector space search, and dual cross-encoder logit re-ranking.
