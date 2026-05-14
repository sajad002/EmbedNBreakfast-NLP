# EmbedNBreakfast — RecipeNLG NLP Project (Assignment 1)

This repository contains our end-to-end NLP project built on the **RecipeNLG** dataset (`full_dataset.csv`). The work is organized as a set of Jupyter/Colab notebooks covering:

- Exploratory analysis + **classical ML** cuisine classification
- Keyword + lexical + hybrid **recipe search** (BM25 / TF‑IDF)
- Domain **Word2Vec** embeddings on Recipes1M subset
- Sentence-embedding **clustering** (USE + HDBSCAN + Bayesian GMM)
- **Zero-shot** recipe type classification (BART‑MNLI)
- Domain-adaptive **DistilBERT MLM fine-tuning** with **LoRA** + embedding extraction + FAISS similarity search
- **GPT‑2 fine-tuning** for recipe generation (+ 0/1/few‑shot comparisons)
- A Colab-based **Voice Assistant** (Whisper STT + FAISS RAG + Llama‑3.2 + gTTS)

## Project video

- https://drive.google.com/file/d/1oJV0VnB70kXYLk8B0poXQjMIUvORPPzG/view?usp=drive_link

## Team

**TEAM:** EmbedNBreakfast

- Mauro Orazio DRAGO
- Francesco Maria MOSCA
- Simone COLECCHIA
- Sajjad SHAFFAF
- Mohammad RASHID KHAN MOFRAD

## Repository structure

- `1. Preliminary analysis.ipynb`
- `Preliminary_analysis_Cuisine_classification_W2V_SR.ipynb` (variant/extended notebook)
- `2. Recipe embeddings clustering.ipynb`
- `3. Zero-Shot Classification.ipynb`
- `4. Fine-tuning BERT.ipynb`
- `5. GPT_2_fine_tuning.ipynb`
- `6. Vocal Assistant.ipynb`
- `GroupAssignment2025.pdf` (assignment specification)

## Dataset

All notebooks assume **RecipeNLG** in CSV format with list-like fields stored as strings.

- File name used throughout: `full_dataset.csv`
- Parsed columns (via `ast.literal_eval` converters): `ingredients`, `directions`, `ner`
- Several notebooks also use the `source` column to filter the **Recipes1M** subset (e.g., `df[df['source'] == 'Recipes1M']`).

The dataset is **not included** in this repo (size constraints). Place it somewhere accessible and update the notebook paths accordingly.

### Colab paths
Many notebooks are written for Google Colab and use paths like:

- `/content/drive/MyDrive/rcp-nlp/full_dataset.csv`

If you run locally, replace these with your local path.

## Environment / dependencies

The notebooks use Python data/NLP tooling including:

- Data/ML: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`
- Classical NLP: `gensim` (Word2Vec), `rank_bm25`
- Embeddings + clustering: `tensorflow_hub` (USE), `umap-learn`, `hdbscan`
- Transformers stack: `transformers`, `datasets`, `torch`, `accelerate`, `evaluate`
- Vector search: `faiss-cpu`, LangChain `FAISS`
- Voice assistant: `openai-whisper`, `gTTS`, `ffmpeg-python`

GPU is strongly recommended for transformer training/inference.

## Notebooks (what we implemented)

### 1) Preliminary analysis + cuisine classification + search + Word2Vec

Notebook: `1. Preliminary analysis.ipynb` (and the extended version `Preliminary_analysis_Cuisine_classification_W2V_SR.ipynb`).

**Cuisine label inference**
- `prepare_cuisine_data(...)` assigns cuisine labels using a keyword dictionary and removes very rare cuisines.

**Memory-optimized vocabulary features**
- `extract_vocabulary_features_memory_optimized(...)` computes vocabulary statistics and builds TF‑IDF features in batches.
- TF‑IDF configuration:
  - `max_features=3000`
  - `min_df=10`
  - `ngram_range=(1, 2)`
- Features are saved to disk as sparse `.npz` with companion label files under a `SaSha/cuisine_features/` folder.

**RandomForest cuisine classifier**
- `RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1)`
- Saved model artifact: `random_forest_classifier.joblib`

**Recipe search engine (BM25 / TF‑IDF / hybrid)**
- `RecipeSearchEngine` combines:
  - BM25 (`rank_bm25`)
  - TF‑IDF cosine similarity
  - Hybrid retrieval with a weighted score
- Model persistence under `SaSha/recipe_search_models/*.pkl`

**Word2Vec (domain embeddings)**
- Trained on recipes with `source == "Recipes1M"`
- Parameters:
  - `vector_size=300`, `window=5`, `min_count=5`, `sg=1`
- Saved under `SaSha/W2V_models/`

### 2) Recipe embeddings + clustering

Notebook: `2. Recipe embeddings clustering.ipynb`

- Sentence embeddings using **Universal Sentence Encoder** (TF Hub)
- Saved embeddings: `recipe_embeddings.h5`
- HDBSCAN clustering:
  - `min_cluster_size=30`
  - `cluster_selection_method='eom'`
- Visualization: t‑SNE plot saved as `hdbscan_clusters.png`
- Soft clustering / tuning via `BayesianGaussianMixture` (Dirichlet-process prior), with PCA/UMAP projection and silhouette-based comparison.

### 3) Zero-shot classification

Notebook: `3. Zero-Shot Classification.ipynb`

- Zero-shot classifier: `facebook/bart-large-mnli`
- Candidate labels:
  - `appetizer`, `main dish`, `side dish`, `dessert`, `drink`, `snack`
- Embedding visualization:
  - `SentenceTransformer("sentence-transformers/all-mpnet-base-v2")`
  - 3D UMAP + Plotly visualization (qualitative inspection)

### 4) Fine-tuning DistilBERT (MLM) with LoRA + embeddings + FAISS

Notebook: `4. Fine-tuning BERT.ipynb`

**Goal**: domain-adaptive masked-language-model fine-tuning on recipe text.

**Data preprocessing**
- A `text` column is created from recipe fields:
  - `TITLE: ... [SEP] INGREDIENTS: ... [SEP] DIRECTIONS: ...`
- Recipes1M-only subset is extracted from `full_dataset.csv` via `df[df['source'] == 'Recipes1M']` and saved (Colab Drive path in the notebook):
  - `/content/drive/MyDrive/rcp-nlp/recipes1m.csv`

**Model + training setup**
- Base model: `distilbert-base-uncased`
- Data collator: `DataCollatorForLanguageModeling(mlm_probability=0.15)`
- LoRA (PEFT) configuration:
  - `r=8`, `lora_alpha=32`, `lora_dropout=0.01`
  - `target_modules=['q_lin']`
- TrainingArguments:
  - `output_dir="./distilbert-mlm-lora-checkpoints"`
  - `num_train_epochs=5`
  - `per_device_train_batch_size=16`
  - `learning_rate=1e-3`
  - `eval_strategy="epoch"`, `save_strategy="epoch"`
  - `load_best_model_at_end=True`, `save_total_limit=2`
  - `logging_steps=100`, `fp16=True`
- Adapter output: `distilbert-mlm-lora-adapters/` (saved via `model.save_pretrained(...)`)

**Embedding extraction + similarity search**
- Token embedding matrix saved to `./bert_embeddings.pt`
- Filtered vocabulary embeddings saved to `./recipe_vocab_embeddings.pt`
- Recipe-level embeddings generated by averaging token vectors and stored efficiently as:
  - `recipe_emb_float32.npy` (memmap float32 matrix)
  - `recipe_keys.npy` (maps rows back to dataset items)
- FAISS cosine-similarity search implemented via L2-normalization + `faiss.IndexFlatIP`.

### 5) GPT-2 recipe generation (0/1/few-shot + fine-tuning)

Notebook: `5. GPT_2_fine_tuning.ipynb`

**Data loading**
- Loads a random sample from `full_dataset.csv` using row-skipping (seeded):
  - `SAMPLE_SIZE = 1_000_000`

**Fine-tuning format**
Recipes are serialized as a single text prompt per example:

```
Generate a new recipe with this TITLE: <title>

TITLE: <title>

INGREDIENTS:
- ...

DIRECTIONS:
1. ...

```

- Training file written to: `recipes_train.txt`

**Model + training**
- Base: `gpt2`
- Padding token set to EOS: `tokenizer.pad_token = tokenizer.eos_token`
- Tokenization block size used in the custom dataset: `block_size=750` (padding to max length)
- TrainingArguments:
  - `num_train_epochs=3`
  - `per_device_train_batch_size=8`
  - `fp16=True`
  - `save_strategy="no"`
- Saved model folder: `recipe_generator_model/`

**Evaluation**
For selected recipes, generation is evaluated with:
- Semantic similarity (SBERT cosine): `SentenceTransformer('all-MiniLM-L6-v2')`
- ROUGE‑1 F1 (`evaluate.load("rouge")`)
- METEOR (`evaluate.load("meteor")`)

Also includes a qualitative comparison of **zero-shot vs one-shot vs few-shot** prompting.

### 6) Voice Assistant (Whisper + FAISS RAG + Llama + gTTS)

Notebook: `6. Vocal Assistant.ipynb`

**Vector index (recipes knowledge base)**
- Embeddings: `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")`
- FAISS index folder: `faiss_index_folder/`
- If the index does not exist, it is built from a dataset sample (default in notebook):
  - `sample_size=10000`

**LLM (generation)**
- Model: `meta-llama/Llama-3.2-1B-Instruct`
- The notebook logs into Hugging Face using a Colab secret:
  - `HF_TOKEN` retrieved via `google.colab.userdata`

**RAG chain**
- Uses `RetrievalQA` with a custom prompt template (“You are SugoGPT...”) and `vectorstore.as_retriever()`.
- Also defines a baseline `LLMChain` without retrieval for comparison.

**Automatic evaluation (RAG vs No‑RAG)**
- Creates 30 question/answer pairs from the vectorstore documents.
- Metrics computed via `evaluate`:
  - BLEU, ROUGE, METEOR, BERTScore

**Voice interface (Colab-specific)**
- Records audio with an HTML/JS widget loaded from `AUDIO.html` (included in this repo; ensure the notebook's working directory contains it).
- Speech-to-text: Whisper (`base`)
- Text-to-speech: `gTTS`
- Audio conversion: `ffmpeg-python` (webm → wav)
- Conversation log file: `conversation_log.json`

> Note: the audio UI is designed for Colab. Running locally requires replacing the recording widget.

## Notes for GitHub / reproducibility

- Don’t commit `full_dataset.csv` (it’s large). Consider adding it to `.gitignore` when publishing.
- Some notebooks load/save artifacts under user-specific folders (e.g., `SaSha/`, Drive paths). If you rerun, you may want to standardize paths (e.g., a `data/` and `artifacts/` folder).
- `FAISS.load_local(..., allow_dangerous_deserialization=True)` is used in the voice assistant for convenience; only load indexes you trust.
