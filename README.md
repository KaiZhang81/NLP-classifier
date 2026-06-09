# Project Setup — Required Libraries

Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn torch transformers datasets accelerate
```

NLTK also needs one-time resource downloads (handled in `task2.py`):
- `punkt`, `punkt_tab` (tokenization)
- `wordnet` (lemmatization)
- `averaged_perceptron_tagger_eng` (POS tagging)

Run order: `task1.py` → `task2.py` → `task3.py` → `task4.py` → `task5.py`. Task 2 produces `processed.json` consumed by Tasks 3 and 4.

# Task 1 — Exploratory Data Analysis 

## What the task asks

Investigate dataset structure: class distribution, text length, vocabulary, frequent words. Visualize and interpret. Worth **15%**.

## What we did

- Loaded raw `Sentences_50Agree.txt` (4846 sentences, latin-1 encoding)
- Plotted class distribution (bar chart)
- Plotted word-count distribution per sentiment (histogram with KDE)
- Generated word clouds for positive and negative classes separately

## Key observations

- **Moderate class imbalance** (~59% neutral / 28% positive / 12% negative, 5:1 ratio) — motivates macro-F1 as the headline metric in Task 3.
- **Short sentences** (mostly 10–30 words) — limited context per example, harder classification task.
- **Raw word clouds dominated by stop words** ("the", "of", "and") — motivates the preprocessing pipeline in Task 2.

# Task 2 — Text Preprocessing

## What the task asks

Apply preprocessing to the financial news text and **justify the design decisions** — both what we did and what we deliberately did not do. Worth **10%**. Rubric rewards justification, not completeness.

## Dataset

- File: `FinancialPhraseBank-v1.0/Sentences_50Agree.txt`
- 4846 sentences, format: `text@label` (positive / neutral / negative)
- **Encoding: ISO-8859-1 (latin-1)**, not UTF-8

## Pipeline

```
lowercase
  → split letter-digit combos ("EUR3" → "EUR 3")
  → tokenize (NLTK word_tokenize)
  → numbers → "numtoken"
  → keep alphabetic + hyphenated tokens (drops punctuation)
  → POS-tag, then lemmatize with WordNet
  → drop stray single-char tokens
```

Output columns: `Tokens` (list) and `Processed_Text` (joined string).

## Decisions

| Step | Applied? | Why |
|---|---|---|
| Lowercasing | Yes | Vocab reduction; capitalization signal is weak for sentiment. |
| Number normalization → `numtoken` | Yes | Magnitude doesn't carry sentiment; raw numbers inflate vocab. |
| Punctuation removal | Yes | No signal in BoW on formal financial text. |
| Hyphenated word preservation | Yes | "Finnish-owned", "start-up", "year-on-year" are meaningful. |
| Letter-digit splitting | Yes | Recovers currency mentions like "EUR3" → `eur numtoken`. |
| POS-aware lemmatization | Yes | Without POS, "was → wa", "has → ha". With it, "was → be". |
| Stop-word removal | **No** | Default lists strip negation/direction words that flip sentiment. |
| Stemming | **No** | Lemmatization preferred — produces valid words, more interpretable. |
| Named-entity replacement | **No** | Companies/currencies act as sentiment context. |

# Task 3 — Sentiment Classification

## What the task asks

Build classical ML models with 80/20 train/test split, both BoW and TF-IDF representations, both Naive Bayes and a feed-forward NN, plus error analysis and a binary follow-up. Worth **35%**.

## Phases

Phase 1 — Load + split
Phase 2 — Naive Bayes (BoW + TF-IDF)
Phase 3 — Feed-forward NN (BoW + TF-IDF)
Phase 4 — Error analysis
Phase 5 — NB vs MLP comparison
Phase 6 — Binary classification

## Metrics

| Metric | Meaning |
|---|---|
| Accuracy | Fraction correct overall. Misleading on imbalanced data. |
| Precision (per class) | Of predictions for class X, how many were actually X. |
| Recall (per class) | Of true X examples, how many the model found. |
| F1 | Harmonic mean of precision and recall. |
| **Macro F1** | Unweighted mean of per-class F1. **Headline metric** — treats all classes equally. |
| Support | Number of true examples per class. |
| Confusion matrix | True labels vs predicted. Diagonal = correct, off-diagonal = errors. |

## Parameters used

- **`train_test_split`**: `test_size=0.2`, `stratify=y` (preserve class proportions), `random_state=42` (reproducibility)
- **`CountVectorizer()`** / **`TfidfVectorizer()`**: defaults. We already preprocessed in Task 2, so we don't re-enable their lowercase / stopword options.
- **`MultinomialNB()`**: defaults (`alpha=1.0` Laplace smoothing, `fit_prior=True` learns priors from training data)
- **`MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, early_stopping=True, random_state=42)`**: one hidden layer of 128 neurons. Early stopping is important — with ~6700 features and only ~3900 training examples, the network would overfit otherwise.

**Critical pattern:** vectorizer is `fit` on training data only, then `transform` on test. Fitting on test = data leak.

## Multiclass results

| Model | Accuracy | Macro F1 | Neg F1 | Neu F1 | Pos F1 |
|---|---|---|---|---|---|
| NB + BoW | 0.732 | 0.672 | 0.609 | 0.811 | 0.597 |
| NB + TF-IDF | 0.677 | 0.433 | 0.079 | 0.807 | 0.413 |
| **MLP + BoW** | **0.755** | **0.696** | 0.615 | 0.826 | 0.646 |
| MLP + TF-IDF | 0.742 | 0.682 | 0.606 | 0.818 | 0.623 |

Vocab size: **6747** tokens. Confusion matrices saved in `results/`.

### Key observations

1. **MLP + BoW is the overall best** (macro F1 = 0.696).
2. **NB + TF-IDF collapses to the majority class** (neutral recall = 99%, negative recall = 4%). TF-IDF reweighting + class imbalance lets the class prior dominate predictions.
3. **MLP fixes the TF-IDF problem.** Macro F1 jumped from 0.43 → 0.68 going from NB to MLP on the same TF-IDF features. MLPs learn class boundaries directly and don't rely on a strong prior.
4. **MLP is robust to representation choice** (BoW vs TF-IDF differs by only 0.014 macro F1). NB is the opposite — highly sensitive.
5. **Negative class is the weak spot across all models** (F1 ≈ 0.60–0.62 at best).
6. **Trade-off:** MLP needed noticeably more training time than NB for a ~2 pp macro F1 improvement over NB + BoW.

### Error analysis

Misclassification breakdown (238 errors of 970 test samples):

| Error type | Count |
|---|---|
| positive → neutral | 96 |
| neutral → positive | 58 |
| negative → neutral | 37 |
| neutral → negative | 19 |
| negative → positive | 17 |
| positive → negative | 11 |

**Headline finding:** errors concentrate on the *neutral axis* — 210 of 238 errors involve neutral. Direct positive ↔ negative confusion is rare (28). The model separates positive from negative well when it commits; the hard decision is whether sentiment is present at all.

### Binary classification (drop neutral)

Binary dataset: 1967 sentences. Class ratio: positive 69.3% / negative 30.7%.

| Model | Accuracy | Macro F1 | Neg F1 | Pos F1 |
|---|---|---|---|---|
| **NB + BoW** | 0.860 | **0.835** | 0.770 | 0.900 |
| NB + TF-IDF | 0.761 | 0.609 | 0.365 | 0.853 |
| MLP + BoW | 0.858 | 0.829 | 0.759 | 0.899 |
| MLP + TF-IDF | 0.840 | 0.806 | 0.725 | 0.887 |

### Key observations

1. **Huge improvement across all models** (+0.12 to +0.18 macro F1). Negative-class F1 jumped from ~0.61 in multiclass to ~0.77.
2. **NB + BoW slightly edges out MLP + BoW** in binary — when the task is cleaner, the simpler model is competitive. MLP's edge in multiclass came from handling noisier decisions.
3. **NB + TF-IDF prior dominance persists** but is milder (negative recall 22% vs 4% in multiclass). Confirms diagnosis: it's the *interaction* of TF-IDF reweighting with class imbalance, not severe imbalance alone.

# Task 4 — PMI-based Word Similarity

## What the task asks

Build word vectors using PMI (window=1) — **no external packages** for the core math. Pick 10 random words from the data and return their most similar words via cosine similarity. Worth **15%**.

## What we did

- Loaded preprocessed tokens from Task 2's `processed.json`
- **Pruned vocabulary** to words appearing ≥ 5 times (reduces matrix size and noise from rare words)
- Built **symmetric word-word co-occurrence matrix** (window=1, both directions counted)
- Computed **PMI**: `log₂( P(x,y) / (P(x)·P(y)) )`, clipped negatives to 0 → **PPMI**
- **L2-normalized** rows so that cosine similarity reduces to a dot product
- Picked 10 random words (`random.seed(20)` for reproducibility), returned top 5 similar by cosine similarity

## Notes

- Vectorized via numpy (`np.outer`, `np.dot`) — runs in seconds despite the V×V matrix
- PPMI clipping handles `log(0)` and noisy negatives from sparse co-occurrence
- "No external packages" rule respected — numpy used only for linear algebra, no gensim / sklearn similarity functions

# Task 5 — Fine-tuning Pre-trained Transformer

## What the task asks

Fine-tune one or two pre-trained transformer models, evaluate in both multiclass and binary settings, compare with classical ML approaches from previous tasks. Worth **25%**.

## What we did

- Selected **DistilBERT-base-uncased** (lightweight BERT variant, ~65M params, faster to fine-tune)
- Used the **raw `Text` column**, not `Processed_Text` — transformers have their own subword tokenizers, and preprocessing actively hurts them
- Same **80/20 stratified split** with `random_state=42` → identical test set as Task 3 for fair comparison
- Hyperparameters: `lr=2e-5`, `batch_size=16`, `epochs=3`, `max_length=128`, `early_stopping_patience=1`
- Repeated for both multiclass (3 labels) and binary (drop neutral, 2 labels)

## Results

### Multiclass

| Model | Accuracy | Macro F1 | Neg F1 | Neu F1 | Pos F1 |
|---|---|---|---|---|---|
| Best classical (MLP + BoW) | 0.755 | 0.696 | 0.615 | 0.826 | 0.646 |
| **DistilBERT** | **0.841** | **0.816** | **0.797** | **0.882** | **0.770** |

### Binary

| Model | Accuracy | Macro F1 | Neg F1 | Pos F1 |
|---|---|---|---|---|
| Best classical (NB + BoW) | 0.860 | 0.835 | 0.770 | 0.900 |
| **DistilBERT** | **0.952** | **0.944** | **0.923** | **0.965** |

### Key observations

1. **DistilBERT decisively beats classical** — +0.12 macro F1 in multiclass, +0.11 in binary.
2. **Biggest gains on the negative class** — multiclass negative F1 went **0.62 → 0.80** (+0.18). Self-attention captures negation and context that BoW destroys.
3. **Error mode flipped on negatives.** Classical models under-predicted negative (high precision, low recall). DistilBERT over-predicts slightly (recall 0.86 > precision 0.74).
4. **Binary near-perfect** (95.2% accuracy) — confirms neutral was the absorbing/hard class across the entire project.
5. **Early stopping fired in multiclass** — eval_loss rose at epoch 2, training stopped before overfitting.
6. **Trade-off:** classical NB trains in seconds; DistilBERT took ~12 min (multiclass) and ~5 min (binary) on CPU. ~12 pp macro F1 for orders of magnitude more compute.
