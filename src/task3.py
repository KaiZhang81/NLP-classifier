import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

# evaluation helper 
def evaluate(model, X_test_vec, y_test, label, save_path=None):
    y_pred = model.predict(X_test_vec)
    print(f"\n{'='*50}\n{label}\n{'='*50}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Macro F1:  {f1_score(y_test, y_pred, average='macro'):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title(label)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return y_pred

# ================================================================
# Phase 1
# ================================================================

# Load preprocessed data from Task 2
df = pd.read_json('processed.json', orient='records', lines=True)

# Sanity check
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}\n")

print("Class distribution (full dataset):")
print(df['Sentiment'].value_counts())
print(df['Sentiment'].value_counts(normalize=True).round(3), "\n")

# X = preprocessed text (string), y = sentiment label
X = df['Processed_Text']
y = df['Sentiment']

# 80/20 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,         # preserve class proportions in both splits
    random_state=42     # maintain reproducibility
)

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}\n")

print("Train class distribution:")
print(y_train.value_counts(normalize=True).round(3), "\n")

print("Test class distribution:")
print(y_test.value_counts(normalize=True).round(3))

# ================================================================
# Phase 2
# ================================================================

# Bag-of-Words
bow_vec = CountVectorizer()
X_train_bow = bow_vec.fit_transform(X_train)   # fit only on train
X_test_bow  = bow_vec.transform(X_test)        # transform on test

nb_bow = MultinomialNB()
nb_bow.fit(X_train_bow, y_train)
y_pred_bow = evaluate(nb_bow, X_test_bow, y_test,
                      "Naive Bayes + Bag-of-Words",
                      save_path='results/cm_nb_bow.png')

# TF-IDF
tfidf_vec = TfidfVectorizer()
X_train_tfidf = tfidf_vec.fit_transform(X_train)
X_test_tfidf  = tfidf_vec.transform(X_test)

nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = evaluate(nb_tfidf, X_test_tfidf, y_test,
                        "Naive Bayes + TF-IDF",
                        save_path='results/cm_nb_tfidf.png')

print(f"\nVocab size (BoW):    {len(bow_vec.vocabulary_)}")
print(f"Vocab size (TF-IDF): {len(tfidf_vec.vocabulary_)}")

# ================================================================
# Phase 3
# ================================================================

# MLP + Bag-of-Words
mlp_bow = MLPClassifier(
    hidden_layer_sizes=(128,),     # one hidden layer with 128 neurons
    max_iter=200,
    early_stopping=True,           # holds out 10% of train as validation; stops when val loss plateaus
    random_state=42,
)
mlp_bow.fit(X_train_bow, y_train)
y_pred_mlp_bow = evaluate(mlp_bow, X_test_bow, y_test,
                          "MLP + Bag-of-Words",
                          save_path='results/cm_mlp_bow.png')

# MLP + TF-IDF
mlp_tfidf = MLPClassifier(
    hidden_layer_sizes=(128,),
    max_iter=200,
    early_stopping=True,
    random_state=42,
)
mlp_tfidf.fit(X_train_tfidf, y_train)
y_pred_mlp_tfidf = evaluate(mlp_tfidf, X_test_tfidf, y_test,
                            "MLP + TF-IDF",
                            save_path='results/cm_mlp_tfidf.png')

# ================================================================
# Phase 4 
# ================================================================
errors_df = pd.DataFrame({
    'Original_Text':  df.loc[X_test.index, 'Text'].values,
    'Processed_Text': X_test.values,
    'True':           y_test.values,
    'Predicted':      y_pred_mlp_bow,        # best model from Phase 3
})
errors_df = errors_df[errors_df['True'] != errors_df['Predicted']].copy()
errors_df['Error_Type'] = errors_df['True'] + ' -> ' + errors_df['Predicted']

print("Misclassification breakdown:")
print(errors_df['Error_Type'].value_counts())

# ================================================================
# Phase 6
# ================================================================

binary_df = df[df['Sentiment'] != 'neutral'].copy()
print(f"\nBinary dataset size: {len(binary_df)}")
print("Binary class distribution:")
print(binary_df['Sentiment'].value_counts(normalize=True).round(3), "\n")

X_bin = binary_df['Processed_Text']
y_bin = binary_df['Sentiment']

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_bin, y_bin,
    test_size=0.2, stratify=y_bin, random_state=42,
)

# Re-fit vectorizers on the binary training data
bow_vec_bin = CountVectorizer()
X_train_bin_bow = bow_vec_bin.fit_transform(X_train_bin)
X_test_bin_bow  = bow_vec_bin.transform(X_test_bin)

tfidf_vec_bin = TfidfVectorizer()
X_train_bin_tfidf = tfidf_vec_bin.fit_transform(X_train_bin)
X_test_bin_tfidf  = tfidf_vec_bin.transform(X_test_bin)

# NB + BoW (binary)
nb_bow_bin = MultinomialNB()
nb_bow_bin.fit(X_train_bin_bow, y_train_bin)
evaluate(nb_bow_bin, X_test_bin_bow, y_test_bin,
         "BINARY: Naive Bayes + BoW",
         save_path='results/cm_bin_nb_bow.png')

# NB + TF-IDF (binary)
nb_tfidf_bin = MultinomialNB()
nb_tfidf_bin.fit(X_train_bin_tfidf, y_train_bin)
evaluate(nb_tfidf_bin, X_test_bin_tfidf, y_test_bin,
         "BINARY: Naive Bayes + TF-IDF",
         save_path='results/cm_bin_nb_tfidf.png')

# MLP + BoW (binary)
mlp_bow_bin = MLPClassifier(
    hidden_layer_sizes=(128,), max_iter=200,
    early_stopping=True, random_state=42,
)
mlp_bow_bin.fit(X_train_bin_bow, y_train_bin)
evaluate(mlp_bow_bin, X_test_bin_bow, y_test_bin,
         "BINARY: MLP + BoW",
         save_path='results/cm_bin_mlp_bow.png')

# MLP + TF-IDF (binary)
mlp_tfidf_bin = MLPClassifier(
    hidden_layer_sizes=(128,), max_iter=200,
    early_stopping=True, random_state=42,
)
mlp_tfidf_bin.fit(X_train_bin_tfidf, y_train_bin)
evaluate(mlp_tfidf_bin, X_test_bin_tfidf, y_test_bin,
         "BINARY: MLP + TF-IDF",
         save_path='results/cm_bin_mlp_tfidf.png')