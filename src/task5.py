import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback


# evaluation helper (adapted from task 3)
def evaluate_transformer(y_test, y_pred, classes, label, save_path=None):
    print(f"\n{'='*50}\n{label}\n{'='*50}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Macro F1:  {f1_score(y_test, y_pred, average='macro'):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=classes, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title(label)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# main training pipeline to avoid duplicate code for multiclass/binary
def run_finetuning(df_subset, label_mapping, model_name, run_title, save_name):
    # map string labels to integers
    df_subset['label'] = df_subset['Sentiment'].map(label_mapping)

    # 80/20 split using raw 'Text', not 'Processed_Text' for transformers
    X_train, X_test, y_train, y_test = train_test_split(
        df_subset['Text'].tolist(), df_subset['label'].tolist(),
        test_size=0.2, stratify=df_subset['label'], random_state=42
    )

    # tokenize texts via Hugging Face datasets
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, max_length=128)

    train_dataset = Dataset.from_dict({'text': X_train, 'labels': y_train}).map(tokenize_fn, batched=True)
    test_dataset  = Dataset.from_dict({'text': X_test,  'labels': y_test }).map(tokenize_fn, batched=True)

    # load pre-trained model
    num_labels = len(label_mapping)
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

    # training arguments (hyperparameters)
    training_args = TrainingArguments(
        output_dir=f'./results_transformer/{save_name}',
        num_train_epochs=3,                  # 3 epochs is standard
        per_device_train_batch_size=16,      # small batches to save memory
        per_device_eval_batch_size=16,
        eval_strategy="epoch",               # evaluate at the end of each epoch
        save_strategy="epoch",
        load_best_model_at_end=True,         # load the best model
        learning_rate=2e-5,
        report_to="none"                     # disables online logging tools
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,                 # enables dynamic padding per batch
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]  # stop if no improvement
    )

    print(f"\nTraining {run_title}...")
    trainer.train()

    # predict and evaluate
    preds = trainer.predict(test_dataset)
    y_pred = np.argmax(preds.predictions, axis=1)

    classes = list(label_mapping.keys())
    evaluate_transformer(y_test, y_pred, classes, run_title, save_path=f'results/cm_transformer_{save_name}.png')



if __name__ == '__main__':
    # load data
    df = pd.read_json('processed.json', orient='records', lines=True)

    # 1. Multiclass Classification
    mapping_multi = {'negative': 0, 'neutral': 1, 'positive': 2}
    run_finetuning(df.copy(), mapping_multi, 'distilbert-base-uncased', 'DistilBERT (Multiclass)', 'multi')

    # 2. Binary Classification (drop neutral)
    df_binary = df[df['Sentiment'] != 'neutral'].copy()
    mapping_binary = {'negative': 0, 'positive': 1}
    run_finetuning(df_binary, mapping_binary, 'distilbert-base-uncased', 'DistilBERT (Binary)', 'binary')