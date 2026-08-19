"""
Trains a TF-IDF + Logistic Regression baseline intent classifier and
evaluates it with a proper train/test split + cross-validation.
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = "/home/claude/voice_command_classifier/data/intents_dataset.csv"
MODEL_PATH = "/home/claude/voice_command_classifier/models/baseline_pipeline.joblib"


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} examples, {df['intent'].nunique()} intents")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["intent"], test_size=0.2, random_state=42, stratify=df["intent"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=5.0)),
    ])

    # 5-fold cross-validation on the training set to sanity-check stability
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    print(f"\nCross-val accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\n=== Held-out test set performance ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # quick manual sanity checks, including phrasing not seen verbatim in training
    print("\n=== Manual sanity checks ===")
    test_phrases = [
        "abeg I wan check how much money dey my line",
        "my calls keep dropping, network is terrible",
        "can you connect me to someone who can help",
        "I lost my phone please block the number now",
        "good morning",
    ]
    for phrase in test_phrases:
        pred = pipeline.predict([phrase])[0]
        proba = pipeline.predict_proba([phrase]).max()
        print(f"  '{phrase}' -> {pred} (confidence: {proba:.2f})")


if __name__ == "__main__":
    main()
