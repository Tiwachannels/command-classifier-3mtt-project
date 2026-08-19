"""
Trains the full 150-intent general-purpose command classifier on the
real CLINC-style dataset (English only, French dropped as out of scope
for the Nigerian-market use case).
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

DATA_PATH = "/home/claude/voice_command_classifier/data/full_150_intents_en.csv"
MODEL_PATH = "/home/claude/voice_command_classifier/models/full_150_pipeline.joblib"


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} examples, {df['intent'].nunique()} intents")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["intent"], test_size=0.2, random_state=42, stratify=df["intent"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, C=5.0)),
    ])

    print("Running 5-fold cross-validation (this covers 150 classes, may take a moment)...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, n_jobs=-1)
    print(f"Cross-val accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    print(f"\nOverall test accuracy: {report['accuracy']:.3f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.3f}")
    print(f"Weighted F1: {report['weighted avg']['f1-score']:.3f}")

    # Show the 10 worst-performing intents - honest signal of where this
    # model is weakest, e.g. semantically overlapping intents
    f1_by_intent = {k: v["f1-score"] for k, v in report.items()
                     if k not in ("accuracy", "macro avg", "weighted avg")}
    worst = sorted(f1_by_intent.items(), key=lambda x: x[1])[:10]
    print("\n10 lowest-F1 intents (likely confused with a semantically similar intent):")
    for intent, f1 in worst:
        print(f"  {intent}: {f1:.2f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
