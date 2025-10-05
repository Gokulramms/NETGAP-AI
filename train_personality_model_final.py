"""
train_personality_model_final.py
--------------------------------
Trains a production-ready personality-role classifier pipeline.
Outputs:
    personality_pipeline.joblib          # Main model
    personality_pipeline.joblib.meta.pkl # Metadata (classes, SBERT flag)
Usage:
    python train_personality_model_final.py
    python train_personality_model_final.py --data your_dataset.csv
    python train_personality_model_final.py --use-sbert
"""

import os
import argparse
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Optional SBERT support
USE_SBERT_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    USE_SBERT_AVAILABLE = True
except Exception:
    USE_SBERT_AVAILABLE = False

# ----------------------------
# Question-category mapping
# ----------------------------
QUESTION_TO_CATEGORY = {}
for i in range(1, 5):
    QUESTION_TO_CATEGORY[f"q{i}"] = "Analytical Thinking"
for i in range(5, 9):
    QUESTION_TO_CATEGORY[f"q{i}"] = "Risk-Taking"
for i in range(9, 13):
    QUESTION_TO_CATEGORY[f"q{i}"] = "Collaboration"
for i in range(13, 17):
    QUESTION_TO_CATEGORY[f"q{i}"] = "Curiosity"
for i in range(17, 21):
    QUESTION_TO_CATEGORY[f"q{i}"] = "Stability"

ROLE_LABELS = ["Stabilizer", "Explorer", "Collaborator", "Visionary", "Strategist"]

# ----------------------------
# Synthetic question banks
# ----------------------------
questions_bank = {
    "Analytical Thinking": [
        "I prefer solving problems step by step rather than jumping to conclusions.",
        "I enjoy analyzing data or patterns to make decisions.",
        "I rely more on logic than intuition when making choices.",
        "I break complex problems into smaller parts before solving them.",
        "I double-check facts before making decisions.",
        "I like creating structured frameworks for solving issues.",
        "I value accuracy over speed in decision-making.",
        "I prioritize reasoning over gut feelings.",
        "I enjoy debugging and finding root causes.",
        "I compare multiple solutions before deciding."
    ],
    "Risk-Taking": [
        "I feel comfortable making decisions even with uncertain outcomes.",
        "I see failure as a learning opportunity rather than a setback.",
        "I am willing to invest time/resources into ideas without guaranteed results.",
        "I take initiative even when the outcome is unpredictable.",
        "I prefer trying new things over sticking to safe paths.",
        "I am not afraid of rejection when testing bold ideas.",
        "I trust my instincts in high-pressure situations.",
        "I encourage experimentation even if it may fail.",
        "I believe growth comes from taking risks.",
        "I act quickly when opportunities appear."
    ],
    "Collaboration": [
        "I enjoy working in teams more than working alone.",
        "I adjust my communication style to fit the group I’m in.",
        "I believe the best results come from collective effort.",
        "I am open to feedback and ideas from others.",
        "I try to resolve conflicts calmly and fairly.",
        "I share credit with my teammates for success.",
        "I believe everyone’s voice matters in decisions.",
        "I actively listen during group discussions.",
        "I encourage quieter teammates to share ideas.",
        "I celebrate team achievements enthusiastically."
    ],
    "Curiosity": [
        "I enjoy exploring new fields, even outside my expertise.",
        "I often ask 'why' and 'how' when learning something new.",
        "I actively look for new challenges to grow my skills.",
        "I experiment with new tools, methods, or techniques often.",
        "I keep reading or researching after solving a doubt.",
        "I like questioning existing rules or systems.",
        "I seek diverse perspectives before forming opinions.",
        "I try out hobbies outside of my comfort zone.",
        "I enjoy finding out how things work behind the scenes.",
        "I often learn from unrelated domains and apply ideas."
    ],
    "Stability": [
        "I prefer structured plans over spontaneous actions.",
        "I feel more comfortable when routines are predictable.",
        "I value consistency over frequent change.",
        "I prefer security and clarity over taking unnecessary risks.",
        "I like working with well-defined processes.",
        "I stay calm in stressful situations by following routines.",
        "I focus on long-term stability over short-term excitement.",
        "I prefer gradual improvements over sudden big changes.",
        "I find comfort in predictable schedules.",
        "I adapt slowly but steadily to changes."
    ]
}

# Situational banks (Q21 and Q22)
situational_bank = {
    "Q21": [
        "A teammate strongly disagrees with your solution. How would you respond?",
        "You propose an idea but the team dismisses it. What would you do?",
        "A conflict arises in your group project. How do you resolve it?",
        "Your suggestion is criticized harshly. How do you react?",
        "A colleague refuses to cooperate. How do you handle it?",
        "You and your peer have opposite views on implementation. Next step?",
        "A teammate accuses you of dominating discussions. Your response?",
        "Team deadlines are clashing with personal ideas. How do you balance?",
        "Your group wants to follow an inefficient method. Your action?",
        "How do you ensure collaboration when ideas clash?"
    ],
    "Q22": [
        "Your company suddenly faces a major unexpected challenge. What’s your first step?",
        "A critical system breaks down overnight. What do you do first?",
        "Your manager assigns you an impossible deadline. How do you react?",
        "The client changes requirements last minute. What’s your approach?",
        "You notice a major financial error. How do you act?",
        "A competitor launches a disruptive product. Your first move?",
        "Your company faces negative media attention. What’s your reaction?",
        "An unexpected crisis threatens your project. First step?",
        "Sudden budget cuts affect your project. What will you do?",
        "Market conditions change drastically overnight. How do you respond?"
    ]
}

# ----------------------------
# Helper functions
# ----------------------------
def aggregate_cores_from_qs(row):
    cat_sums = {cat: 0 for cat in ["Analytical Thinking", "Risk-Taking", "Collaboration", "Curiosity", "Stability"]}
    for qk, cat in QUESTION_TO_CATEGORY.items():
        val = row.get(qk, 0)
        try:
            val = float(val)
        except Exception:
            val = 0.0
        cat_sums[cat] += val
    return cat_sums

def combine_situational_text(row):
    s1 = row.get("situational_q21", "")
    s2 = row.get("situational_q22", "")
    return (str(s1) + " " + str(s2)).strip()

def create_synthetic_dataset(n_per_role=400, noise=0.4):
    rows = []
    for role in ROLE_LABELS:
        for _ in range(n_per_role):
            q_vals = {}
            for q in range(1, 21):
                qkey = f"q{q}"
                q_vals[qkey] = np.random.randint(1, 6)

            if role == "Strategist":
                favored = "Analytical Thinking"
            elif role == "Visionary":
                favored = "Risk-Taking"
            elif role == "Collaborator":
                favored = "Collaboration"
            elif role == "Explorer":
                favored = "Curiosity"
            else:
                favored = "Stability"

            for qk, cat in QUESTION_TO_CATEGORY.items():
                if cat == favored:
                    q_vals[qk] = min(5, max(3, int(np.random.normal(4.3, noise))))
                else:
                    q_vals[qk] = min(5, max(1, int(np.random.normal(3.0, noise + 0.3))))

            s1 = random.choice(questions_bank[favored])
            s2 = random.choice(random.choice(list(questions_bank.values())))
            rows.append({
                **q_vals,
                "situational_q21": s1 + " " + random.choice(situational_bank["Q21"]),
                "situational_q22": s2 + " " + random.choice(situational_bank["Q22"]),
                "label": role
            })
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# ----------------------------
# Training function
# ----------------------------
def build_and_train(df, out_path="personality_pipeline.joblib", use_sbert=False):
    numeric_df = df.apply(lambda r: pd.Series(aggregate_cores_from_qs(r)), axis=1)
    df_numeric = numeric_df.rename(columns={
        "Analytical Thinking": "core_Analytical",
        "Risk-Taking": "core_RiskTaking",
        "Collaboration": "core_Collaboration",
        "Curiosity": "core_Curiosity",
        "Stability": "core_Stability"
    })
    df["situational_text"] = df.apply(combine_situational_text, axis=1)
    X = pd.concat([df_numeric, df["situational_text"]], axis=1)
    y = df["label"].astype(str)

    numeric_transform = Pipeline([("scaler", StandardScaler())])
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=6000)

    if use_sbert and USE_SBERT_AVAILABLE:
        print("[INFO] Using SBERT embeddings.")
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        sbert_transformer = FunctionTransformer(lambda X: sbert.encode(X.ravel().tolist(), convert_to_numpy=True), validate=False)
        preproc = ColumnTransformer([
            ("num", numeric_transform, ["core_Analytical", "core_RiskTaking", "core_Collaboration", "core_Curiosity", "core_Stability"]),
            ("tfidf", tfidf, "situational_text"),
            ("sbert", sbert_transformer, "situational_text")
        ], remainder="drop", sparse_threshold=0)
    else:
        print("[INFO] Using TF-IDF only for text.")
        preproc = ColumnTransformer([
            ("num", numeric_transform, ["core_Analytical", "core_RiskTaking", "core_Collaboration", "core_Curiosity", "core_Stability"]),
            ("tfidf", tfidf, "situational_text")
        ], remainder="drop", sparse_threshold=0)

    estimators = [
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ("rf", RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1)),
        ("svc", LinearSVC(class_weight="balanced", max_iter=5000))
    ]

    stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=2000), n_jobs=-1, cv=5)
    pipeline = Pipeline([("pre", preproc), ("clf", stacking)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, stratify=y, random_state=42)
    param_grid = {"clf__final_estimator__C": [0.5, 1.0, 3.0]}
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1, verbose=1)

    print("[INFO] Starting training (GridSearchCV)...")
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    y_pred = best.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Test F1 (weighted):", f1_score(y_test, y_pred, average="weighted"))
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(best, out_path)
    meta = {"classes": list(best.classes_), "use_sbert": use_sbert and USE_SBERT_AVAILABLE}
    joblib.dump(meta, out_path + ".meta.pkl")
    print(f"[INFO] Saved model → {out_path}")
    print(f"[INFO] Saved metadata → {out_path}.meta.pkl")

    return best

# ----------------------------
# CLI entry
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="CSV path with labeled examples (optional).")
    parser.add_argument("--out", type=str, default="personality_pipeline.joblib", help="Output pipeline path")
    parser.add_argument("--use-sbert", action="store_true", help="Use SBERT if available.")
    args = parser.parse_args()

    if args.data and os.path.exists(args.data):
        df = pd.read_csv(args.data)
    else:
        print("[INFO] No dataset provided — generating synthetic dataset.")
        df = create_synthetic_dataset(n_per_role=300)

    build_and_train(df, out_path=args.out, use_sbert=args.use_sbert)
    print("[DONE] Training completed.")

if __name__ == "__main__":
    main()
