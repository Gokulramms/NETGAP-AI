import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Fixed Survey Questions (1–5 Likert Scale)
# -------------------------
questions = {
    "Analytical Thinking": [
        "I prefer solving problems step by step rather than jumping to conclusions.",
        "I enjoy analyzing data or patterns to make decisions.",
        "I rely more on logic than intuition when making choices.",
        "I break complex problems into smaller parts before solving them."
    ],
    "Risk-Taking": [
        "I feel comfortable making decisions even with uncertain outcomes.",
        "I see failure as a learning opportunity rather than a setback.",
        "I am willing to invest time/resources into ideas without guaranteed results.",
        "I take initiative even when the outcome is unpredictable."
    ],
    "Collaboration": [
        "I enjoy working in teams more than working alone.",
        "I adjust my communication style to fit the group I’m in.",
        "I believe the best results come from collective effort.",
        "I am open to feedback and ideas from others."
    ],
    "Curiosity": [
        "I enjoy exploring new fields, even outside my expertise.",
        "I often ask 'why' and 'how' when learning something new.",
        "I actively look for new challenges to grow my skills.",
        "I experiment with new tools, methods, or techniques often."
    ],
    "Stability": [
        "I prefer structured plans over spontaneous actions.",
        "I feel more comfortable when routines are predictable.",
        "I value consistency over frequent change.",
        "I prefer security and clarity over taking unnecessary risks."
    ]
}

# -------------------------
# Situational Questions
# -------------------------
situational_questions = [
    "A teammate strongly disagrees with your solution. How would you respond?",
    "Your company suddenly faces a major unexpected challenge. What’s your first step?"
]

# -------------------------
# Keywords for Situational Scoring (per category)
# -------------------------
keywords = {
    "Analytical Thinking": ["analyze", "data", "logic", "evidence", "step", "reasoning"],
    "Risk-Taking": ["risk", "uncertain", "bold", "decision", "initiative", "try"],
    "Collaboration": ["team", "together", "listen", "feedback", "discuss", "share"],
    "Curiosity": ["explore", "learn", "discover", "why", "how", "experiment"],
    "Stability": ["stable", "secure", "consistent", "routine", "predictable", "structured"]
}

# -------------------------
# Category Mapping Rules
# -------------------------
def determine_category(final_scores):
    sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    top1, top2 = sorted_scores[0], sorted_scores[1]

    # Single strong category
    if top1[1] - top2[1] >= 5:
        return {
            "Analytical Thinking": "Strategist",
            "Risk-Taking": "Visionary",
            "Collaboration": "Collaborator",
            "Curiosity": "Explorer",
            "Stability": "Stabilizer"
        }[top1[0]]

    # Combo categories
    combos = {
        ("Analytical Thinking", "Curiosity"): "Strategic Explorer",
        ("Analytical Thinking", "Risk-Taking"): "Calculated Visionary",
        ("Curiosity", "Risk-Taking"): "Innovative Explorer",
        ("Collaboration", "Stability"): "Reliable Team Builder",
        ("Analytical Thinking", "Stability"): "Practical Strategist",
    }

    key = tuple(sorted([top1[0], top2[0]]))
    return combos.get(key, f"{top1[0]} + {top2[0]} Thinker")


# -------------------------
# NLP Scoring Function
# -------------------------
def score_situational_answer(answer, keywords):
    scores = {cat: 0 for cat in keywords}
    for category, words in keywords.items():
        documents = [" ".join(words), answer]
        tfidf = TfidfVectorizer().fit_transform(documents)
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        scores[category] = round(sim * 10)  # 0–10 scale
    return scores

# -------------------------
# Run Survey
# -------------------------
def run_survey():
    print("=== NETGAP SURVEY ===")
    print("Answer on scale: 1=Strongly Disagree, 5=Strongly Agree\n")

    # Core scores
    core_scores = {cat: 0 for cat in questions}

    for category, qs in questions.items():
        print(f"\n-- {category} --")
        for q in qs:
            while True:
                try:
                    ans = int(input(f"{q} (1-5): "))
                    if 1 <= ans <= 5:
                        core_scores[category] += ans
                        break
                except:
                    pass
            print()

    # Save before situational
    print("\nScores before Situational:", core_scores)

    # Situational scores
    situational_total = {cat: 0 for cat in questions}
    for q in situational_questions:
        ans = input(f"\n{q}\nYour response: ")
        situational_scores = score_situational_answer(ans, keywords)
        for cat, val in situational_scores.items():
            situational_total[cat] += val

    print("\nSituational Contribution:", situational_total)

    # Final
    final_scores = {cat: core_scores[cat] + situational_total[cat] for cat in core_scores}
    print("\nFinal Scores:", final_scores)

    category = determine_category(final_scores)
    print(f"\nYour Mindset Category: {category}")


# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    run_survey()
