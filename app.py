import random
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# -------------------------------------
# Load Trained Model Safely
# -------------------------------------
model = None
try:
    model = joblib.load("personality_pipeline.joblib")
    print("[INFO] Personality model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None
    print("[WARNING] Running without model. Predictions will be mocked.")

# -------------------------------------
# Question Bank (your existing code)
# -------------------------------------
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
        "I adjust my communication style to fit the group I'm in.",
        "I believe the best results come from collective effort.",
        "I am open to feedback and ideas from others.",
        "I try to resolve conflicts calmly and fairly.",
        "I share credit with my teammates for success.",
        "I believe everyone's voice matters in decisions.",
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
        "Your company suddenly faces a major unexpected challenge. What's your first step?",
        "A critical system breaks down overnight. What do you do first?",
        "Your manager assigns you an impossible deadline. How do you react?",
        "The client changes requirements last minute. What's your approach?",
        "You notice a major financial error. How do you act?",
        "A competitor launches a disruptive product. Your first move?",
        "Your company faces negative media attention. What's your reaction?",
        "An unexpected crisis threatens your project. First step?",
        "Sudden budget cuts affect your project. What will you do?",
        "Market conditions change drastically overnight. How do you respond?"
    ]
}

# -------------------------------------
# Routes
# -------------------------------------
@app.route("/")
def home():
    selected_questions = {cat: random.sample(qs, 4) for cat, qs in questions_bank.items()}
    selected_situational = {
        "Q21": random.choice(situational_bank["Q21"]),
        "Q22": random.choice(situational_bank["Q22"])
    }
    return render_template("index.html", core=selected_questions, situational=selected_situational)

@app.route("/submit", methods=["POST"])
def submit():
    try:
        data = request.json
        
        # If model is not loaded, return mock prediction
        if not model:
            mock_roles = ["Strategist", "Visionary", "Collaborator", "Explorer", "Stabilizer"]
            mock_role = random.choice(mock_roles)
            return jsonify({
                "category": mock_role,
                "confidence": "85%",
                "message": f"Mock result: You are most likely a {mock_role} (85% confidence). Model not loaded properly.",
                "scores": {
                    "Analytical": "75%",
                    "Risk-Taking": "60%", 
                    "Collaboration": "80%",
                    "Curiosity": "70%",
                    "Stability": "65%"
                }
            })

        # Extract Q1–Q20 Likert answers and aggregate by category
        core_sums = {
            "Analytical Thinking": 0,
            "Risk-Taking": 0,
            "Collaboration": 0,
            "Curiosity": 0,
            "Stability": 0
        }
        
        # Count questions per category to calculate averages
        question_counts = {
            "Analytical Thinking": 0,
            "Risk-Taking": 0,
            "Collaboration": 0,
            "Curiosity": 0,
            "Stability": 0
        }
        
        for cat, answers in data["core"].items():
            for answer in answers:
                core_sums[cat] += int(answer)
                question_counts[cat] += 1
        
        # Calculate average scores per category
        core_averages = {}
        for cat in core_sums:
            if question_counts[cat] > 0:
                core_averages[cat] = core_sums[cat] / question_counts[cat]
            else:
                core_averages[cat] = 0

        # Extract Q21 & Q22 text
        situational_text = " ".join(data["situational"])

        # Create the input data in the same format as during training
        # This matches the structure expected by the ColumnTransformer
        input_data = pd.DataFrame([{
            "core_Analytical": core_averages["Analytical Thinking"],
            "core_RiskTaking": core_averages["Risk-Taking"],
            "core_Collaboration": core_averages["Collaboration"],
            "core_Curiosity": core_averages["Curiosity"],
            "core_Stability": core_averages["Stability"],
            "situational_text": situational_text
        }])

        # Predict
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        confidence = round(np.max(probabilities) * 100, 2)

        return jsonify({
            "category": prediction,
            "confidence": f"{confidence}%",
            "message": f"You are most likely a {prediction} ({confidence}% confidence).",
            "scores": {
                "Analytical Thinking": f"{core_averages['Analytical Thinking']:.1f}/5.0",
                "Risk-Taking": f"{core_averages['Risk-Taking']:.1f}/5.0", 
                "Collaboration": f"{core_averages['Collaboration']:.1f}/5.0",
                "Curiosity": f"{core_averages['Curiosity']:.1f}/5.0",
                "Stability": f"{core_averages['Stability']:.1f}/5.0"
            }
        })

    except Exception as e:
        print("[ERROR]", str(e))
        import traceback
        print("[TRACEBACK]", traceback.format_exc())
        return jsonify({"error": "Submit failed", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)