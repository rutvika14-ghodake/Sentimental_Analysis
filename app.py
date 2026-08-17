# app.py
from flask import Flask, render_template_string, request
import pickle
import os

app = Flask(__name__)

# Load the vectorizer and model
def load_models():
    try:
        with open("cv.pkl", "rb") as f:
            cv = pickle.load(f)
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        return cv, model
    except Exception as e:
        print(f"Error loading pickle files: {e}")
        return None, None

vectorizer, model = load_models()

# HTML & CSS Template inside app.py
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis Portal</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            min-height: 100vh;
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            color: #f8fafc;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            width: 100%;
            max-width: 650px;
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 40px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(to right, #a855f7, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }

        .header p {
            color: #94a3b8;
            font-size: 0.95rem;
        }

        .form-group {
            margin-bottom: 24px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            color: #cbd5e1;
            font-size: 0.9rem;
            font-weight: 600;
            letter-spacing: 0.5px;
        }

        textarea {
            width: 100%;
            height: 140px;
            padding: 16px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            color: #f8fafc;
            font-size: 1rem;
            resize: none;
            outline: none;
            transition: all 0.3s ease;
        }

        textarea:focus {
            border-color: #a855f7;
            box-shadow: 0 0 12px rgba(168, 85, 247, 0.3);
        }

        textarea::placeholder {
            color: #64748b;
        }

        .btn {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
            border: none;
            border-radius: 12px;
            color: white;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(236, 72, 153, 0.3);
        }

        .btn:active {
            transform: translateY(0);
        }

        .result-box {
            margin-top: 30px;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 1.2rem;
            font-weight: 600;
            animation: fadeIn 0.4s ease-in-out;
        }

        .positive {
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.4);
            color: #4ade80;
        }

        .negative {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #f87171;
        }

        .neutral {
            background: rgba(234, 179, 8, 0.15);
            border: 1px solid rgba(234, 179, 8, 0.4);
            color: #facc15;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Sentiment Analysis</h1>
            <p>Analyze text emotional tone instantly using ML</p>
        </div>
        <form method="POST" action="/predict">
            <div class="form-group">
                <label for="text">ENTER YOUR TEXT</label>
                <textarea id="text" name="text" placeholder="Type or paste text here to analyze sentiment..." required>{{ text }}</textarea>
            </div>
            <button type="submit" class="btn">Analyze Sentiment</button>
        </form>

        {% if prediction %}
            <div class="result-box {{ sentiment_class }}">
                Result: {{ prediction }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_LAYOUT, text="", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    if not vectorizer or not model:
        return render_template_string(
            HTML_LAYOUT, 
            text="", 
            prediction="Model files (cv.pkl / model.pkl) not found!", 
            sentiment_class="neutral"
        )
    
    text = request.form.get("text", "")
    if text:
        # Transform input text using vectorizer
        transformed_text = vectorizer.transform([text])
        prediction_val = model.predict(transformed_text)[0]

        # Handle numerical or string predictions
        if str(prediction_val).lower() in ["1", "positive"]:
            prediction = "Positive Sentiment 😊"
            sentiment_class = "positive"
        elif str(prediction_val).lower() in ["0", "negative"]:
            prediction = "Negative Sentiment 😞"
            sentiment_class = "negative"
        else:
            prediction = f"Sentiment: {prediction_val}"
            sentiment_class = "neutral"

        return render_template_string(
            HTML_LAYOUT, 
            text=text, 
            prediction=prediction, 
            sentiment_class=sentiment_class
        )

    return render_template_string(HTML_LAYOUT, text="", prediction=None)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
