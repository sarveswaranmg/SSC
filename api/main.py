from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Initialize models
summarizer = pipeline("summarization")
sentiment_analyzer = pipeline("sentiment-analysis")
classifier = pipeline("zero-shot-classification")

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data['text']
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    return jsonify({"summary": summary[0]['summary_text']})

@app.route('/sentiment', methods=['POST'])
def sentiment():
    data = request.json
    text = data['text']
    result = sentiment_analyzer(text)
    return jsonify({"sentiment": result[0]['label']})

@app.route('/category', methods=['POST'])
def category():
    data = request.json
    text = data['text']
    candidate_labels = ["Technology", "Business", "Health", "Entertainment"]
    result = classifier(text, candidate_labels)
    return jsonify({"category": result['labels'][0]})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
