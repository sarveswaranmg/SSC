from transformers import pipeline
from json import dumps
from flask import Flask, request, jsonify

# Initialize models
summarizer = pipeline("summarization")
sentiment_analyzer = pipeline("sentiment-analysis")
classifier = pipeline("zero-shot-classification")

def handler(request):
    data = request.json
    text = data['text']
    
    # Perform Summarization
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    
    # Perform Sentiment Analysis
    sentiment = sentiment_analyzer(text)
    
    # Perform Category Classification
    candidate_labels = ["Technology", "Business", "Health", "Entertainment"]
    category = classifier(text, candidate_labels)
    
    # Prepare output
    result = {
        "summary": summary[0]['summary_text'],
        "sentiment": sentiment[0]['label'],
        "category": category['labels'][0]
    }
    
    return dumps(result)
