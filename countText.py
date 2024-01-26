from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from dbconnection import data
from preprocess import preprocess
import time

def analyze_sentiment_batch(texts, model, tokenizer):
    preprocessed_texts = [preprocess(text) for text in texts]
    encoded_inputs = tokenizer(preprocessed_texts, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**encoded_inputs)
    logits = outputs.logits
    probabilities = softmax(logits.detach().numpy(), axis=1)
    return probabilities

# Start the timer
start_time = time.time()

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

texts = data

# Analyze sentiment for the batch of texts
probabilities_batch = analyze_sentiment_batch(texts, model, tokenizer)



# Stop the timer
end_time = time.time()

# Print the time taken
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")