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

# Count the number of positive, negative, and neutral texts
positive_count = 0
negative_count = 0
neutral_count = 0

for i in range(probabilities_batch.shape[0]):
    ranking = np.argsort(probabilities_batch[i])[::-1]
    top_label = config.id2label[ranking[0]]
    
    # Adjust threshold as needed
    threshold = 0.4
    
    if top_label == 'positive' and probabilities_batch[i, ranking[0]] > threshold:
        positive_count += 1
    elif top_label == 'negative' and probabilities_batch[i, ranking[0]] > threshold:
        negative_count += 1
    else:
        neutral_count += 1

# Print the results
print("\nNumber of Positive Tweets:", positive_count)
print("Number of Negative Tweets:", negative_count)
print("Number of Neutral Tweets:", neutral_count)

# Stop the timer
end_time = time.time()

# Print the time taken
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")