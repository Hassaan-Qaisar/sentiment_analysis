import csv
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from dbconnection import data
import time
from preprocess import preprocess

'''
Description:
This script leverages the RoBERTa base model for sentiment analysis to categorize each text individually into
positive, negative, and neutral sentiments. It prints each text along with its corresponding sentiment probability.
'''

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

# Print cumulative results
for i, text in enumerate(texts):
    ranking = np.argsort(probabilities_batch[i])[::-1]
    print(f"\nResults for Text {i+1} - '{text}':")
    for j in range(probabilities_batch.shape[1]):
        label = config.id2label[ranking[j]]
        score = probabilities_batch[i, ranking[j]]
        print(f"{j+1}) {label}: {np.round(float(score), 4)}")

# Specify the CSV file path
csv_file_path = "sentiment_results_preprocess.csv"

# Open the CSV file in write mode
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    # Create a CSV writer
    csv_writer = csv.writer(csvfile)

    # Write header to the CSV file
    header = ["Tweet"] + [f"{label}" for label in config.id2label.values()]
    csv_writer.writerow(header)

    # Write results for each text to the CSV file
    for i, text in enumerate(texts):
        row_data = [text] + [f"{prob:.4f}" for prob in probabilities_batch[i]]
        csv_writer.writerow(row_data)

print(f"Results saved to '{csv_file_path}'")

end_time = time.time()

# Print the time taken
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")
