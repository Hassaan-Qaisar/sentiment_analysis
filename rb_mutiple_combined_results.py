from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from dbconnection import texts
from preprocess import preprocess

def analyze_sentiment_batch(texts, model, tokenizer):
    preprocessed_texts = [preprocess(text) for text in texts]
    encoded_inputs = tokenizer(preprocessed_texts, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**encoded_inputs)
    logits = outputs.logits
    probabilities = softmax(logits.detach().numpy(), axis=1)
    return probabilities



MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

text = texts

# Analyze sentiment for the batch of texts
probabilities_batch = analyze_sentiment_batch(text, model, tokenizer)

# Print cumulative results
cumulative_results = np.mean(probabilities_batch, axis=0)
print("\nCumulative Results:")
ranking = np.argsort(cumulative_results)[::-1]
for i in range(cumulative_results.shape[0]):
    label = config.id2label[ranking[i]]
    score = cumulative_results[ranking[i]]
    print(f"{i+1}) {label}: {np.round(float(score), 4)}")

