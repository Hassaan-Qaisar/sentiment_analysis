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
for i, text in enumerate(texts):
    ranking = np.argsort(probabilities_batch[i])[::-1]
    print(f"\nResults for Text {i+1} - '{text}':")
    for j in range(probabilities_batch.shape[1]):
        label = config.id2label[ranking[j]]
        score = probabilities_batch[i, ranking[j]]
        print(f"{j+1}) {label}: {np.round(float(score), 4)}")


