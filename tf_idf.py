import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from dbconnection import data
import time
from sklearn.feature_extraction.text import TfidfVectorizer

'''
Description:
Combining TF-IDF for impactful word extraction and RoBERTa base model for sentiment classification on Twitter data.
While leveraging TF-IDF for impactful word extraction improves time efficiency, it may lead to a drop in accuracy 
compared to using RoBERTa alone.
'''

def extract_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # Emoticons
                               u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # Transport & map symbols
                               u"\U0001F700-\U0001F77F"  # Alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric shapes
                               u"\U0001F800-\U0001F8FF"  # Miscellaneous symbols
                               u"\U0001F900-\U0001F9FF"  # Supplemental symbols and pictographs
                               u"\U0001FA00-\U0001FA6F"  # Extended-A
                               u"\U0001FA70-\U0001FAFF"  # Extended-B
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    return ''.join(emoji_pattern.findall(text)) 

def get_impactful_words(tweet, top_n=5):
    stop_words = set(stopwords.words('english'))
    new_text = []

    tweet = tweet.lower()
    tweet = re.sub('@[A-Za-z0-9_]+', '', tweet)    
    tweet = re.sub(r'http\S+', '', tweet)
    emojis = extract_emojis(tweet)

    tweet = re.sub('[^A-Za-z0-9]+', ' ', tweet)
    # Tokenize and remove stopwords
    words = [word.lower() for word in word_tokenize(tweet) if word.isalnum() and word.lower() not in stop_words]

    # Convert list of words to a string (required by TfidfVectorizer)
    processed_tweet = ' '.join(words)

    # Apply TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_tweet])
    
    # Get feature names and TF-IDF scores
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    
    # Get indices of top TF-IDF words
    top_tfidf_indices = tfidf_scores.argsort()[-top_n:][::-1]
    
    # Get top TF-IDF words
    top_tfidf_words = [feature_names[i] for i in top_tfidf_indices]

    # Convert the list of words to a string with spaces
    impactful_words_string = ' '.join(top_tfidf_words)

    new_text.append(impactful_words_string + " " + emojis)

    cleaned_text = " ".join(new_text)

    cleaned_text = re.sub(' +', ' ', cleaned_text)

    return cleaned_text

def analyze_sentiment_batch(texts, model, tokenizer):
    preprocessed_texts = [get_impactful_words(text) for text in texts]
    print(preprocessed_texts) 
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

# Count the number of positive, negative, and neutral tweets
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
