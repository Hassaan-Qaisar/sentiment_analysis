import re
import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

start_time = time.time()

from dbconnection import data

analyzer = SentimentIntensityAnalyzer()

def preprocess_tweet(tweet):
    # Remove mentions
    tweet = re.sub(r'@[A-Za-z0-9_]+', '', tweet)
    # Remove HTTP links
    tweet = re.sub(r'http\S+', '', tweet)
    return tweet

# Define thresholds
positive_threshold = 0.05
negative_threshold = -0.05

# Initialize counters
positive_count = 0
negative_count = 0
neutral_count = 0

# Process each sentence
for sentence in data:
    # Preprocess tweet
    preprocessed_sentence = preprocess_tweet(sentence)
    # Analyze sentiment
    vs = analyzer.polarity_scores(preprocessed_sentence)
    # Classify tweet based on compound score
    compound_score = vs['compound']
    pos_score = vs['pos']
    neg_score = vs['neg']   
    if (compound_score > positive_threshold) and (pos_score > 0.2):
        positive_count += 1
    elif compound_score < negative_threshold:
        negative_count += 1
    else:
        neutral_count += 1

# Print counts
print("Number of Positive Tweets:", positive_count)
print("Number of Negative Tweets:", negative_count)
print("Number of Neutral Tweets:", neutral_count)

# Create a CSV file to write results
csv_file_path = 'vader_sentiment_results.csv'
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    # Create a CSV writer
    csv_writer = csv.writer(csvfile)

    # Write header
    csv_writer.writerow(["Text", "Compound", "Positive", "Neutral", "Negative"])

    # Process each sentence
    for sentence in data:
        # Preprocess tweet
        preprocessed_sentence = preprocess_tweet(sentence)
        # Analyze sentiment
        vs = analyzer.polarity_scores(preprocessed_sentence)

        print("{:<65} {}".format(preprocessed_sentence, str(vs)))

        # Write results to CSV
        csv_writer.writerow([preprocessed_sentence, vs['compound'], vs['pos'], vs['neu'], vs['neg']])

# Stop the timer
end_time = time.time()

# Print the time taken
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")
print(len(data))

