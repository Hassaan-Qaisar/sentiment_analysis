# checking another way to do sentiment analysis
import csv
from textblob import TextBlob
from dbconnection import data

positive_count = 0
negative_count = 0
neutral_count = 0

results = []

for sentence in data:
    blob = TextBlob(sentence)
    sentiment = blob.sentiment

    # Determine sentiment polarity
    polarity = sentiment.polarity

    # Classify tweets based on polarity
    if polarity > 0.2:
        positive_count += 1
        results.append(("Positive", sentence))
    elif polarity < 0:
        negative_count += 1
        results.append(("Negative", sentence))
    else:
        neutral_count += 1
        results.append(("Neutral", sentence))

# Print counts
print(f"Positive tweets: {positive_count}")
print(f"Negative tweets: {negative_count}")
print(f"Neutral tweets: {neutral_count}")

# Write results to CSV file
csv_file_path = 'textblob_sentiment_results.csv'
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Sentiment", "Tweet"])
    csv_writer.writerows(results)

print("Results saved to CSV file:", csv_file_path)