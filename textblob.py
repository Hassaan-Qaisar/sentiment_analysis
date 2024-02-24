# checking another way to do sentiment analysis
from textblob import TextBlob
from dbconnection import data

positive_count = 0
negative_count = 0
neutral_count = 0

for sentence in data:
    blob = TextBlob(sentence)
    sentiment = blob.sentiment

    # Determine sentiment polarity
    polarity = sentiment.polarity

    # Classify tweets based on polarity
    if polarity > 0.2:
        positive_count += 1
    elif polarity < 0:
        negative_count += 1
    else:
        neutral_count += 1

# Print counts
print(f"Positive tweets: {positive_count}")
print(f"Negative tweets: {negative_count}")
print(f"Neutral tweets: {neutral_count}")
