from textblob import TextBlob

text = "This is a great tutorial on sentiment analysis!"
blob = TextBlob(text)

sentiment = blob.sentiment
print(sentiment)
