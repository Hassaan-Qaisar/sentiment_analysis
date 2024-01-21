from pymongo import MongoClient

# add your mongoDB url
client = MongoClient('')

# add your database collection name
db = client['']
collection = db['']

# condition if any
documents = collection.find({"": ""})

# Loading data from first 50 documents
texts = []
for i, doc in enumerate(documents):
    if i == 50:
        break
    texts.append(doc['tweet'])

# Print the length of the 'texts' array
print(f"Number of tweets: {len(texts)}")