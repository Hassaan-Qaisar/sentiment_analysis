import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)

input_text = "Hey @user! Can't believe it's finally #Friday! 🎉 Let's celebrate this amazing day with some great food! #TGIF 😊"
output_text = remove_stopwords(input_text)

print("Original Text:")
print(input_text)
print("\nProcessed Text:")
print(output_text)
