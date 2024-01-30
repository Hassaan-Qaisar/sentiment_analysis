import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Function to remove stopwords from text
def remove_stopwords(text):
    # Get the set of English stopwords
    stop_words = set(stopwords.words('english'))
    # Tokenize the input text into words
    word_tokens = word_tokenize(text)
    # Create a new list of words excluding stopwords
    filtered_text = [word for word in word_tokens if word not in stop_words]
    # Join the filtered words into a single string
    return ' '.join(filtered_text)

# Join the filtered words into a single string
input_text = ""
# Join the filtered words into a single string
output_text = remove_stopwords(input_text)

# Print the original and processed text
print("Original Text:")
print(input_text)
print("\nProcessed Text:")
print(output_text)
