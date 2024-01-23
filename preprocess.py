import re
from emoji import extract_emojis

def preprocess(text):
    # Split the input text into a list of words
    new_text = []

    for t in text.split(" "):
        # Remove mentions (e.g., @user)
        t = re.sub('@[A-Za-z0-9_]+', '', t)

        # Remove URLs (e.g., http://example.com)
        t = re.sub(r'http\S+', '', t)

        # Remove hashtags (e.g., #happy)
        t = re.sub('#', '', t)  # Remove only the '#' symbol

        # Extract emojis using the extract_emojis function
        emojis = extract_emojis(t)

        # Remove non-alphanumeric characters excluding emojis
        t = re.sub('[^A-Za-z0-9]+', '', t)

        # Append the cleaned word to the new_text list
        new_text.append(t + emojis)

    # Join the cleaned words into a single string
    cleaned_text = " ".join(new_text)

    # Remove consecutive spaces
    cleaned_text = re.sub(' +', ' ', cleaned_text)

    return cleaned_text.strip()  # Remove leading and trailing spaces

# Random input with mentions, URLs, special characters, and emojis
input_text = ""

# Apply the preprocess function
output_text = preprocess(input_text)

# Display the results
print("Original Text:")
print(input_text)
print("\nProcessed Text:")
print(output_text)
