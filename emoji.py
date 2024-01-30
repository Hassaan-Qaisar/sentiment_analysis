import re

def extract_emojis(text):
    # Define the regex pattern to match emojis
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
    # Find all emojis in the input text using the regex pattern
    emojis_found = emoji_pattern.findall(text)
    
    # Join the found emojis into a single string
    return ''.join(emojis_found)

# Example text containing emojis
text_with_emojis = "I love Python! 😊🚀"

# Apply the extract_emojis function
extracted_emojis = extract_emojis(text_with_emojis)

# Print the original text and extracted emojis
print("Original Text:")
print(text_with_emojis)
print("\nExtracted Emojis:")
print(extracted_emojis)