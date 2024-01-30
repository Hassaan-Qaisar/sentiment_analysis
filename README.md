# Sentiment Analysis with RoBERTa Base Model

This project utilizes the RoBERTa base model for sentiment analysis and includes various optimization techniques to improve efficiency. The sentiment analysis is performed on a collection of texts, and the project consists of several Python scripts for different aspects of the analysis.

## Introduction

Sentiment analysis is a natural language processing task that involves determining the sentiment conveyed in a piece of text. In this project, we leverage the RoBERTa base model, a powerful transformer-based model, for sentiment analysis. Several preprocessing techniques and optimizations are applied to enhance the model's efficiency.

## Dependencies

Make sure to install the following dependencies before running the scripts:

- Python 3.x
- Required Python packages (install using `pip install -r requirements.txt`):
  - transformers
  - nltk
  - pymongo

## How to Run

1. Clone the repository:
   
```sh
git clone https://github.com/your-username/your-repo.git
```
   
2. Install the dependencies:

```sh
pip install -r requirements.txt
```  

3. Execute the desired script:


  ```bash
  python countText.py  # For counting positive, negative, and neutral texts
  python rb_mutiple_combined_results.py  # For cumulative probabilities of sentiment
  python rb_mutiple_individual_results.py  # For individual sentiment probabilities
  ```


## Code Files

- main.py: Original code implementing RoBERTa base model for sentiment analysis.
- dbconnection.py: Establishes a connection to MongoDB and fetches texts from the database.
- emoji.py: Utilizes regex to retain emojis in the text.
- stopwords.py: Removes stopwords from the text.
- preprocess.py: Applies various preprocessing steps, including removing mentions, hashtags, URLs, and special characters. It also incorporates emoji and stopwords functions.

## Python Scripts for Sentiment Analysis

- countText.py: Counts the number of positive, negative, and neutral texts.
- rb_mutiple_combined_results.py: Displays cumulative probabilities of sentiment for the entire dataset.
- rb_mutiple_individual_results.py: Displays individual sentiment probabilities for each text.

Feel free to explore and modify the code files according to your requirements.

