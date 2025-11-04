import re
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bangla_stemmer.stemmer import stemmer
# os.chdir(os.getcwd())
from modules.slang_text import slang_text_dict
from modules.bangla_stopwords import bangla_stopwords

nltk.download('stopwords')

def replace_slang(text):
    """ Replace slang words in the input text with their standard forms."""
    for word in text.split():
        if word.lower() in slang_text_dict:
            text = text.replace(word, slang_text_dict[word])
    return text

def remove_punctuation(text):
    """ Remove punctuation from the input text."""
    punc = string.punctuation
    return text.translate(str.maketrans(' ', ' ', punc))

# Combine English and Bangla stopwords
mixed_stopwords = stopwords.words('english')
# print(f"Number of English Stopwords: {len(mixed_stopwords)}")
# print(f"Number of Bangla Stopwords: {len(bangla_stopwords)}")

mixed_stopwords.extend(bangla_stopwords)
# print(f"Total Number of Mixed Stopwords: {len(mixed_stopwords)}")

# function to remove stopwords
def remove_stopwords(text):
    new_text = []

    for word in text.lower().split():
        if word not in mixed_stopwords:
            new_text.append(word)
    
    return " ".join(new_text)

# function to remove numbers
def remove_numbers(text):
    return re.sub(r'\d+', '', text)

# function to remove bangla numbers
def remove_bangla_numbers(text):
    return re.sub(r'[\u09E6-\u09EF]', '', text)

# function to remove special characters
def remove_special_characters(text):
    pattern = re.compile(r'[-#@;:,।.!—?<>/\\|`~$%^&*()_=+{}\[\]\']')
    return pattern.sub('', text)

# helper function to recognizee language
def is_bangla(text):
    for ch in text:
        if '\u0980' <= ch <= '\u09FF':
            return True
    return False

# helper function to recognizee language
def is_english(text):
    for ch in text:
        if 'a' <= ch <= 'z' or 'A' <= ch <= 'Z':
            if not is_bangla(text):
                return True
    return False


# # Stemming Function
english_stemmer = PorterStemmer()
bangla_stemmer = stemmer.BanglaStemmer()

def stem_text(text, verbose=False):
    processed_text = []

    for word in text.split():
        if is_english(word):
            if verbose: print("Stemming English Word:", word)
            stemmed_word = english_stemmer.stem(word)
        elif is_bangla(word):
            if verbose: print("Stemming Bangla Word:", word)
            stemmed_word = bangla_stemmer.stem(word)
        else:
            stemmed_word = word
        processed_text.append(stemmed_word)
    
    return " ".join(processed_text)


# print(f"Number of English Stopwords: {len(mixed_stopwords)}")
# print(f"Number of Bangla Stopwords: {len(bangla_stopwords)}")
# print(f"Total Number of Mixed Stopwords: {len(mixed_stopwords)}")