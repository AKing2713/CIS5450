import json
import random
import pandas as pd

def random_sample(sample_fraction=0.1):
    """
    Generates a random sample (without replacement) of the data so we can parse it quicker than using the full 5GB file.
    The random seed is used so we can explore the same data.
    For now the default is 10%.
    The resulting DF is saved out to your working directory as a parquet for later use.

    Prams:
        sample_fraction(0.1): The percentage used to sample the data in decimal form (i.e. 0.1 = 10%)

    Returns:
        Null (Saves the file in working directory)
    """
    random.seed(42)
    
    sample = []
    
    with open("arxiv-metadata-oai-snapshot.json", "r") as f:
        for line in f:
            if random.random() < sample_fraction:
                sample.append(json.loads(line))
    
    df_sample = pd.DataFrame(sample)
    df_sample.to_parquet("./arxiv_sample.parquet", index=False)



import re
import string
import pandas as pd
import nltk

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# --------------------------------------------------
# 1. Download required NLTK resources
# --------------------------------------------------
nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")


# --------------------------------------------------
# 2. Helper objects
# --------------------------------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Optional: extend stopwords with arXiv/common academic filler terms
custom_stopwords = {
    "using", "used", "use", "show", "paper", "result", "results",
    "method", "methods", "model", "models", "approach", "based"
}
stop_words = stop_words.union(custom_stopwords)


# --------------------------------------------------
# 3. POS tag conversion for better lemmatization
# --------------------------------------------------
def get_wordnet_pos(treebank_tag):
    """
    Convert NLTK POS tags to WordNet POS tags for improved lemmatization.
    Default to noun if tag is unknown.
    """
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# --------------------------------------------------
# 4. Text cleaning + NLP preprocessing
# --------------------------------------------------
def preprocess_abstract(text, remove_numbers=True, min_token_len=3):
    """
    Preprocess a single abstract for NLP EDA.

    Steps:
    - lowercase
    - remove LaTeX/math-like markup and escaped characters
    - tokenize
    - remove punctuation
    - optionally remove numeric tokens
    - remove stopwords
    - remove short tokens
    - POS-tag + lemmatize

    Params:
        text: texts to process
        remove_numbers(True): Removes numbers from text if True
        min_token_len(3): Only counts a word if length is 3 or greater
    
    Returns:
        cleaned_tokens: list[str]
        cleaned_text: str
    """
    if pd.isna(text):
        return [], ""

    text = str(text).lower()

    # Remove common LaTeX-ish patterns / escaped sequences
    text = re.sub(r"\\[a-zA-Z]+", " ", text)      # \alpha, \beta, etc.
    text = re.sub(r"\$.*?\$", " ", text)          # inline math
    text = re.sub(r"\{|\}|\[|\]|\(|\)", " ", text)
    text = re.sub(r"\n|\r|\t", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize
    tokens = word_tokenize(text)

    # Remove tokens based on rules
    filtered_tokens = []
    for token in tokens:
        if remove_numbers and token.isdigit():
            continue
        if token in stop_words:
            continue
        if len(token) < min_token_len:
            continue
        if not token.isalpha():
            continue
        filtered_tokens.append(token)

    # POS tagging + lemmatization
    pos_tags = pos_tag(filtered_tokens)
    lemmatized_tokens = [
        lemmatizer.lemmatize(token, get_wordnet_pos(pos))
        for token, pos in pos_tags
    ]

    cleaned_text = " ".join(lemmatized_tokens)

    return lemmatized_tokens, cleaned_text


# --------------------------------------------------
# 5. Main function to apply to a dataframe
# --------------------------------------------------
def prepare_abstracts_for_nlp(df, id_col="id", abstract_col="abstract"):
    """
    Takes a dataframe containing arXiv metadata and returns a new dataframe
    with NLP-ready abstract fields while preserving the id for linking back
    to other metadata.

    Output columns:
    - id
    - abstract_raw
    - abstract_tokens_clean
    - abstract_clean
    - abstract_token_count

    Params:
        df: The DataFrame to apply the cleaning to
        id_col(id): The column name for the id column from the original DataFrame in order to link back to metadata
        abstract_col(abstract): The column name for the abstracts column

    Returns:
        DataFrame
    """
    working_df = df[[id_col, abstract_col]].copy()
    working_df = working_df.rename(columns={abstract_col: "abstract_raw"})

    processed = working_df["abstract_raw"].apply(preprocess_abstract)

    working_df["abstract_tokens_clean"] = processed.apply(lambda x: x[0])
    working_df["abstract_clean"] = processed.apply(lambda x: x[1])
    working_df["abstract_token_count"] = working_df["abstract_tokens_clean"].apply(len)

    return working_df
