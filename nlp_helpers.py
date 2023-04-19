"""
The module "nlp_helpers" contains all necessary functions to generate
a Pandas DataFrame, which contains a finalized corpus.
The corpus can then be used for modeling the topics.
"""

import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation

#%% 1 Data Loading
def get_complaints(file="CDPH_Environmental_Complaints.csv", year=1900):
    """Prepares the passed csv file so that it can be further processed.

    Parameters
    ----------
    file : str
        The filename of the csv file. This should always be in the same
        directory as complaints.py.
        Default: "CDPH_Environmental_Complaints.csv".
    year : int
        Complaints are determined that are greater than or equal
        to the year passed.
        Default: 1900 (equals all rows).

    Returns
    -------
    DataFrame
        Populated Pandas DataFrame
    """
    df = pd.read_csv(file)

    print(f"Processing complaints from {year} and later...")

    # Remove rows with no/empty 'COMPLAINT DETAIL'
    df["COMPLAINT DETAIL"].replace('', np.nan, inplace=True)
    df.dropna(subset=["COMPLAINT DETAIL"], inplace=True)

    # Remove duplicate rows according to the 'COMPLAINT ID'
    df.drop_duplicates(
        subset=["COMPLAINT ID"],
        keep="last",
        ignore_index=True,
        inplace=True
    )

    # Convert 'complaint date' column to datetime
    df["COMPLAINT DATE"] = pd.to_datetime(df["COMPLAINT DATE"])

    # Remove rows depending on the year of the complaint
    df = df[df["COMPLAINT DATE"].dt.year >= year]

    # Copy selected columns to a new dataframe
    df_new = df[
        ["COMPLAINT ID",
         "COMPLAINT TYPE",
         "COMPLAINT DETAIL",
         "COMPLAINT DATE"]
    ].copy()

    # Shorten column namens and replace the original ones
    cols = [
        "id",
        "type",
        "complaint",
        "date"]
    df_new.columns = cols

    # Recreate the DataFrame index
    df_new.reset_index(drop=True, inplace=True)

    return df_new

#%% 2 Text Cleaning
def lower_complaint(complaint):
    complaint = complaint.lower()
    return complaint

def remove_punctuation(complaint):
    complaint = complaint.translate(str.maketrans("", "", string.punctuation))
    return complaint

def remove_numbers(complaint):
    complaint = re.sub("[0-9]", "", complaint)
    return complaint

def remove_stopwords(complaint, stop_words):
    stop_words = stopwords.words('english')
    complaint = [i for i in complaint if i not in stop_words]
    return complaint

def lemmatize_complaints(complaint, lemmatizer):
    complaint = [lemmatizer.lemmatize(i) for i in complaint]
    return complaint

def clean_text(df):
    print ("Cleaning the data...")
    df.loc[:, "complaint"] = df["complaint"].apply(
        lambda x: lower_complaint(x)
        )

    df.loc[:, "complaint"] = df["complaint"].apply(
        lambda x: remove_punctuation(x)
        )

    df.loc[:, "complaint"] = df["complaint"].apply(
        lambda x: remove_numbers(x)
        )

    df['complaints_list'] = df['complaint'].str.split()
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    df.loc[:, "complaints_list"] = df["complaints_list"].apply(
        lambda x: remove_stopwords(x, stop_words)
        )

    lemmatizer = WordNetLemmatizer()
    df.loc[:, "complaints_list"] = df["complaints_list"].apply(
        lambda x: lemmatize_complaints(x, lemmatizer)
        )

    df.loc[:, 'complaints_list'] = df['complaints_list'].apply(
        lambda x: ' '.join(x)
        )

    return df

#%% 3 Feature Extraction
def get_feature_data(complaints, mode=2):
    """Processes the text corpus using either BoW or TF-IDF.

    Parameters
    ----------
    complaints : list-like
        The corpus with the complaints.
    mode : int
        The mode that determines the technique according to which
        the features are extracted.
        1 => Bag of Words (BoW)
        2 => Term Frequency-Inverse Document Frequency (TF-IDF)
        Default: 2.

    Returns
    -------
    tuple
        Element #1: A vocabulary in vector form
        Element #2: A sparse feature matrix.
        Element #3: A Pandas DataFrame with the vocabulary and its values.
    """
    vector = None
    matrix = None
    values = None

    if mode == 1:
        print("Vectorizing the texts using BoW...")
        vector = CountVectorizer(ngram_range=(1,1),
                                 max_features=5000,
                                 # min_df=0.01,
                                 # max_df=0.95
                                 )
        matrix = vector.fit_transform(complaints)
        values = pd.DataFrame(matrix.toarray(),
                              columns=vector.get_feature_names_out())
    elif mode == 2:
        print("Vectorizing the text using TF-IDF...")
        vector = TfidfVectorizer(ngram_range=(1,1),
                                 max_features=5000,
                                 use_idf=True,
                                 smooth_idf=True,
                                 # min_df=0.01,
                                 # max_df=0.95
                                 )
        matrix = vector.fit_transform(complaints)
        values = pd.DataFrame(matrix.toarray(),
                              columns=vector.get_feature_names_out())

    return vector, matrix, values

#%% 4 Topic Modeling
def get_topics(vector, matrix, topics_number=5, words_per_topic=10, mode=1):
    def get_topics_with_lsa(vector, matrix, topics_number, words_per_topic):
        print(f"The following {topics_number} topics were identified by LSA:")
        model = TruncatedSVD(n_components=topics_number,
                             algorithm='randomized',
                             n_iter=10,
                             random_state=42
                             )

        model.fit_transform(matrix)

        vocabulary = vector.get_feature_names_out()

        for i, comp in enumerate(model.components_):
            vocabulary_comp = zip(vocabulary, comp)
            sorted_words = sorted(vocabulary_comp,
                                  key=lambda x:x[1],
                                  reverse=True
                                  )[:words_per_topic]
            print("Topic " + str(i+1) + ": ")
            for t in sorted_words:
                print(t[0], end=" ")
            print("\n")

    def get_topics_with_lda(vector, matrix, topics_number, words_per_topic):
        print(f"The following {topics_number} topics were identified by LDA:")
        print("\n")
        model = LatentDirichletAllocation(n_components=topics_number,
                                          learning_method="online",
                                          max_iter=1,
                                          random_state=42
                                          )

        model.fit_transform(matrix)

        vocabulary = vector.get_feature_names_out()

        for i, comp in enumerate(model.components_):
            vocabulary_comp = zip(vocabulary, comp)
            sorted_words = sorted(vocabulary_comp,
                                  key=lambda x:x[1],
                                  reverse=True
                                  )[:words_per_topic]
            print("Topic " + str(i+1) + ": ")
            for t in sorted_words:
                print(t[0], end=" ")
            print("\n")

    if mode == 1:
        get_topics_with_lsa(vector, matrix, topics_number, words_per_topic)
    elif mode == 2:
        get_topics_with_lda(vector, matrix, topics_number, words_per_topic)
    elif mode == 3:
        get_topics_with_lsa(vector, matrix, topics_number, words_per_topic)
        get_topics_with_lda(vector, matrix, topics_number, words_per_topic)
