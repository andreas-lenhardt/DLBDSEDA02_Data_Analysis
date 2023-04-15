import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


#%% 1 Data Loading and Preparing
df = pd.read_csv("CDPH_Environmental_Complaints.csv")

# Remove rows with no 'complaint detail'
df = df.dropna(subset=["COMPLAINT DETAIL"])

# Remove duplicate rows according to the 'complaint id'
df.drop_duplicates(
    subset=["COMPLAINT ID"], keep="last", ignore_index=True, inplace=True
)

# Convert 'complaint date' column to datetime
df["COMPLAINT DATE"] = pd.to_datetime(df["COMPLAINT DATE"])

# Remove rows depending on the year of the complaint
df = df[df["COMPLAINT DATE"].dt.year >= 2013]

# Copy selected columns to a new dataframe
df2 = df[
    ["COMPLAINT ID", "COMPLAINT TYPE", "COMPLAINT DETAIL", "COMPLAINT DATE"]
].copy()

# Shorten column namens
cols = ["id", "type", "complaint", "date"]
df2.columns = cols

# Recreate the DataFrame index
df2.reset_index(drop=True, inplace=True)


#%% 2.1 Text Cleaning
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
    #stop_words = stopwords.words('english')
    complaint = [i for i in complaint if i not in stop_words]
    return complaint

def lemmatize_complaints(complaint):
    complaint = [lemmatizer.lemmatize(i) for i in complaint]
    return complaint


df2.loc[:, "complaint"] = df2["complaint"].apply(lambda x: lower_complaint(x))

df2.loc[:, "complaint"] = df2["complaint"].apply(lambda x: remove_punctuation(x))

df2.loc[:, "complaint"] = df2["complaint"].apply(lambda x: remove_numbers(x))

df2['cP_list'] = df2['complaint'].str.split()
stop_words = stopwords.words('english')
df2.loc[:, "cP_list"] = df2["cP_list"].apply(lambda x: remove_stopwords(x, stop_words))

lemmatizer = WordNetLemmatizer()
df2.loc[:, "cP_list"] = df2["cP_list"].apply(lambda x: lemmatize_complaints(x))

df2.loc[:, 'cP_list'] = df2['cP_list'].apply(lambda x: ' '.join(x))


#%% 2.2 Feature Extraction (BoW, TF-IDF)
# vect_bow = CountVectorizer(ngram_range=(1,1),
#                             max_features=100,
#                            # min_df=1,
#                            # max_df=1.0
#                            )
# matrix_bow = vect_bow.fit_transform(df2["cP_list"])
# data_bow = pd.DataFrame(matrix_bow.toarray(),
#                         columns=vect_bow.get_feature_names_out())


vect_tfidf = TfidfVectorizer(ngram_range=(1, 1),
                             max_features=1000,
                             # min_df=0.01,
                             # max_df=0.95
                             )
matrix_tfidf = vect_tfidf.fit_transform(df2["cP_list"])
data_tfidf = pd.DataFrame(matrix_tfidf.toarray(),
                          columns=vect_tfidf.get_feature_names_out())

data_tfidf.to_csv("test.csv", sep=";")


#%% 2.3 Topic Modeling (LSA, LDA)
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(matrix_tfidf)

print(lsa_top)
print(lsa_top.shape)  # (no_of_doc*no_of_topics)

l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
    print("Topic ", i ," : ", topic*100)




print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)




# most important words for each topic
vocab = vect_tfidf.get_feature_names_out()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic " + str(i) + ": ")
    for t in sorted_words:
        print(t[0], end=" ")
    print("\n")


