import pandas as pd
import re
import string
from nltk.corpus import stopwords


#%% 1 Load and prepare the source data
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
df = df[df["COMPLAINT DATE"].dt.year >= 1900]

# Copy selected columns to a new dataframe
df2 = df[
    ["COMPLAINT ID", "COMPLAINT TYPE", "COMPLAINT DETAIL", "COMPLAINT DATE"]
].copy()

# Shorten column namens
cols = ["id", "type", "complaint", "date"]
df2.columns = cols

# Recreate the DataFrame index
df2.reset_index(drop=True, inplace=True)


#%% 2.1 Text cleaning
def lower_complaint(complaint):
    complaint = complaint.lower()
    return complaint

def remove_punctuation(complaint):
    complaint = complaint.translate(str.maketrans("", "", string.punctuation))
    return complaint

def remove_numbers(complaint):
    complaint = re.sub("[0-9]", "", complaint)
    return complaint

def remove_stopwords(complaint):
    stop_words = stopwords.words('english')
    complaint = [i for i in complaint if i not in stop_words]
    return complaint


df2.loc[:, "complaint"] = df2["complaint"].apply(lambda x: lower_complaint(x))

df2.loc[:, "complaint"] = df2["complaint"].apply(lambda x: remove_punctuation(x))

df2.loc[:, "complaint"] = df2["complaint"].apply(lambda x: remove_numbers(x))

df2['cP_list'] = df2['complaint'].str.split()
df2.loc[:, "cP_list"] = df2["cP_list"].apply(lambda x: remove_stopwords(x))
df2.loc[:, 'cP_list'] = df2['cP_list'].apply(lambda x: ' '.join(x))
