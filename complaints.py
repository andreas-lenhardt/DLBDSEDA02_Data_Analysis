import nlp_helpers as nlp

#%% 0 Configuration (please see README.md for details)
file_name = "CDPH_Environmental_Complaints.csv"
year = 2020
feature_mode = 2
modeling_mode = 2
ngram_range = (1,2)
topics_number = 3
words_per_topic = 10
calculate_cs = 0

#%% 1 Data Loading
df = nlp.get_complaints(file_name, year)

#%% 2 Text Cleaning
df = nlp.clean_text(df)

#%% 3 Feature Extraction
vector, matrix, data = nlp.get_feature_data(df["complaints_list"],
                                            feature_mode
                                            )

#%% 4 Topic Modeling
model = nlp.get_topics(vector,
               matrix,
               topics_number=topics_number,
               words_per_topic=words_per_topic,
               mode=modeling_mode
               )

#%% 5 Topic Coherence Score
if calculate_cs == 1 and modeling_mode == 2:
    nlp.get_coherence_score(model, df["complaints_list"])
