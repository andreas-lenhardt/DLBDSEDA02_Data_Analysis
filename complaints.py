import pandas as pd

#%% 1 Load and prepare the source data
df = pd.read_csv('CDPH_Environmental_Complaints.csv')

# Remove rows with no 'complaint detail'
df = df.dropna(subset=['COMPLAINT DETAIL'])

# Convert 'complaint date' column to datetime
df['COMPLAINT DATE']= pd.to_datetime(df['COMPLAINT DATE'])

# Remove rows depending on the year of the complaint
df = df[df['COMPLAINT DATE'].dt.year >= 2022]

# Copy selected columns to a new dataframe
df2 = df[['COMPLAINT ID'
          ,'COMPLAINT TYPE'
          ,'COMPLAINT DETAIL'
          ,'COMPLAINT DATE']].copy()

# Shorten column namens
cols = ['id', 'type', 'complaint', 'date']
df2.columns = cols

# Recreate the index
df2.reset_index(drop=True, inplace=True)


#%% 2.1 Case Normalization
df2.loc[:, 'complaint'] = df2['complaint'].map(str.lower)

