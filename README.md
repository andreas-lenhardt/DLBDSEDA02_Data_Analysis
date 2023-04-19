[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

# DLBDSEDA02_Data_Analysis

Use NLP Techniques to Analyze a Collection of Texts


## Setting up the Environment
The appropriate environment for text analysis can be setup using the provided [YAML](https://github.com/andreas-lenhardt/DLBDSEDA02_Data_Analysis/tree/main/Conda_Environment_Config) file.

## Required Files for Data Analysis
The following three files are required:

- **complaints.py**: This file contains the configuration part and imports the "nlp_helpers.py" module. This script must be executed.
- **nlp_helpers.py**: This module contains all required functions and executes them according to the passed configuration parameters.
- **CDPH_Environmental_Complaints.csv**: The source file with the complaint texts that are analyzed. This is included or a recent version can be downloaded (see the following section).

## Getting the Required Source Data
There are two ways to obtain the data:

1. Unpack the zip file "CDPH_Environmental_Complaints.zip" included in the repository. This contains the required csv file.

2. Download the latest version of the csv file from the publisher [here](https://data.cityofchicago.org/Environment-Sustainable-Development/CDPH-Environmental-Complaints/fypr-ksnz)
![grafik](https://user-images.githubusercontent.com/75860355/233159371-b7a0a053-d4ca-4885-8b10-9fff1b80bd52.png)

- First click on "Export" in the upper right corner.
- Then select the "CSV" option.

In both cases, the filename of the unzipped file should be **CDPH_Environmental_Complaints.csv** and the file should be located in the same folder as the two Python files. However, the file name can be changed in the configuration section (see next section).

## The Configuration for Running the Text Analysis
![grafik](https://user-images.githubusercontent.com/75860355/233167232-bc353af8-5793-446a-9946-5fa8fcf5bc33.png)

1. **file_name (string)**: The filename of the file to be imported. Must be changed if it does not match the default name.
2. **year (int)**: The source file contains complaints from 1995 onwards. The configured year only takes into account complaints from this year and later.
3. **feature_mode (int)**: Two different modes for vectorization are possible:
- BoW (1)
- TF-IDF (2)
4. **modeling_mode (int)**: The method for modeling the topics:
- LSA (1)
- LDA (2)
5. **topics_number (int)**: The number of topics to be discovered.
6. **words_per_topic (int)**: The number of words linked to each topic that will be displayed.
