## Text Analytics project 2  "TheAnalyzer"

## Author: Purushotham Vadde

## Packages Required for Project:
- nltk
- json
- re
- sklearn
- pandas
In this project i am taking the text from json format file related to  **Yummly Food Cuisine** and performing the KNNClassification on the data to predict the type of cuisine.

The projects have below files:
## Analyzer.py
The Analyzer file contains the below functions

### Reading_jsondata(json_file ):

In this function the we pass the Json_file as input file we read the json file data using json package and stored into the dataframe using pandas and returns the dataframe.

### Text_Normalization(data_frame,input_list):

1.The data frame and the input list that we need to predict is passed as input arguments to the Text_Normalization function, we iterate through the **ingredients**  column in the dataframe and perform the text cleaning steps. 
2. After cleaning the text we perform the word tokenization and add to  list_for_matrix.
3. The inputlist is also added to the list_for_matrix at the index0.
4. the function returns the list_for_matrix.
>list_for_matrix.insert(0,input_list)

###
