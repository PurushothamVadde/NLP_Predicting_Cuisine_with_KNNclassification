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

1.The data frame and the input list that we need to predict is passed as input arguments to the Text_Normalization function, we iterate through the **ingredients**  column in the dataframe and perform the text cleaning steps. \
2. After cleaning the text we perform the word tokenization and add to  list_for_matrix. \
3. The inputlist is also added to the list_for_matrix at the index0. \
4. the function returns the list_for_matrix. 
>list_for_matrix.insert(0,input_list)

### Vectorizer (Normalized_data):
The  list_for_matrix from the above function is passed as input argument to Vectorizer function, we convert the input list into the matrix format using the count vectorizer. 

>    vectorizer = TfidfVectorizer(stop_words='english')
>    matrix = vectorizer.fit_transform(Normalized_data)

The 0 row in the matrix is addded to the Input_matrix which is the vector of input list. \
the matrix from [1:] added to the new_matrix which is used for the tain and test the model. 



### Finging_Kneighbors(X,dataframe):
The new_matrix and dataframe is passed as input argument to the function,the cuisine column in the dataframe is converted into numerical for each cateogry using the label encoder package.

>    LabelEncoder = preprocessing.LabelEncoder()
>    LabelEncoder.fit(df['cuisine']) 
>    Y = LabelEncoder.transform(df['cuisine'])

The matrix X is split into 70 and 30 percentage for train and test the model by using below code:

>    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=100)

Finding the Neighborsize: \ 
I used the KNeighborsClassifier model to predict the cuisine. \
To find the optimal number of neighbors i iterate from 1 to 25 neighbors and find the accuracy for each neighbor size. \
The accuracy for each neighbor size is added to the score_list, the score_list is sorted and find the neighborhood with high accuracy. \
The function returns the neighbors size.
> for neighbors in range(25):
>>>        K_neighbors = neighbors + 1
>>>        KNN_model = KNeighborsClassifier(n_neighbors=K_neighbors, weights='uniform', algorithm="auto")
>>>        KNN_model.fit(X_train, Y_train)
>>>        Y_prediction = KNN_model.predict(X_test)
>>>        score = metrics.accuracy_score(Y_test, Y_prediction)
>>>        score_list.append (score)
>>>        print("Accuracy score  ", score, "% for K Neighbors-Value:", K_neighbors)
>>    Kneighbors = score_list.index(max(score_list)) \
>>    print("The number of neighbors with good accuracy score is :", Kneighbors+1 ) \
>>    return Kneighbors

















