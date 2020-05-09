## Text Analytics project 2  "TheAnalyzer"

## Author: Purushotham Vadde


### How did you turn your text into features and why?

I used the TfidfVectorizer to convert the text into features.
The Input_matrix contains the features for input ingredients.
The new_matrix contains the features of input json file ingredients.

> def Vectorizer (Normalized_data): 
>>    vectorizer = TfidfVectorizer(stop_words='english') \
>>    matrix = vectorizer.fit_transform(Normalized_data) \
>>    Input_matrix = matrix[0] \
>>    new_matrix = matrix[1:] \
>>    return new_matrix,Input_matrix


### What classifiers/clustering methods did you choose and why?

- I used KNeighborsClassifier model to predict the cuisine, the KNeighborsClassifier model works by calculating the distance between each vector near to input_matrix vector. The optimal Kneighbors are find and passed as input argument to the model. 
- The model is trained with matrix of ingredients text.
- The KNeighborsClassifier works effectivelly for classification.

> KNN_model = KNeighborsClassifier(n_neighbors=Kneighbors, weights='uniform', algorithm="auto") \
> KNN_model.fit(X_train, Y_train) \
> Y_prediction = KNN_model.predict(Input_matrix)


### What N did you choose and why?

- Finding the Neighborsize: 
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

The optimum best N value is **13** with accuracy score  73.4 %.

![N Value](https://github.com/PurushothamVadde/The-Analyzer/blob/master/Neighbor%20size.png)



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

- Finding the Neighborsize: 
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


### KNN_Classification_Model(X,dataframe,Input_matrix,Kneighbors):

In this method we preidct the type of cuisine and the nearest matching cuisines.
1. The function takes matrix for ingredients from the  json file and stored into X.
2. The Target variable is cuisine from the data frame is converted into numerical and stored into the Y variable.
3. The X and Y data is split into 30 and 70 percentage to train and test the model.
4. The KNeighborsClassifier model is used for the classification of cuisine based on the input ingredients.
5. The Kneighbors find from the Finging_Kneighbors is passed to the model as tuning parameter.
6. The model is trained with input ingredients data using fit function.
7. The Input matrix is passed to the model to predict using predict function.

> X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=100) \
> KNN_model = KNeighborsClassifier(n_neighbors=Kneighbors, weights='uniform', algorithm="auto") \
> KNN_model.fit(X_train, Y_train) \
> Y_prediction = KNN_model.predict(Input_matrix)

After predicting the cuisine to get the actual name of cuisine we use to label encoding package.

>predicted_cuisine = LabelEncoder.inverse_transform(Y_prediction)

To get the closest 5 cuisine recipes  we use the Cosine similarity function. \
The cosine similarity function gives score for each cuisine with respect to the input ingredients matrix.
> cosine_score = cosine_similarity(Input_matrix, X)

The cosine scores are added to the data frame and sorted to get the higest scored cuisines.
> dataframe['Cuisine_score'] = Cuisine_score \
> DataFrame = dataframe.sort_values(by ='Cuisine_score',ascending=False) \
> Nearest_cuisine = DataFrame[['id', 'Cuisine_score']].head(Kneighbors)





## Steps to Run project

- **Step1** \
clone the project directory using below command 
> git clone  https://github.com/PurushothamVadde/The-Analyzer.git

- **Step2** \
Navigate to directory that we cloned from git **Analyzer** and run the below command.

>pipenv run python Analyzer.py







