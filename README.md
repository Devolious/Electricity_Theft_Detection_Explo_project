# Explo_project
For the classification of suspicious customers by using data from smart meters.

The data was crude by nature, so first we had to prepare the data so that accordingly so we can model it. So first, the data was to be visualized. After visualizing it with different techniques, the appropriate data was to be pre-processed to use it directly for our analysis. When the data was ready, it was evident that there are various segments of users depending on their consumption patterns. So, the data clustering was performed using Hierarchical Clustering taking appropriate number of clusters. Finally, precise number of datapoints of suitable clusters were taken to be manipulated & replaced in the original data.

This changed data was finally used to build the generic model for which we chose various algorithms which are Random Forest Algorithm, XG Boost Classifier, Naive Bayes, KNN.
The model was then trained for this manipulated data for increasing its accuracy.

Since we assumed that the data provided to us is entirely from honest customers, therefore to enable our model to detect theft, it was necessary to manipulate some of the data and modify it such that it represents theft data. So from the 3 clusters of monthly data, we took the middle one and performed further analysis on that data. We assumed that the value of theft data will most probably be less than the normal data.

So we took 500 IDs from the cluster. Then we divided them into 4 parts and did following 4 operations on them :

To 1st part, we multiplied every element by a random number( between 0.1 and 0.8), one number per row.
To 2nd part, we multiplied every element by a random number( between 0.1 and 0.8), one number per datapoint.
To 3rd part, we generated a random number(0.1 to 0.8). We then replaced every element of a row by the product of that number and the row wise mean.
In 4th part, we replaced every element of a row by the row-wise mean.

We then unpivoted the daily data such that it had columns as Day, Customer IDs, Daily Energy Consumption. The Unhealthy Data was the data of the 500 IDs manipulated in the above process and was replaced in the daily data taking note of those IDs. This unhealthy data was labelled as ‘1’(Theft) and the rest data was labelled as ‘0’ (Non-Theft). Now, the data contained both healthy and unhealthy data along with labels for theft (1) and non-theft(0). By using the day column, column of Daily Average Temperature from the weather data was merged with our data (since energy consumption is affected by temperature variations ). Day column was then dropped from the data. The final Dataset now had columns Customer IDs, Daily Consumption, Daily Average Temperature, Label and could now be used efficiently for model building.

The final step of our project was the Final Model Building. We used various algorithms which are Random Forest Regressor, XGBoost classifier,  Naive Bayes, KNN to build the model. Then we have drawn the confusion matrix and determined the accuracy score and F1 score of each model, and we have found that Random Forest regressor and XGBoost algorithm have performed really well on the training dataset.

The Repository contains code for theft detection model, cluster determination techniques, finding correlation of weather data along with the link for the Kaggle dataset
used in the project. We kept a ratio of 70-30 between the training and testing dataset.
