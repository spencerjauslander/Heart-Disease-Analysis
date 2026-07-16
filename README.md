# Python Rutgers Bootcamp Challenge - Project Four - Heart Disease Understanding 

This activity is broken down into various pieces of code to understand the dynamics of heart disease and choose a model using various variables to note potential predictions for heart disease. 

A presentation of our analysis can be found [here](https://github.com/oliverkisza/Final-Project-Team-1/blob/main/Understanding%20Heart%20Disease.pdf)

<summary>Contents</summary>
  <ol>
    <li><a href="#desc">Description</a></li>
    <li><a href="#dpp">Data Preprocessing</a></li>
    <li><a href="#cte">Compiling, Training, & Evaluation</a></li>
    <li><a href="#optimize">Optimization</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#Analysis">Report & Analysis</a></li>
  </ol>

<a name="desc"></a>

## Description

The purpose of this analysis is to ascertain possible models for the prediction of heart disease. Our group chose the [following dataset from Kaggle](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/) which was itself collected from the CDC's [Behavioral Risk Factor Surveillance System (BRFSS)](https://www.cdc.gov/brfss/index.html). Per the CDC, the BRFSS "completes more than 400,000 adult interviews each year, making it the largest continuously conducted health survey system in the world."

![Screenshot 2023-12-14 113319](https://github.com/oliverkisza/Final-Project-Team-1/assets/18508699/ca14849d-d5d9-4c1c-a943-cc6e848c27d0)


Due to the dataset being completed by the CDC and the ability to compare and contrast recent datasets of heart disease data (2020 and 2022 data), our group felt this could be a great way to predict potential heart disease across the American populace. 

<a name="dpp"></a>

## Step 1a: Data Preprocessing

In order to best understand and target variables to use in the model, the group sought a number of techniques and tools to understand both datasets. 

In an optional data evaluation, the group used [IBM's SPSS](https://www.ibm.com/spss) to do quick stepwise multiple regression analyses to ascertain the top variables to use in the model. 

![Screenshot 2023-12-14 121232](https://github.com/oliverkisza/Final-Project-Team-1/assets/18508699/224fcdbe-6538-49c2-925f-f341dd184ceb)

For the 2020 dataset set, **Heart Disease (Yes/No)** was used as primary dependent variable and all others were used as independent variables. Similar regression models were run with the 2022 data which ran two different dependent variables, **Heart Attack (Yes/No)** and **High Risk Last Year (Yes/No)**. 

![Screenshot 2023-12-14 121608](https://github.com/oliverkisza/Final-Project-Team-1/assets/18508699/5cbd3ffa-51f4-4e32-a898-7c62a1395cbf)

There were not large r2 or correlations in any of the analyses we saw, but it did give us some directions on the potential usage for variables in the models. 

The group also used Tableau to visually examine the data. You can examine the data yourself in the following public Tableau pages: 

* [Heart Disease in America - 2020](https://public.tableau.com/app/profile/christopher.manfredi/viz/HeartDiseaseInAmerica/HeartDiseaseinAmerica#1)
* [Heart Disease in America - 2022](https://public.tableau.com/app/profile/christopher.manfredi/viz/HeartAttackInAmerica-2022/HeartDiseaseinAmerica-2022?publish=yes)

![Screenshot 2023-12-14 133458](https://github.com/oliverkisza/Final-Project-Team-1/assets/18508699/a94bbf30-d73d-4284-bd96-868415b8b3b9)

![Screenshot 2023-12-14 133439](https://github.com/oliverkisza/Final-Project-Team-1/assets/18508699/31128b36-2c65-4920-8fa4-a1b66099ccc2)


## Step 1b: Data Cleaning

Data cleaning was completed using Python for both datasets. The following are breakdown of both data sets: 

**2020 Dataset** and [2020 cleaning code](https://github.com/oliverkisza/Final-Project-Team-1/blob/main/FA_data_cleaning_2020.ipynb)

- 17 feature columns (heart disease indicators)
- 1 target column (HeartDisease)
- ~320,000 rows
- Renamed columns to match identical columns in heart_2022 with slightly different names(ex: GenHealth:GeneralHealth; PhysicaHealth: PhysicalHealthDays; DiffWalking:DifficultyWalking,etc.)
- Renamed other columns for clarity such as SleepTime:HoursOfSleep
- Removed rows with ambiguous data: Diabetic (Yes, during pregnancy, No, borderline diabetes) to yield binary column
- Created dummy variables ideal for binary categories
- Mapped ordinal variables from least to greatest starting from 0 (GeneralHealth and AgeCategory)

**2022 Dataset** and [2022 cleaning code](https://github.com/oliverkisza/Final-Project-Team-1/blob/main/data_cleaning_2022.ipynb)

- 39 feature columns (heart disease indicators)
- 1 target column (“HadHeartAttack”)
- ~246,000 rows

- Removed columns (“State”, “HadDiabetes”, etc)
- Renamed other columns for clarity such as “SleepTime”: “HoursOfSleep”
- Removed NaN
- Created dummy variables ideal for binary categories (“Sex”, “PhysicalActivities”, “HadHeartAttack”, “HadAngina”, etc)
= Mapped ordinal variables from least to greatest starting from 0 (“GeneralHealth”, “AgeCategory”, “LastCheckupTime”, “RemovedTeeth”, “SmokerStatus”, and “ECigaretteUsage”)


<a name="cte"></a>

## Step 2: Compiling, Training, & Evaluation

Both datasets were stored in Github and read through Spark for light querying and to have a better understanding of the data. 
** [Spark notebook](https://raw.githubusercontent.com/oliverkisza/Final-Project-Team-1/main/Spark_Heart_Data.ipynb)

- First we read the 2020 cleaned data as a csv and created a pandas dataframe from it.
- Repeated the same steps to create a pandas dataframe for the 2022 data as well.
- Created a temporary view of the pandas dataframe using createOrReplaceTempView and named it 'heart20'.  This allows us to read the data as a table
  and allows us to run Spark SQL queries.
- We cached the table "heart20".
- Ran the following query so that we could see the total count of Yes/No for HeartDisease.  We also ran some percentages to check for any relations but realized
  these were skewed due to imbalanced data:

a2020q1 = """
SELECT
  HeartDisease,
  COUNT(*) AS TOTAL,
  ROUND(COUNT(CASE WHEN Smoking = 'Yes' THEN Smoking END) / TOTAL * 100,2) AS PERCENT_SMOKING,
  ROUND(COUNT(CASE WHEN AlcoholDrinking = 'Yes' THEN AlcoholDrinking END) / TOTAL * 100,2) AS PERCENT_DRINKERS,
  ROUND(COUNT(CASE WHEN Stroke = 'Yes' THEN Stroke END) / TOTAL * 100,2) AS PERCENT_STROKE,
  ROUND(COUNT(CASE WHEN Diabetic = 'Yes' THEN Diabetic END) / TOTAL * 100,2) AS PERCENT_DIABETIC,
  ROUND(COUNT(CASE WHEN Asthma = 'Yes' THEN Asthma END) / TOTAL * 100,2) AS PERCENT_ASTHMA,
  ROUND(COUNT(CASE WHEN KidneyDisease = 'Yes' THEN KidneyDisease END) / TOTAL * 100,2) AS PERCENT_KIDNEY_DISEASE,
  ROUND(COUNT(CASE WHEN SkinCancer = 'Yes' THEN SkinCancer END) / TOTAL * 100,2) AS PERCENT_SKIN_CANCER
FROM heart20
Group by HeartDisease
ORDER BY HeartDisease DESC
"""
spark.sql(a2020q1).show()

-Ran a second query to show the General Health of the population and what percentage was at risk.  Here we noticed whoever considered themselves
having excellent or very good health had a very low percentage of being at risk, while those who considered themselves fair or poor had the highest risk.

a2020q2 = """
SELECT
  GeneralHealth,
  Count(GeneralHealth) AS Total,
  COUNT(CASE WHEN HeartDisease = 'Yes' THEN HeartDisease END) AS HighRiskCount,
  ROUND(HighRiskCount /  Total * 100,2) AS Percent_At_HighRisk
FROM heart20
GROUP BY GeneralHealth
ORDER BY Percent_At_HighRisk ASC
"""
spark.sql(a2020q2).show()

-Uncached the table and exported our dataframe to a csv for modeling.
-Created a temporary view of the 2022 dataframe and ran similar queries which presented the same conclusions as the 2020 data.
-Uncached the table and exported our dataframe as a csv for modeling.

[**First Model Run**](https://raw.githubusercontent.com/oliverkisza/Final-Project-Team-1/main/first_heart_disease_model_2022.ipynb)

-Read our cleaned data as named our dataframe 'df'.
-Dropped columns which were not binary features.
-Fit the model using our training data.
-Generated testing predictions and our confusion matrix.
-Created our training and testing classification reports.

**Training Report**

![image](https://github.com/oliverkisza/Final-Project-Team-1/assets/134735921/c10a6a68-4fdf-4bd7-9e2b-6e61e0077b60)

**Testing Report**

![image](https://github.com/oliverkisza/Final-Project-Team-1/assets/134735921/2d49aa0c-70d7-473e-aa21-8509fd3cc470)

-Scaled our data and used 3 hidden layers.
-Fit the model and ran 20 epochs.
-Calculated weights for each feature.
-Calculate accuracy of the model using the following code:

model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

![image](https://github.com/oliverkisza/Final-Project-Team-1/assets/134735921/b2def1ea-ea0b-4e42-b2ef-756a789545ca)


<a name="optimize"></a>

## Step 3: Optimize the Model

Several steps were taken to improve our models:
- Binning certain column data (manually mapping ordinal data)
- Applied various resampling techniques;
    - Oversampling
    - Undersampling
    - A combination of both
 
We experimented with different balances of the 1 and 0 values in the target column (in training data only), but were unable to substantially improve model performance. Our model(s) based on the resampled data sacrificed some precision for improved recall scores, which are more relevant when predicting if patients have heart disease.

### SMOTE (Sample Minority Oversampling Technique)

[Oversampling Optimization](https://github.com/oliverkisza/Final-Project-Team-1/blob/main/Testing%2C%20SMOTE%2C%20Random%20Forests%20-%202020%20Data.ipynb)

- There’s an imbalance ratio of No responses for heart disease to Yes responses. This causes the accuracy metric to be biased and not preferable. 
- SMOTE alters the training set by increasing the number of Yes data points to match the volume of No data points. 
- Unlike Random Oversampling where the minority data is duplicated, SMOTE is Interpolation Oversampling that synthesizes new data.
- Creates synthetic samples by taking a random instance of the minority class, finding its k-nearest neighbors and placing an new instance at a random distance between.
- Also ran the model with ADASYN, SMOTETomek & and SMOTEENN. SMOTE performed the best out of all of them.

![SMOTE changes](https://github.com/oliverkisza/Final-Project-Team-1/assets/137104602/9f61bc7e-2f63-42be-b17f-4577e1cc0a24)

![SMOTE comparison](https://github.com/oliverkisza/Final-Project-Team-1/assets/137104602/308ed154-5129-41a5-b8fd-e637d02a4364)

### Random Forests

- Random Forests is an ensemble learning method. Instead of one complex decision tree, it samples the data and constructs a multitude of simple decision trees.
- Unlike normal decision trees, Random Forests is robust against overfitting. SMOTE runs the risk of introducing noisy instances and overfitting problems.


![RF changes](https://github.com/oliverkisza/Final-Project-Team-1/assets/137104602/bc93c6b0-7e2f-435f-8301-d38e2017f3db)

 <a name="results"></a>
 ## Step 4: Results
 
![Screenshot 2023-12-11 200612](https://github.com/oliverkisza/Final-Project-Team-1/assets/137104602/b4fd472f-2deb-4c46-80f9-eb1c152fc038)

- Through SMOTE + Random Forests, the model classification report had low precision and high recall for the population with heart disease, which is preferable.
- Low precision: the model is incorrectly predicting heart disease for people who don’t have it; many false positives. 
- High recall: for the population that does have heart disease, the model is correctly identifying a majority of them; few false negatives. 
- False positives will lead to further testing and safety precautions. A high number of false negatives may result in heart disease being untreated which might have serious consequences.

<a name="Analysis"></a>

 ## Step 5: Analysis

Overall, our analysis did not meet our expectations of demonstrating meaningful predictive power at least 75% classification accuracy or 0.80 R-squared.

Our models consistently noted high precisions at the "0" or "No Heart Disease" with fairly excellent recall and f1-scores. However, none of our models could effectively predict "1" or who had heart disease.

Despite SMOTE, Random Forest, and Undersampling as well as combined techniques, our model was never able to get near our 75% rate we were shooting for. We do feel though our track was important because we found professional data scientists working the same angles in [their research.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9322725/)

## Dependencies

* Used IBM SPSS for regression analysis (optional)
* Using Python for cleaning of data and exporting clean CSV file
* Using Spark for model ingestion, low level analysis, and exporting file
* Using Google Collaboratory to do the coding and reading in the CSV file and Tensor Flow workloads 

## Installing

* No modifications needed to be made to files/folders

## Help

No help required. 
