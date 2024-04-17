<div align="left">

[![MIT](https://img.shields.io/badge/Licence-MIT%20-%20green?style=flat)](https://github.com/sonti-roy/featureSelection_california_housing/blob/main/LICENSE)


# Traffic volume prediction using ML 

This repository contains code for data processing, feature selection and multiple model fitting and evaluation using traffic dataset. Feature selection was performed using supervised and unsupervised methods to achieve higher accuracy. Hyperparameter tuning suing gridsearchCV was performed.

## Implementation Details

- Dataset - Traffic volume dataset
- Model evaluated: 
  - [Linear Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
  - [KNeighborsRegression](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
  - [SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
  - [BayesianRidge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
  - [DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
  - [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
  - [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- Feature selection methods:
    - Supervised feature selection
        - [mutual info regression](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html)
        - [f_regression](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html)
        - [Pearson Correlation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.r_regression.html)
     - Unsupervised feature selection
         - [Principal component analysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- Hyperparameter tuning
  - Using GridSearchCV
  


## Dataset Details

| Variable Name         | Role    | Type       | Description                                                  | Units  |
|-----------------------|---------|------------|--------------------------------------------------------------|--------|
| holiday               | Feature | Categorical| US National holidays plus regional holiday, Minnesota State Fair | -     |
| temp                  | Feature | Continuous | Average temperature in Kelvin                                 | Kelvin |
| rain_1h               | Feature | Continuous | Amount in millimeters of rain that occurred in the hour       | mm     |
| snow_1h               | Feature | Continuous | Amount in millimeters of snow that occurred in the hour       | mm     |
| clouds_all            | Feature | Integer    | Percentage of cloud cover                                     | %      |
| weather_main          | Feature | Categorical| Short textual description of the current weather              | -      |
| weather_description   | Feature | Categorical| Longer textual description of the current weather              | -      |
| date_time             | Feature | Date       | Hour of the data collected in local CST time                   | -      |
| traffic_volume        | Target  | Integer    | Hourly I-94 ATR 301 reported westbound traffic volume         | -      |


## Data processing 
### 1. Target column - traffic_volume
The rows containing NaN values for this column were droped as it is the ground truth column, imputing or replacing NaN values will affect the dataset originiality.

### 2. Date_time column processing
The NaN containing row in date_time column were droped as it doesnot make sense to impute.

### 3. Holiday column
- The NaN value in the were converted to 0 and the holiday to 1.
- 
### 4. Temperature, rain, snow and clouds column
The NaN value in these columns were filled using ffill after the datset was sorted by date_time column.

### 5. Weather description and weather main column
The Na values in weather main was filled using an approach where the last word of weather description was used to fill weather main NA values. And then remaining NA cointaining rows were droped.

### 6. Drop duplicates and unnecessary columns
Next the duplicates rows were droped and also date_time, weather description column were droped.

### 7. Encoding categorical text into numerical value.
Here weather main, day, month, year, day_time, weekend were converted to numerical encoding suing serial values starting from 1 till len(var).

## Data exploration 

### 1. Histogram
All columns were visualized using histogram.

![alt text](https://github.com/sonti-roy/Traffic-volume-prediction/blob/main/plots/histogram.png)

### 2. Box plot
To check the outlier in certain column box plot is used to visualize and detect.

![alt text](https://github.com/sonti-roy/Traffic-volume-prediction/blob/main/plots/tv_box_plot.png)

![alt text](https://github.com/sonti-roy/Traffic-volume-prediction/blob/main/plots/day_box_plot.png)

![alt text](https://github.com/sonti-roy/Traffic-volume-prediction/blob/main/plots/hour_box_plot.png)

## Base Model development

For model development multiple estimator has been screened and the appropriate model was further fine tuned later.
The dataset was spilited to training dataset and evaluation dataset. Here are score for model screening.

|    Model                | R2     | MSE          |
|-------------------------|--------|--------------|
| SVR                     | 0.1953 | 3187710.8458 |
| LinearRegression        | 0.3232 | 2681101.2234 |
| KNeighborsRegression    | 0.7063 | 1163396.5928 |
| SGDRegressor            | 0.3213 | 2688483.8860 |
| BayesianRidge           | 0.3231 | 2681126.1147 |
| DecisionTreeRegressor   | 0.9340 | 261360.1047  |
| GradientBoostingRegressor| 0.9017 | 389401.9654  |
| RandomForestRegressor   | 0.9652 | 137792.2456  |
| XGBRegressor            | 0.9725 | 108748.1525  |

RandomForest and XGBRegressor was found to be performing best among all the models.

## Cross valadatiuon was performed to see if there is overfitting.

| CV Number | Score     |
|-----------|-----------|
| 1         | 0.92805313|
| 2         | 0.9287948 |
| 3         | 0.94174875|

The scores are less than the score genearted from evaluation dataset. It indicates overfitting of the model.

## Supervised feature selection

### Pearson Correlation - feature selection

[Pearson correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) is a correlation coefficient that measures linear correlation between two sets of data. It is the ratio between the covariance of two variables and the product of their standard deviations.

![alt text](https://github.com/sonti-roy/Traffic-volume-prediction/blob/main/plots/cor_plot.png)

**Fig : Pearson correlation coefficient for all the pairwise combinations of features.**

### Evaluating mutual info regression method for feature selection

[Mutual information (MI)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html) between two random variables is a non-negative value, which measures the dependency between the variables. It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency.

![alt text](https://github.com/sonti-roy/Traffic-volume-prediction/blob/main/plots/mi.png)

**Fig-1: The plot shows the dependency of the target on each feature.**

Top 70% of the features were selected and evaluated for it accuracy with all features dataset using RandomForestRegressor.

| Metric                         | Value                           |
|--------------------------------|---------------------------------|
| R2 Score                       | 0.9656273468841373              |
| Mean Squared Error             | 133272.57017553216              |
| Cross-Validation Scores        | [0.96480622, 0.96424541, 0.95989378] |
| Percentage Deviation from Mean | 11.21                           |


### 2. Selecting features using f_regression

[f_regression](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html) uses univariate linear regression tests returning F-statistic and p-values.

![alt text](https://github.com/sonti-roy/Traffic-volume-prediction/blob/main/plots/f_reg.png)

**Fig-2: Plot for F ststistics for all feature against the target.**

Top 60% of the features were selected and evaluated for it accuracy with all features dataset using RandomForestRegressor.

| Metric                         | Value                           |
|--------------------------------|---------------------------------|
| R2 Score                       | 0.944669474729565               |
| Mean Squared Error             | 214532.21219487552              |
| Cross-Validation Scores        | [0.94521279, 0.94479329, 0.93781204] |
| Percentage Deviation from Mean | 14.22                           |


## Unsupervised feature selection

### 1. Principal component analysis

[PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) is defined as an orthogonal linear transformation on a real inner product space           that transforms the data to a new coordinate system such that the greatest variance by some scalar projection of the data comes to lie on the               first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on.
        
9 component were selected for the transformed space and evalauated it using RandomForestRegressor.
        
| Metric                         | Value                           |
|--------------------------------|---------------------------------|
| R2 Score                       | 0.5565759676724207              |
| Mean Squared Error             | 1719281.3213077914              |
| Cross-Validation Scores        | [0.53416057, 0.53846765, 0.53979648] |
| Percentage Deviation from Mean | 40.27                           |



# Hyperparameter tuning

Hyparameter tuning was performed using GridSearchCV with parameter

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

The best paarmeter was used to train the RandomForestRegressor Model on train dataset and evaluated on test dataset. The scores are




*Inference - GradientBoostingRegressor performed the best with the original dataset i.e without any feature selection and the top 2nd model is also GradientBoostingRegressor with mutual_info_regression feature selection.*

## Key Takeaways

*How to perform feature selection using variuous method and perform ML model fitting and evaluate the performance of the model.*


## Code 

*The code is is avaiable in a python notebook **<u>model.ipynb</u>**. To view the code please click below*

[*Click here*](https://github.com/sonti-roy/featureSelection_california_housing/blob/main/model.ipynb)

## Libraries 

**Language:** Python

**Packages:** Sklearn, Matplotlib, Pandas, Seaborn

## Acknowledgements

*Resources used* 

 - [scikit-learn](https://scikit-learn.org/stable/index.html)
 - OpenAI. (2024). ChatGPT (3.5) Large language model. https://chat.openai.com


## Contact

If you have any feedback/are interested in collaborating, please reach out to me at [LinkdIn](https://www.linkedin.com/in/sonti-roy-phd-8589b711a/)


## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

