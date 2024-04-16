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
### Target column - traffic_volume
The rows containing NaN values for this column were droped as it is the ground truth column, imputing or replacing NaN values will affect the dataset originiality.

### Date_time column processing
The NaN containing row in date_time column were droped as it doesnot make sense to impute.

### holiday column
- The NaN value in the were converted to 0 and the holiday to 1.
### Temperature, rain, snow and clouds column
The NaN value in these columns were filled using ffill after the datset was sorted by date_time column.

### Weather description and weather main column
The Na values in weather main was filled using an approach where the last word of weather description was used to fill weather main NA values. And then remaining NA cointaining rows were droped.

### Drop duplicates and unnecessary columns
Next the duplicates rows were droped and also date_time, weather description column were droped.

### encoding categorical text into numerical value.
Here weather main, day, month, year, day_time, weekend were converted to numerical encoding suing serial values starting from 1 till len(var).

## Data exploration 

### Histogram
All columns were visualized using histogram.

![alt text](https://github.com/sonti-roy/featureSelection_california_housing/blob/main/plots/histogram.png)

### Box plot


## Supervised feature selection

### 1. Evaluating mutual info regression method for feature selection

[Mutual information (MI)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html) between two random variables is a non-negative value, which measures the dependency between the variables. It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency.

![alt text](https://github.com/sonti-roy/featureSelection_california_housing/blob/main/plots/mutual_info_regression_comparasion.png)

**Fig-1: The plot shows the dependency of the target on each feature.**

Top 60% of the features were selected and evaluated for it accuracy with all features dataset using Linear regression.

| Dataset                    | R2       | MSE      |
|----------------------------|----------|----------|
| Original                   | 0.575787706032451| 0.5558915986952441 |
| Subset                     | 0.5732648513884984 | 0.5591975700714763 |



### 2. Selecting features using f_regression

[f_regression](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html) uses univariate linear regression tests returning F-statistic and p-values.

![alt text](https://github.com/sonti-roy/featureSelection_california_housing/blob/main/plots/f_regression_comparasion.png)

**Fig-2: Plot for F ststistics for all feature against the target.**

As the range was highly variable for different features. Top 4 features were selected out of total 8 features and subset data was generated. The score was compared for original and subset dataset.

| Dataset                    | R2       | MSE      |
|----------------------------|----------|----------|
| Original                   | 0.575787706032451| 0.5558915986952441 |
| subset                     | 0.5043169272470043 | 0.6495475488975627 |


### 3. Pearson Correlation - feature selection

[Pearson correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) is a correlation coefficient that measures linear correlation between two sets of data. It is the ratio between the covariance of two variables and the product of their standard deviations.

![alt text](https://github.com/sonti-roy/featureSelection_california_housing/blob/main/plots/correlation_plot.png)

**Fig-3: Pearson correlation coefficient for all the pairwise combinations of features.**

Longitude and latitude, AveRooms and AveBedrms are highly correlated with -0.92 and 0.85 coeffiecient respectively. For removal of any one feature from the combination variance was analysed for the 4 features.

| Features                   | Variance |
|----------------------------|----------|
| Longitude                  | 4.014139367081251| 
| Latitude                   | 4.562292644202798 | 
| AveRooms                   | 6.12153272384879 | 
| AveBedrms                  | 0.2245915001886127 | 

Based on the variance data, longitude and AveBedrms are removed manually and evaluated the model on original and subset dataset using linear regression.

| Dataset                    | R2       | MSE      |
|----------------------------|----------|----------|
| Original                   | 0.575787706032451| 0.5558915986952441 |
| subset                     | 0.5059804263462322 | 0.6473676847426387 |

### 4. Recurive Feature Elimination (RFE)

[Recursive feature elimination (RFE)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through any specific attribute or callable. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.

Estimator/model used is Lasso as it inheriently do feature selection and subset the dataset and evaluated uisng linear regression model.

| Dataset                    | R2       | MSE      |
|----------------------------|----------|----------|
| Original                   | 0.575787706032451| 0.5558915986952441 |
| subset                     | 0.58111575601825 | 0.5489096741573366 |

### 5. Sequential Feature Selection

[Sequential Feature Selector](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html) adds (forward selection) or removes (backward selection) features to form a feature subset in a greedy fashion. At each stage, this estimator chooses the best feature to add or remove based on the cross-validation score of an estimator. In the case of unsupervised learning, this Sequential Feature Selector looks only at the features (X), not the desired outputs (y).

RidgeCV estimator is being used for this and selected 6 best features.

| Dataset                    | R2       | MSE      |
|----------------------------|----------|----------|
| Original                   | 0.575787706032451| 0.5558915986952441 |
| subset                     | 0.5099337366296423 | 0.6421872314534861 |

## Unsupervised feature selection

### 1. Principal component analysis

[PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) is defined as an orthogonal linear transformation on a real inner product space           that transforms the data to a new coordinate system such that the greatest variance by some scalar projection of the data comes to lie on the               first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on.
        
7 component were selected for the transformed space and evalauated it using Linear Regression. 7 was acheived by running at different value and             accessing the score.
        
| Dataset                    | R2       | MSE      |
|----------------------------|----------|----------|
| Original                   | 0.575787706032451| 0.5558915986952441 |
| subset                     | 0.5827018886341137 | 0.5468311917368283 |


## Evaluate different model on the subset of x 

Around 7 regression model was evaluated on the original dataset set and all the subset dataset and compared their metrics to find the best model with high accuracy.

![alt text](https://github.com/sonti-roy/featureSelection_california_housing/blob/main/plots/r2_comparasion_plot.png)

**Fig-4: R2 comparasion for all the model on different subset of data generated through feature selection.**

![alt text](https://github.com/sonti-roy/featureSelection_california_housing/blob/main/plots/mse_comparasion_plot.png)

**Fig-5: MSE comparison for all the models on different subset of data generated through feature selection.**


**Table - Top 10 model with different dataset.**

| data_subset            | r2                       | mse                      | model                   |
|------------------------|--------------------------|--------------------------|-------------------------|
| original               | 0.776                    | 0.294                    | GradientBoostingRegressor |
| mutual_info_regression | 0.773                    | 0.297                    | GradientBoostingRegressor |
| RFE                    | 0.750                    | 0.328                    | GradientBoostingRegressor |
| mutual_info_regression | 0.750                    | 0.328                    | KNeighborsRegression      |
| PCA                    | 0.713                    | 0.376                    | GradientBoostingRegressor |
| subset                 | 0.706                    | 0.386                    | GradientBoostingRegressor |
| SFS                    | 0.671                    | 0.431                    | GradientBoostingRegressor |
| RFE                    | 0.667                    | 0.436                    | DecisionTreeRegressor     |
| mutual_info_regression | 0.660                    | 0.445                    | DecisionTreeRegressor     |
| original               | 0.631                    | 0.484                    | DecisionTreeRegressor     |


*Inference - GradientBoostingRegressor performed the best with the original dataset i.e without any feature selection and the top 2nd model is also GradientBoostingRegressor with mutual_info_regression feature selection.*

## Key Takeaways

*How to perform feature selection using variuous method and perform ML model fitting and evaluate the performance of the model.*


## Code 

*The code is is avaiable in a python notebook **<u>model.ipynb</u>**. To view the code please click below*

[*Click here*](https://github.com/sonti-roy/featureSelection_california_housing/blob/main/model.ipynb)

## Roadmap

1. *Feature engineering could be explored to further improve the model accuracy*
2. *Hyperparameter Tuning*
3. *Exploring Other Ways to Improve Model*

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

