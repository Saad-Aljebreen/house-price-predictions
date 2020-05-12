

## Introduction:
In this project we're going to predict the sale price of a given house by modeling a data set of 1460 houses and their features after analysing these features and preparing the data for the model.
This project is about regression modeling and we will use a few examples of regression models to arrive to a good score of accuracy. Our model should take a data set of houses and predict their prices. Then, we can check the accuracy of the model on kaggle.



## Problem Statment:
We can tell alot about real estate and economy in a certain city by knowing the relation (correlation) between the prices of houses and certain features of these houses. 
We want to develop a model to predict the price of a house based on the training data set we have of 1460 house along with their features and provided sale price. 


## Executive Summary:
1\ Objective: Developing a model that can predict houses prices based on certain features.

2\ Datasets Description: 

https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

We have 2 main datasets. 
1- The main data set which has 1460 entries(each representing a house) along with 81 features (columns) , starting with ID for each row/house. and ending with Sale Price of that house. The sale price is our target feautre which is the base we want to develop a model to predict for other houses. 
2- The testing data set which has 1459 entries(each representing a house) along with 80 features (columns) , starting with ID for each row/house. There is no sale price since the goal of this set is to predict the price. 

3\ Data Importing & Cleaning: We import the data, look at it, and find out that it has many categorial and numerical values. We start looking at missing values (nulls), meaningless values (specially categories that are 95% or more on one type). Then we fill nulls with median, mode or mean depending on the presence of outliers and the estimation of what's appropriate. 

4\ EDA & Data Visualization: We start plotting categorical and numerical features against the sale price to see the possible effect and basic insight for each feature. We then plot some features together to see the relation between these features. 
By looking at some features frequncy bias to one certain category, we decided to drop them. We ended up dropping 21 features that clearly will not benifit the modeling of the data.  

5\ Data Modeling:

We use a variety of differnt models keeping in mind some basic knowledge about models:

1- Lasso will take features that are relevant to the target automatically so we can start with it.
2- We use test a variety of models including linear regression, lasso, lasso cv, random forest and grid search cv with differnent types of estimators and paramaters. We have a good result from grid search cv with lasso cv and so far this is the best model. 


## Datasets Description:
|Feature|Type|Dataset|Description|
|---|---|---|---|
|SalePrice|int64|train|The property's sale price in dollars. This is the target variable that you're trying to predict.|  
|MSSubClass|Object|train/test|The building class| 
|MSZoning|object|train/test|The general zoning classification|
|LotFrontage|float64|train/test|Linear feet of street connected to property|
|LotArea|int64|train/test| Lot size in square feet|
|Street|object|train/test|Type of road access|
|Alley|object|train/test|Type of alley access|
|LotShape|object|train/test| General shape of property|
|LandContour|object|train/test|Flatness of the property|
|Utilities|object|train/test|Type of utilities available|
|LotConfig|object|train/test|Lot configuration|
|LandSlope|object|train/test|Type of alley access|
|LotShape|object|train/test| Slope of property|
|Neighborhood|object|train/test|Physical locations within Ames city limits|
|Condition1|object|train/test|Proximity to main road or railroad|
|Condition2|object|train/test|Proximity to main road or railroad (if a second is present)|
|BldgType|object|train/test|Type of dwelling|
|HouseStyle|object|train/test|Style of dwelling|
|OverallQual|int64|train/test|Overall material and finish quality)|
|OverallCond|object|train/test|OverallCond|
|YearBuilt|int64|train/test|Original construction date|
|BuildingAge|int64|train/test|Age of the building|
|RemodelAge|int64|train/test|number of years since last remodel|
|YearRemodAdd|int64|train/test|Remodel date|
|RoofStyle|object|train/test|Type of roof|
|RoofMatl|object|train/test|Roof material|
|Exterior1st|object|train/test|Exterior covering on house|
|Exterior2nd|object|train/test|Exterior covering on house (if more than one material)|
|MasVnrType|object|train/test|Masonry veneer type|
|MasVnrArea|int|train/test|Masonry veneer area in square feet|
|ExterQual|object|train/test|Exterior material quality|
|ExterCond|object|train/test|Present condition of the material on the exterior|
|Foundation|object|train/test|Type of foundation|
|BsmtQual|object|train/test|Height of the basement|
|BsmtCond|object|train/test|General condition of the basement|
|BsmtExposure|object|train/test|Walkout or garden level basement walls|
|BsmtFinType1|object|train/test|Quality of basement finished area|
|BsmtFinSF1|float64|train/test|Type 1 finished square feet|
|BsmtFinType2|object|train/test|Quality of second finished area (if present)|
|BsmtFinSF2|float64|train/test|Type 2 finished square feet|
|BsmtUnfSF|float64|train/test|Unfinished square feet of basement area|
|TotalBsmtSF|float64|train/test|Total square feet of basement area|
|Heating|object|train/test|Type of heating|
|HeatingQC|object|train/test|Heating quality and condition|
|CentralAir|object|train/test|Central air conditioning|
|Electrical|object|train/test|Electrical system|
|1stFlrSF|int64|train/test|First Floor square feet|
|2ndFlrSF|int64|train/test|Second floor square feet|
|LowQualFinSF|object|train/test|Low quality finished square feet (all floors)|
|GrLivArea|int64|train/test|Above grade (ground) living area square feet|
|BsmtFullBath|float64|train/test|Basement full bathrooms|
|BsmtHalfBath|float64|train/test|Basement half bathrooms|
|FullBath|int64|train/test|Full bathrooms above grade|
|HalfBath|int64|train/test|Half baths above grade|
|BedroomAbvGr|int64|train/test|Number of bedrooms above basement level|
|KitchenAbvGr|int64|train/test|Number of kitchens|
|KitchenQual|object|train/test|Kitchen quality|
|TotRmsAbvGrd|int64|train/test|Total rooms above grade (does not include bathrooms)|
|Functional|object|train/test|Home functionality rating|
|Fireplaces|int64|train/test|Number of fireplaces|
|FireplaceQu|object|train/test|Fireplace quality|
|GarageType|object|train/test|Garage location|
|GarageYrBlt|int|train/test|Year garage was built|
|GarageFinish|object|train/test|Interior finish of the garage|
|GarageCars|object|train/test|Size of garage in car capacity|
|GarageArea|object|train/test|Size of garage in square feet|
|GarageQual|object|train/test|Garage quality|
|GarageCond|object|train/test|Garage condition|
|PavedDrive|object|train/test|Paved driveway|
|WoodDeckSF|int|train/test|Wood deck area in square feet|
|OpenPorchSF|int|train/test|Open porch area in square feet|
|EnclosedPorch|object|train/test|Enclosed porch area in square feet|
|3SsnPorch|int|train/test|Three season porch area in square feet|
|ScreenPorch|int|train/test|Screen porch area in square feet|
|PoolArea|int|train/test|Pool area in square fee|
|PoolQC|object|train/test|Pool quality|
|Fence|object|train/test|Fence quality|
|MiscFeature|object|train/test|Miscellaneous feature not covered in other categories|
|MiscVal|int|train/test|Screen porch area in square feet|
|PoolArea|int|train/test|Pool area in square fee|
|PoolQC|object|train/test|Pool quality|
|Fence|object|train/test|Fence quality|
|MiscVal|int64|train/test|Value of miscellaneous feature|
|MoSold|int64|train/test|Month Sold|
|YrSold|int64|train/test|Year Sold|
|SaleType|object|train/test|Type of sale|
|SaleCondition|object|train/test|Condition of sale|



## Conclusions:
We are looking to create a model to predict the prices of houses based on a data set we have for 1460 house along with 79 features for each house.
 
The data features have categorial and numerical values. We start cleaning and looking at missing values (nulls), meaningless values (specially categories that are 95% or more on one type). Then we fill nulls with appropriate values and we finish the data cleaning process. 
then we start EDA & visualizations. 
We drop some features and edit some and prepare the data for the next phase which is modeling.
We test a variety of models to finally reach the best outcome/score with grid search cv and lasso cv. 

Insights: 

1- There are many various correlations between a house price and its features and the more these features are better/more the higher the price; as expected. 
