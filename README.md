# Mobile Price Prediction
This project aims to predict the price of mobile phones using regression and decision tree algorithms. The dataset contains various features of mobile phones, and the target variable is the price.
*****
## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)
*****

## Introduction
In this project, I use both Linear Regression and Decision Tree Regression to predict the price of mobile phones based on their features. The objective is to compare the performance of these models and determine which one provides better accuracy.

I load the dataset from the repository and perform data preprocessing, feature engineering, model training, evaluation, and tuning. Depending on your data, these techniques can make a significant difference in model performance. Proper data preprocessing and feature engineering are essential to ensure the models receive high-quality input data.
****
## Dataset
The dataset used in this project includes a diverse range of mobile phone specifications and their corresponding prices. It contains features such as brand, RAM size, battery capacity, camera quality, and other relevant attributes that influence the price of a mobile phone. The dataset is sourced from reputable databases and cleaned to ensure high-quality data for training and testing the model.
****
## Project Structure

1.**Data Loading and Initial Exploration**

`Load the CSV file`

2. **Data Preprocessing**

`Convert the 'Price' column to a numerical type`//
`Drop unnecessary columns`//
`Fill missing values with the mode of each column`//
`Label encode categorical variables`

3. **Feature Engineering**

`Remove duplicate rows`//
`Create dummy variables for categorical features`//
`Identify and retain features with a correlation greater than 0.15 with the target variable`

4. **Model Training and Evaluation**

`Split the dataset into training and testing sets`//
`Initialize and train the Linear Regression model`//
`Initialize and train the Decision Tree model`

5. **Model Tuning**

`Predict with both models`//
`Calculate performance metrics`//
`Plot actual vs predicted values for Linear Regression`//
`Plot actual vs predicted values for Decision Tree`
*****
## Installation
To get started with this project, ensure you have Python 3 installed. Clone the repository and install the required dependencies:

1. Clone the repository
  
2. Install dependencies:

`pip install -r requirements.txt`

3. Development Environments:

You can use development environments like VS Code, Jupyter Notebook, or Google Colab to run and edit the project.

*****

## Results
