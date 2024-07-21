# Mobile Price Prediction
This project aims to predict the price of mobile phones using regression and decision tree algorithms. The dataset contains various features of mobile phones, and the target variable is the price.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)


## Introduction
In this project, we use both Linear Regression and Decision Tree Regression to predict the price of mobile phones based on their features. The objective is to compare the performance of these models and determine which one provides better accuracy.

## Dataset
The dataset used in this project includes a diverse range of mobile phone specifications and their corresponding prices. It contains features such as brand, RAM size, battery capacity, camera quality, and other relevant attributes that influence the price of a mobile phone. The dataset is sourced from reputable databases and cleaned to ensure high-quality data for training and testing the model.

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

## Installation


## Usage


## Results
