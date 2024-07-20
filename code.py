


import pandas as pd
from tabulate import tabulate
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# Load the CSV file
data = pd.read_csv('/content/drive/MyDrive/dataset_folder/mobile data/mobile phone price prediction.csv')

# Display the first few rows of the dataframe to understand its structure

print(tabulate(data.head(10), headers='keys', tablefmt='psql'))
