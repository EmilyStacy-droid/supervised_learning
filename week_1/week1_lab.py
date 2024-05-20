# https://geodacenter.github.io/data-and-lab/KingCounty-HouseSales2015/

import scipy as sp
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set color map to have light blue background
# sns.set()
import statsmodels.formula.api as smf
import statsmodels.api as sm
#%matplotlib inline

df = pd.read_csv('data/kc_house_data.csv')

# returns the first five rows of the DataFrame by default
df.head()
print('date:',df.date)
print('data type',type(df.date.iloc[0]))

# extract year and month info from the string
# create new features 'sales_year' and 'sales_month' in df
# date format: 20140521T000000
df['sales_year'] = df.date.apply(lambda x: int(x[:4]))
df['sales_month'] = df.date.apply(lambda x: int(x[4:6]))
df.groupby('sales_month')

print(df.groupby('sales_month')['id'].count())
print(df.groupby('sales_year')['id'].count())

most_sales_month = df.groupby('sales_month')['id'].count().idxmax()
print('most sales month:', most_sales_month)

least_sales_month =  df.groupby('sales_month')['id'].count().idxmin()
print('least sales month:',least_sales_month)

df.info()

price = 'numeric'
bathrooms = 'numeric'
waterfront = 'categorical'
grade = 'numeric'
zipcode = 'categorical'
sales_year = 'numeric'

for c in df.columns[2:]:
    print(c, df[c].unique())

# drop doc: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html, drop unnecessary features, replace df
df = df.drop(['id', 'date', 'zipcode'], axis=1)

# https://www.geeksforgeeks.org/python-pandas-dataframe-corr/
correlation_matrix = df.corr()
# Create a heatmap
# plt.figure(figsize=(15, 10))
# sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
# plt.title("Correlation Matrix")
best_guess_predictor = 'sqft_living'

# set pair plot
# columns_to_include = df.columns[:10]
# sns.pairplot(df[columns_to_include], diag_kind='kde')

# Specify the features (X) and the target variable (y), drop the target price col 
X = df.drop('price', axis=1)  # Features
y = df['price']  # Target variable
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Concatenate 'price' column with X_train and X_test
X_train_with_price = pd.concat([X_train, y_train], axis=1)
# X_test_with_price = pd.concat([X_test, y_test], axis=1)

# Display the first few rows of the training and testing DataFrames
# print("Training DataFrame (X_train_with_price):")
# print(X_train_with_price.head())

print("Training DataFrame (X_train):")
print(X_train.head())
print("\nTesting DataFrame (X_test):")
print(X_test.head())
print("Length of X_train:", len(X_train))
print("Length of X_test:", len(X_test))
plt.show()
# https://www.statsmodels.org/dev/example_formulas.html
# Build the linear regression model
model = smf.ols(formula=f"price ~ {best_guess_predictor}", data=X_train_with_price).fit()
# Calculate adjusted R-squared
adj_R2 = model.rsquared_adj
print(adj_R2)
print(model.summary())
assert len(model.params.index) == 2, 'Check 3b, Number of model parameters (including intercept) does not match. Did you make a univariate model?'

adj_R2_dict = {}
# Iterate over each predictor variable
for predictor in X_train.columns:
    # Fit a linear regression model using the predictor
    model = smf.ols(formula=f"price ~ {predictor}", data=X_train_with_price).fit()
    
    # Store the adjusted R-squared value in the dictionary
    adj_R2_dict[predictor] = model.rsquared_adj

# Sort the dictionary by adjusted R-squared values in descending order
sorted_adj_R2 = sorted(adj_R2_dict.items(), key=lambda x: x[1], reverse=True)
# Print out the list ranking all predictors
print("Ranking of predictors based on adjusted R-squared:")
for i, (predictor, adj_R2) in enumerate(sorted_adj_R2, start=1):
    print(f"{i}. {predictor}: {adj_R2}")

# Print out the top three predictors
top_three_predictors = [pair[0] for pair in sorted_adj_R2[:3]]
print(top_three_predictors)
assert('sqft_above' in top_three_predictors), "Check 3c, the top three list doesn't match."
# print("\nTop three predictors:")
# for i, (predictor, adj_R2) in enumerate(top_three_predictors, start=1):
#  print(f"{i}. {predictor}: {adj_R2}")