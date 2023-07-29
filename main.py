import pandas as pd

weather = pd.read_csv("data.csv", index_col="DATE")

# REMOVING NULL VALUES

null_pct = weather.apply(pd.isnull).sum() / weather.shape[0]  # the % of null values

valid_columns = weather.columns[null_pct < .05]  # selects all columns where number of null values < 5%

weather = weather[valid_columns].copy()

weather.columns = weather.columns.str.lower()  # makes all column names lowercased

weather = weather.ffill()  # fills missing values with the most recently-known value

weather.index = pd.to_datetime(weather.index)  # converts from string to date format

weather["snwd"].plot()  # creates a bar plot showing snow depth by day

weather["target"] = weather.shift(-1)[
    "tmax"]  # pulls the values from the next row back (i.e. the target column shows tomorrow's temperature.

# Obivously, this can't be done for the last day in the dataset so instead use ffill to give it a fitting value

weather = weather.ffill()

#  ALGORITHM - ridge regression model

from sklearn.linear_model import Ridge

# w = weather.corr()  # tests for linear correlation

rr = Ridge(
    alpha=.1)  # initialising ridge regression model. The value determines how much the coefficients are shrunk to account for colinearlarity

predictors = weather.columns[~weather.columns.isin(
    ["target", "name", "station"])]  # fetches all columns except those specified (using ~ the negation operator)


#  cross validation cannot be used with time series data such as this because you cannot use the future to predict the past

#  thus, we are going to use backtesting

def backtest(weather, model, predictors, start=3650,
             step=90):  # start: how many years of data we want to use before we start predicting | step: how many days at a time you make predictions for
    all_predictions = []

    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i, :]  # all data before the current row
        test = weather.iloc[i:(i + step), :]  # the next 90 days to make predictions on

        model.fit(train[predictors], train["target"])

        preds = model.predict(test[predictors])

        preds = pd.Series(preds, index=test.index)  # easier to work with than a numpy array
        combined = pd.concat([test["target"], preds],
                             axis=1)  # converts into a single data frame | axis=1 means treat each of these as a separate column in a single df

        combined.columns = ["actual", "prediction"]

        combined["diff"] = (combined["prediction"] - combined[
            "actual"]).abs()  # the difference between what is predicted and what actually happened

        all_predictions.append(combined)

    return pd.concat(all_predictions)  # axis=0 by default


# CREATING PREDICTIONS

predictions = backtest(weather, rr, predictors)

print(predictions)

from sklearn.metrics import mean_absolute_error

mean_absolute_error(predictions["actual"], predictions["prediction"])  # takes all the differences and finds the average


# IMPROVING THE ALGORITHM USING PREDICTORS

# start by calculating the average precipitation and temp in the past few days

def pct_diff(old, new):
    return (new - old) / old  # percentage difference


def compute_rolling(weather, horizon, col):  # horizon: the number of days we want to compute the rolling average for
    label = f"rolling_{horizon}_{col}"  # format string

    weather[label] = weather[col].rolling(horizon).mean()
    weather[f"{label}_pct"] = pct_diff(weather[label],
                                       weather[col])  # percentage difference between the current and rolling day
    return weather


rolling_horizons = [3, 14]

for horizon in rolling_horizons:
    for col in ["tmax", "tmin", "prcp"]:
        weather = compute_rolling(weather, horizon, col)

weather = weather.iloc[14:, :]  # gets rid of the first 14 rows as they have missing vals

weather = weather.fillna(0)


def expand_mean(df):
    return df.expanding(1).mean()  # returns mean of current row and all rows before

for col in ["tmax", "tmin", "prcp"]:
    weather[f"month_avg_{col}"] = weather[col].groupby(weather.index.month, group_keys=False).apply(expand_mean)  # groups all the values together by month
    weather[f"month_avg_{col}"] = weather[col].groupby(weather.index.day_of_year, group_keys=False).apply(expand_mean)  # same as above but by year

predictions = backtest(weather, rr, predictors)
error = mean_absolute_error(predictions["actual"], predictions["prediction"])

print(predictions.sort_values("diff", ascending=False))

