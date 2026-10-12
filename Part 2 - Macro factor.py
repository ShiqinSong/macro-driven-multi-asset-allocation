#!/usr/bin/env python
# coding: utf-8

# In[82]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas_datareader import data as pdr

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 140)
pd.options.display.float_format = "{:,.6f}".format


# In[83]:


# ---------------------------------------------------------
# Macro data configuration
# ---------------------------------------------------------

MACRO_DOWNLOAD_START = "2001-01-01"
MACRO_SAMPLE_END = "2025-12-31"

# Monthly economic-data series
MONTHLY_ECONOMIC_SERIES = {
    "INDPRO": "Industrial_Production",
    "CPIAUCSL": "CPI"
}

# Daily market-based series
DAILY_MARKET_SERIES = {
    "DGS10": "Ten_Year_Treasury_Yield",
    "BAA10Y": "Baa_Credit_Spread",
    "VIXCLS": "VIX"
}


# In[84]:


# Download monthly economic data from FRED

economic_data_monthly = pdr.DataReader(
    list(MONTHLY_ECONOMIC_SERIES.keys()),
    "fred",
    start=MACRO_DOWNLOAD_START,
    end=MACRO_SAMPLE_END
)

economic_data_monthly = economic_data_monthly.rename(
    columns=MONTHLY_ECONOMIC_SERIES
)

economic_data_monthly = (
    economic_data_monthly
    .sort_index()
)

print("Monthly economic data")
print("---------------------")
print("Shape:", economic_data_monthly.shape)
print("First date:", economic_data_monthly.index.min())
print("Last date:", economic_data_monthly.index.max())

economic_data_monthly.head()


# In[85]:


# Convert economic-data dates to calendar month-end

economic_data_monthly.index = (
    economic_data_monthly.index
    .to_period("M")
    .to_timestamp("M")
)

economic_data_monthly = (
    economic_data_monthly
    .sort_index()
)

economic_data_monthly.head()


# In[86]:


# Download daily market-based data from FRED

market_data_daily = pdr.DataReader(
    list(DAILY_MARKET_SERIES.keys()),
    "fred",
    start=MACRO_DOWNLOAD_START,
    end=MACRO_SAMPLE_END
)

market_data_daily = market_data_daily.rename(
    columns=DAILY_MARKET_SERIES
)

market_data_daily = (
    market_data_daily
    .sort_index()
)

print("Daily market data")
print("-----------------")
print("Shape:", market_data_daily.shape)
print("First date:", market_data_daily.index.min())
print("Last date:", market_data_daily.index.max())

market_data_daily.head()


# In[87]:


# Check daily market-data coverage

daily_market_coverage = pd.DataFrame({
    "First Valid Date":
        market_data_daily.apply(
            lambda series: series.first_valid_index()
        ),

    "Last Valid Date":
        market_data_daily.apply(
            lambda series: series.last_valid_index()
        ),

    "Valid Observations":
        market_data_daily.count(),

    "Missing Daily Observations":
        market_data_daily.isna().sum()
})

daily_market_coverage


# In[88]:


# Convert daily market data to month-end observations

market_data_monthly = (
    market_data_daily
    .resample("M")
    .last()
)

market_data_monthly = (
    market_data_monthly
    .sort_index()
)

print("Monthly market data")
print("-------------------")
print("Shape:", market_data_monthly.shape)
print("First date:", market_data_monthly.index.min())
print("Last date:", market_data_monthly.index.max())

market_data_monthly.head()


# In[89]:


# ---------------------------------------------------------
# Combine economic and market-based macro data
# ---------------------------------------------------------

macro_raw_monthly = economic_data_monthly.join(
    market_data_monthly,
    how="outer"
)

macro_raw_monthly = (
    macro_raw_monthly
    .sort_index()
    .loc[
        MACRO_DOWNLOAD_START:
        MACRO_SAMPLE_END
    ]
)

# Put columns in the intended order.
macro_raw_monthly = macro_raw_monthly[
    [
        "Industrial_Production",
        "CPI",
        "Ten_Year_Treasury_Yield",
        "Baa_Credit_Spread",
        "VIX"
    ]
]

print("Combined macro data")
print("-------------------")
print("Shape:", macro_raw_monthly.shape)
print("First date:", macro_raw_monthly.index.min())
print("Last date:", macro_raw_monthly.index.max())

macro_raw_monthly.head()


# In[90]:


# Check monthly macro-data coverage

macro_data_coverage = pd.DataFrame({
    "First Valid Date":
        macro_raw_monthly.apply(
            lambda series: series.first_valid_index()
        ),

    "Last Valid Date":
        macro_raw_monthly.apply(
            lambda series: series.last_valid_index()
        ),

    "Valid Observations":
        macro_raw_monthly.count(),

    "Missing Monthly Observations":
        macro_raw_monthly.isna().sum()
})

macro_data_coverage


# In[91]:


macro_raw_monthly.loc[
    macro_raw_monthly["CPI"].isna(),
    ["CPI"]
]


# In[92]:


# Check duplicated dates

number_of_duplicated_dates = (
    macro_raw_monthly
    .index
    .duplicated()
    .sum()
)

print(
    "Number of duplicated dates:",
    number_of_duplicated_dates
)


# In[93]:


# Plot raw macroeconomic series

fig, axes = plt.subplots(
    nrows=5,
    ncols=1,
    figsize=(13, 17),
    sharex=True
)

plot_colors = [
    "navy",
    "darkred",
    "darkgreen",
    "purple",
    "darkorange"
]

for column, ax, color in zip(
    macro_raw_monthly.columns,
    axes,
    plot_colors
):
    macro_raw_monthly[column].plot(
        ax=ax,
        color=color,
        linewidth=1.4
    )

    ax.set_title(column)
    ax.set_ylabel("Raw Value")
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("Date")

plt.tight_layout()
plt.show()


# In[ ]:





# In[94]:


# ---------------------------------------------------------
# Save the cleaned raw macro data
# ---------------------------------------------------------

macro_raw_monthly.to_csv(
    "macro_raw_monthly_2002_2025.csv"
)

print(
    "Saved: macro_raw_monthly_2002_2025.csv"
)


# In[ ]:





# In[ ]:





# In[95]:


# ---------------------------------------------------------
# Initialize the macro-factor DataFrame
# ---------------------------------------------------------

macro_factors = pd.DataFrame(
    index=macro_raw_monthly.index
)


# In[96]:


# ---------------------------------------------------------
# Growth factor: Industrial Production YoY growth
# ---------------------------------------------------------

growth_raw = (
    macro_raw_monthly["Industrial_Production"]
    .pct_change(
        periods=12,
        fill_method=None
    )
    * 100
)

growth_raw.name = "Growth_Raw"

growth_raw.head(15)


# In[97]:


# Lag by one month to reflect publication timing.
macro_factors["Growth"] = (
    growth_raw.shift(1)
)


# In[98]:


# ---------------------------------------------------------
# Inflation factor: CPI YoY inflation
# ---------------------------------------------------------

inflation_raw = (
    macro_raw_monthly["CPI"]
    .pct_change(
        periods=12,
        fill_method=None
    )
    * 100
)

inflation_raw.name = "Inflation_Raw"

inflation_raw.head(15)


# In[99]:


# Lag by one month to reflect CPI publication timing.
inflation_available = (
    inflation_raw.shift(1)
)

# If one monthly signal is unavailable, continue using
# the most recently available inflation signal.
inflation_available = (
    inflation_available.ffill(limit=1)
)

macro_factors["Inflation"] = (
    inflation_available
)


# In[100]:


inflation_check = pd.DataFrame({
    "CPI": macro_raw_monthly["CPI"],
    "Inflation_Raw": inflation_raw,
    "Inflation_Available": macro_factors["Inflation"]
})

inflation_check.loc[
    "2025-08-31":"2025-12-31"
]


# In[101]:


# ---------------------------------------------------------
# Interest-rate factor: monthly change in 10-year yield
# ---------------------------------------------------------

macro_factors["Rates"] = (
    macro_raw_monthly[
        "Ten_Year_Treasury_Yield"
    ].diff()
)


# In[102]:


# ---------------------------------------------------------
# Credit factor: monthly change in Baa credit spread
# ---------------------------------------------------------

macro_factors["Credit"] = (
    macro_raw_monthly[
        "Baa_Credit_Spread"
    ].diff()
)


# In[103]:


# ---------------------------------------------------------
# Risk-sentiment factor: monthly change in VIX
# ---------------------------------------------------------

macro_factors["Sentiment"] = (
    macro_raw_monthly["VIX"].diff()
)


# In[104]:


# ---------------------------------------------------------
# Retain the complete macro-factor sample
# ---------------------------------------------------------

macro_factors = macro_factors.loc[
    "2002-02-01":"2025-12-31"
].copy()

print("Macro-factor shape:", macro_factors.shape)
print("First date:", macro_factors.index.min())
print("Last date:", macro_factors.index.max())
print("Missing observations:")
print(macro_factors.isna().sum())

macro_factors.head()


# In[105]:


# ---------------------------------------------------------
# Inspect the constructed macro factors
# ---------------------------------------------------------

macro_factors.head(15)


# In[ ]:





# In[106]:


# ---------------------------------------------------------
# Check missing factor observations
# ---------------------------------------------------------

factor_coverage = pd.DataFrame({
    "First Valid Date":
        macro_factors.apply(
            lambda series: series.first_valid_index()
        ),

    "Last Valid Date":
        macro_factors.apply(
            lambda series: series.last_valid_index()
        ),

    "Valid Observations":
        macro_factors.count(),

    "Missing Observations":
        macro_factors.isna().sum()
})

factor_coverage


# In[107]:


# ---------------------------------------------------------
# Macro-factor descriptive statistics
# ---------------------------------------------------------

macro_factor_summary = (
    macro_factors
    .describe()
    .T
)

macro_factor_summary


# In[108]:


# ---------------------------------------------------------
# Plot the constructed macro factors
# ---------------------------------------------------------

fig, axes = plt.subplots(
    nrows=5,
    ncols=1,
    figsize=(13, 17),
    sharex=True
)

factor_colors = {
    "Growth": "navy",
    "Inflation": "darkred",
    "Rates": "darkgreen",
    "Credit": "purple",
    "Sentiment": "darkorange"
}

for factor, ax in zip(
    macro_factors.columns,
    axes
):
    macro_factors[factor].plot(
        ax=ax,
        color=factor_colors[factor],
        linewidth=1.3
    )

    ax.axhline(
        y=0,
        color="black",
        linestyle="--",
        linewidth=0.8
    )

    ax.set_title(factor)
    ax.set_ylabel("Raw Factor Value")
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("Date")

plt.tight_layout()
plt.show()


# In[109]:


# ---------------------------------------------------------
# Save the constructed macro factors
# ---------------------------------------------------------

macro_factors.to_csv(
    "macro_factors_raw_2003_2025.csv"
)

print(
    "Saved: macro_factors_raw_2003_2025.csv"
)


# In[110]:


macro_factors


# In[111]:


# ---------------------------------------------------------
# Calculate rolling means and standard deviations
# ---------------------------------------------------------
STANDARDIZATION_WINDOW = 60



rolling_factor_mean = (
    macro_factors
    .rolling(
        window=STANDARDIZATION_WINDOW,
        min_periods=STANDARDIZATION_WINDOW
    )
    .mean()
)

rolling_factor_std = (
    macro_factors
    .rolling(
        window=STANDARDIZATION_WINDOW,
        min_periods=STANDARDIZATION_WINDOW
    )
    .std()
)
# ---------------------------------------------------------
# Calculate rolling macro-factor z-scores
# ---------------------------------------------------------

macro_factors_z = (
    macro_factors - rolling_factor_mean
) / rolling_factor_std

macro_factors_z.head(65)


# In[112]:


# ---------------------------------------------------------
# Retain the study period
# ---------------------------------------------------------

macro_factors_z = macro_factors_z.loc[
    "2007-12-01":"2025-12-31"
].copy()

print("Z-score factor shape:", macro_factors_z.shape)
print("First date:", macro_factors_z.index.min())
print("Last date:", macro_factors_z.index.max())

macro_factors_z.head()


# In[113]:


# ---------------------------------------------------------
# Check standardized-factor coverage
# ---------------------------------------------------------

zscore_coverage = pd.DataFrame({
    "First Valid Date":
        macro_factors_z.apply(
            lambda series: series.first_valid_index()
        ),

    "Last Valid Date":
        macro_factors_z.apply(
            lambda series: series.last_valid_index()
        ),

    "Valid Observations":
        macro_factors_z.count(),

    "Missing Observations":
        macro_factors_z.isna().sum()
})

zscore_coverage


# In[114]:


macro_factors_z


# In[115]:


factor_comparison = pd.concat(
    {
        "Raw": macro_factors,
        "Z_Score": macro_factors_z
    },
    axis=1
)

factor_comparison.tail(12)


# In[116]:


# ---------------------------------------------------------
# Plot standardized macro factors
# ---------------------------------------------------------

fig, axes = plt.subplots(
    nrows=5,
    ncols=1,
    figsize=(13, 17),
    sharex=True
)

factor_colors = {
    "Growth": "navy",
    "Inflation": "darkred",
    "Rates": "darkgreen",
    "Credit": "purple",
    "Sentiment": "darkorange"
}

for factor, ax in zip(
    macro_factors_z.columns,
    axes
):
    macro_factors_z[factor].plot(
        ax=ax,
        color=factor_colors[factor],
        linewidth=1.3
    )

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.9
    )

    ax.axhline(
        2,
        color="red",
        linestyle=":",
        linewidth=0.8
    )

    ax.axhline(
        -2,
        color="red",
        linestyle=":",
        linewidth=0.8
    )

    ax.set_title(f"{factor}: 60-Month Rolling Z-Score")
    ax.set_ylabel("Z-Score")
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("Date")

plt.tight_layout()
plt.show()


# In[117]:


# ---------------------------------------------------------
# Save standardized macro factors
# ---------------------------------------------------------

macro_factors_z.to_csv(
    "macro_factors_zscore_2008_2025.csv"
)

print(
    "Saved: macro_factors_zscore_2008_2025.csv"
)


# In[119]:


# ---------------------------------------------------------
# Load monthly asset excess returns
# ---------------------------------------------------------

asset_excess_returns = pd.read_csv(
    "monthly_excess_returns_2008_2025.csv",
    index_col=0,
    parse_dates=True
)

asset_excess_returns.index.name = "DATE"

asset_excess_returns = (
    asset_excess_returns
    .sort_index()
    .loc["2008-01-01":"2025-12-31"]
)

print("Asset excess-return shape:", asset_excess_returns.shape)
print("First date:", asset_excess_returns.index.min())
print("Last date:", asset_excess_returns.index.max())

asset_excess_returns.head()


# In[120]:


# ---------------------------------------------------------
# Inspect asset names and missing values
# ---------------------------------------------------------

print("Asset columns:")
print(asset_excess_returns.columns.tolist())

print("\nMissing observations:")
print(asset_excess_returns.isna().sum())


# In[121]:


# ---------------------------------------------------------
# Align each next-month return with the preceding factor date
# ---------------------------------------------------------

next_month_excess_returns = asset_excess_returns.copy()

next_month_excess_returns.index = (
    next_month_excess_returns.index
    - pd.offsets.MonthEnd(1)
)

next_month_excess_returns.index.name = "Factor_Date"

print("Aligned return shape:", next_month_excess_returns.shape)
print("First factor date:", next_month_excess_returns.index.min())
print("Last factor date:", next_month_excess_returns.index.max())

next_month_excess_returns.head()


# In[122]:


# ---------------------------------------------------------
# Combine macro factors with next-month excess returns
# ---------------------------------------------------------

macro_factors_for_model = macro_factors_z.copy()
macro_factors_for_model.index.name = "Factor_Date"

model_data = macro_factors_for_model.join(
    next_month_excess_returns,
    how="inner"
)

print("Combined model-data shape:", model_data.shape)
print("First factor date:", model_data.index.min())
print("Last factor date:", model_data.index.max())
print("Total missing observations:", model_data.isna().sum().sum())

model_data.head()


# In[123]:


# ---------------------------------------------------------
# Verify factor-to-next-month-return alignment
# ---------------------------------------------------------

alignment_check = pd.DataFrame({
    "Factor_Date": model_data.index,
    "Return_Month": (
        model_data.index
        + pd.offsets.MonthEnd(1)
    )
})

alignment_check.head(12)


# In[125]:


# ---------------------------------------------------------
# Define predictors and prediction targets
# ---------------------------------------------------------

factor_names = [
    "Growth",
    "Inflation",
    "Rates",
    "Credit",
    "Sentiment"
]

asset_names = [
    "US_Equity_Excess",
    "International_Equity_Excess",
    "Treasury_Excess",
    "Investment_Grade_Credit_Excess",
    "High_Yield_Credit_Excess",
    "Commodities_Excess",
    "Gold_Excess"
]

X = model_data[factor_names].copy()
Y = model_data[asset_names].copy()

print("Factor matrix X:", X.shape)
print("Return matrix Y:", Y.shape)

print("\nX sample:")
display(X.head())

print("\nY sample:")
display(Y.head())


# In[126]:


# ---------------------------------------------------------
# Final alignment checks
# ---------------------------------------------------------

print("X first factor date:", X.index.min())
print("X last factor date:", X.index.max())

print("\nY first factor date:", Y.index.min())
print("Y last factor date:", Y.index.max())

print("\nNumber of observations:", len(X))
print("Number of factors:", X.shape[1])
print("Number of assets:", Y.shape[1])

print(
    "Total missing values:",
    X.isna().sum().sum()
    +
    Y.isna().sum().sum()
)

print(
    "X and Y indices identical:",
    X.index.equals(Y.index)
)


# In[127]:


# ---------------------------------------------------------
# Import regression package
# ---------------------------------------------------------

import statsmodels.api as sm


# In[128]:


# ---------------------------------------------------------
# Define the first 60-month training window
# ---------------------------------------------------------

REGRESSION_WINDOW = 60

X_train_first = X.iloc[:REGRESSION_WINDOW].copy()
Y_train_first = Y.iloc[:REGRESSION_WINDOW].copy()

X_predict_first = X.iloc[[REGRESSION_WINDOW]].copy()

print("Training-factor period:")
print(X_train_first.index.min(), "to", X_train_first.index.max())

print("\nCorresponding return period:")
print(
    X_train_first.index.min() + pd.offsets.MonthEnd(1),
    "to",
    X_train_first.index.max() + pd.offsets.MonthEnd(1)
)

print("\nPrediction factor date:")
print(X_predict_first.index[0])

print("\nPredicted return month:")
print(X_predict_first.index[0] + pd.offsets.MonthEnd(1))

print("\nTraining observations:")
print(len(X_train_first))


# In[129]:


# ---------------------------------------------------------
# Add an intercept to the first training window
# ---------------------------------------------------------

X_train_first_with_constant = sm.add_constant(
    X_train_first,
    has_constant="add"
)

X_predict_first_with_constant = sm.add_constant(
    X_predict_first,
    has_constant="add"
)

X_train_first_with_constant.head()


# In[130]:


# ---------------------------------------------------------
# Estimate the first U.S. Equity predictive regression
# ---------------------------------------------------------

first_us_equity_model = sm.OLS(
    Y_train_first["US_Equity_Excess"],
    X_train_first_with_constant
).fit()

first_us_equity_model.summary()


# In[131]:


# ---------------------------------------------------------
# Inspect the estimated coefficients
# ---------------------------------------------------------

first_us_equity_coefficients = (
    first_us_equity_model.params
    .rename("Coefficient")
    .to_frame()
)

first_us_equity_coefficients


# In[132]:


# ---------------------------------------------------------
# Predict January 2013 U.S. Equity excess return
# ---------------------------------------------------------

first_us_equity_prediction = (
    first_us_equity_model.predict(
        X_predict_first_with_constant
    )
)

first_us_equity_prediction.index = (
    first_us_equity_prediction.index
    + pd.offsets.MonthEnd(1)
)

first_us_equity_prediction.name = (
    "Predicted_US_Equity_Excess_Return"
)

first_us_equity_prediction


# In[133]:


# ---------------------------------------------------------
# Compare predicted and realized January 2013 return
# ---------------------------------------------------------

prediction_factor_date = X_predict_first.index[0]

first_prediction_comparison = pd.Series({
    "Predicted Excess Return":
        first_us_equity_model.predict(
            X_predict_first_with_constant
        ).iloc[0],

    "Realized Excess Return":
        Y.loc[
            prediction_factor_date,
            "US_Equity_Excess"
        ],

    "Prediction Error":
        Y.loc[
            prediction_factor_date,
            "US_Equity_Excess"
        ]
        -
        first_us_equity_model.predict(
            X_predict_first_with_constant
        ).iloc[0]
})

first_prediction_comparison.name = "January 2013"

first_prediction_comparison


# In[134]:


# ---------------------------------------------------------
# Estimate the first regression for all seven assets
# ---------------------------------------------------------

first_month_predictions = {}
first_month_realized_returns = {}
first_month_prediction_errors = {}

first_month_coefficients = pd.DataFrame(
    index=[
        "const",
        "Growth",
        "Inflation",
        "Rates",
        "Credit",
        "Sentiment"
    ],
    columns=asset_names,
    dtype=float
)

for asset in asset_names:

    # Estimate the asset-specific regression.
    model = sm.OLS(
        Y_train_first[asset],
        X_train_first_with_constant
    ).fit()

    # Store estimated coefficients.
    first_month_coefficients[asset] = model.params

    # Predict the next-month excess return.
    predicted_return = model.predict(
        X_predict_first_with_constant
    ).iloc[0]

    # Obtain the subsequently realized return.
    realized_return = Y.loc[
        X_predict_first.index[0],
        asset
    ]

    # Store results.
    first_month_predictions[asset] = predicted_return
    first_month_realized_returns[asset] = realized_return
    first_month_prediction_errors[asset] = (
        realized_return - predicted_return
    )


# In[135]:


# ---------------------------------------------------------
# Compare predicted and realized returns for all assets
# ---------------------------------------------------------

first_month_all_asset_comparison = pd.DataFrame({
    "Predicted Excess Return":
        pd.Series(first_month_predictions),

    "Realized Excess Return":
        pd.Series(first_month_realized_returns),

    "Prediction Error":
        pd.Series(first_month_prediction_errors)
})

first_month_all_asset_comparison.index.name = "Asset"

first_month_all_asset_comparison.style.format({
    "Predicted Excess Return": "{:.2%}",
    "Realized Excess Return": "{:.2%}",
    "Prediction Error": "{:.2%}"
})


# In[136]:


# ---------------------------------------------------------
# Display the first-window regression coefficients
# ---------------------------------------------------------

first_month_coefficients.style.format("{:.6f}")


# In[137]:


# ---------------------------------------------------------
# Generate rolling out-of-sample predictions
# ---------------------------------------------------------

REGRESSION_WINDOW = 60

prediction_factor_dates = X.index[
    REGRESSION_WINDOW:
]

rolling_predictions = pd.DataFrame(
    index=prediction_factor_dates,
    columns=asset_names,
    dtype=float
)

rolling_realized_returns = pd.DataFrame(
    index=prediction_factor_dates,
    columns=asset_names,
    dtype=float
)

rolling_model_r_squared = pd.DataFrame(
    index=prediction_factor_dates,
    columns=asset_names,
    dtype=float
)

# Store the time-varying regression coefficients.
coefficient_names = [
    "const",
    "Growth",
    "Inflation",
    "Rates",
    "Credit",
    "Sentiment"
]

rolling_coefficients = {
    asset: pd.DataFrame(
        index=prediction_factor_dates,
        columns=coefficient_names,
        dtype=float
    )
    for asset in asset_names
}

for prediction_position in range(
    REGRESSION_WINDOW,
    len(X)
):

    # Current factor date used to predict next month's return.
    prediction_factor_date = X.index[
        prediction_position
    ]

    # Use only the preceding 60 completed observations.
    training_start = (
        prediction_position
        -
        REGRESSION_WINDOW
    )

    training_end = prediction_position

    X_train = X.iloc[
        training_start:training_end
    ]

    X_predict = X.iloc[
        [prediction_position]
    ]

    X_train_constant = sm.add_constant(
        X_train,
        has_constant="add"
    )

    X_predict_constant = sm.add_constant(
        X_predict,
        has_constant="add"
    )

    for asset in asset_names:

        Y_train = Y[asset].iloc[
            training_start:training_end
        ]

        model = sm.OLS(
            Y_train,
            X_train_constant
        ).fit()

        predicted_return = model.predict(
            X_predict_constant
        ).iloc[0]

        realized_return = Y[asset].iloc[
            prediction_position
        ]

        rolling_predictions.loc[
            prediction_factor_date,
            asset
        ] = predicted_return

        rolling_realized_returns.loc[
            prediction_factor_date,
            asset
        ] = realized_return

        rolling_model_r_squared.loc[
            prediction_factor_date,
            asset
        ] = model.rsquared

        rolling_coefficients[asset].loc[
            prediction_factor_date
        ] = model.params


# In[138]:


# ---------------------------------------------------------
# Convert factor dates to realized return dates
# ---------------------------------------------------------

rolling_predictions.index = (
    rolling_predictions.index
    +
    pd.offsets.MonthEnd(1)
)

rolling_realized_returns.index = (
    rolling_realized_returns.index
    +
    pd.offsets.MonthEnd(1)
)

rolling_model_r_squared.index = (
    rolling_model_r_squared.index
    +
    pd.offsets.MonthEnd(1)
)

for asset in asset_names:

    rolling_coefficients[asset].index = (
        rolling_coefficients[asset].index
        +
        pd.offsets.MonthEnd(1)
    )

rolling_predictions.index.name = "Return_Date"
rolling_realized_returns.index.name = "Return_Date"
rolling_model_r_squared.index.name = "Return_Date"


# In[139]:


# ---------------------------------------------------------
# Check rolling prediction results
# ---------------------------------------------------------

print("Rolling prediction shape:")
print(rolling_predictions.shape)

print("\nFirst predicted return month:")
print(rolling_predictions.index.min())

print("\nLast predicted return month:")
print(rolling_predictions.index.max())

print("\nMissing predicted values:")
print(rolling_predictions.isna().sum())

print("\nPrediction and realized-return indices identical:")
print(
    rolling_predictions.index.equals(
        rolling_realized_returns.index
    )
)

rolling_predictions.head()


# In[140]:


# ---------------------------------------------------------
# Compare early U.S. Equity predictions and realizations
# ---------------------------------------------------------

us_equity_rolling_comparison = pd.DataFrame({
    "Predicted":
        rolling_predictions[
            "US_Equity_Excess"
        ],

    "Realized":
        rolling_realized_returns[
            "US_Equity_Excess"
        ]
})

us_equity_rolling_comparison.head().style.format(
    "{:.2%}"
)


# In[141]:


# ---------------------------------------------------------
# Verify the first prediction month
# ---------------------------------------------------------

first_rolling_month_comparison = pd.DataFrame({
    "Predicted":
        rolling_predictions.iloc[0],

    "Realized":
        rolling_realized_returns.iloc[0]
})

first_rolling_month_comparison["Error"] = (
    first_rolling_month_comparison["Realized"]
    -
    first_rolling_month_comparison["Predicted"]
)

first_rolling_month_comparison.style.format(
    "{:.2%}"
)


# In[142]:


# ---------------------------------------------------------
# Evaluate out-of-sample predictions by asset
# ---------------------------------------------------------

prediction_errors = (
    rolling_realized_returns
    -
    rolling_predictions
)

rmse_by_asset = np.sqrt(
    prediction_errors
    .pow(2)
    .mean()
)

mae_by_asset = (
    prediction_errors
    .abs()
    .mean()
)

prediction_correlation = (
    rolling_predictions
    .corrwith(
        rolling_realized_returns
    )
)

directional_accuracy = (
    np.sign(rolling_predictions)
    ==
    np.sign(rolling_realized_returns)
).mean()

prediction_evaluation = pd.DataFrame({
    "RMSE": rmse_by_asset,
    "MAE": mae_by_asset,
    "Prediction Correlation":
        prediction_correlation,
    "Directional Accuracy":
        directional_accuracy
})

prediction_evaluation.index.name = "Asset"

prediction_evaluation.style.format({
    "RMSE": "{:.2%}",
    "MAE": "{:.2%}",
    "Prediction Correlation": "{:.3f}",
    "Directional Accuracy": "{:.2%}"
})


# In[143]:


# ---------------------------------------------------------
# Calculate monthly cross-sectional Rank IC
# ---------------------------------------------------------

monthly_rank_ic = pd.Series(
    index=rolling_predictions.index,
    dtype=float,
    name="Rank_IC"
)

for return_date in rolling_predictions.index:

    predicted_cross_section = (
        rolling_predictions.loc[return_date]
    )

    realized_cross_section = (
        rolling_realized_returns.loc[return_date]
    )

    monthly_rank_ic.loc[return_date] = (
        predicted_cross_section.corr(
            realized_cross_section,
            method="spearman"
        )
    )

monthly_rank_ic.head()


# In[144]:


# ---------------------------------------------------------
# Summarize cross-sectional Rank IC performance
# ---------------------------------------------------------

rank_ic_mean = monthly_rank_ic.mean()
rank_ic_std = monthly_rank_ic.std()

rank_ic_ir = (
    rank_ic_mean
    /
    rank_ic_std
    *
    np.sqrt(12)
)

positive_rank_ic_ratio = (
    monthly_rank_ic > 0
).mean()

rank_ic_t_statistic = (
    rank_ic_mean
    /
    (
        rank_ic_std
        /
        np.sqrt(monthly_rank_ic.count())
    )
)

rank_ic_summary = pd.Series({
    "Mean Monthly Rank IC":
        rank_ic_mean,

    "Monthly Rank IC Standard Deviation":
        rank_ic_std,

    "Annualized Rank ICIR":
        rank_ic_ir,

    "Positive Rank IC Ratio":
        positive_rank_ic_ratio,

    "Rank IC t-statistic":
        rank_ic_t_statistic
})

rank_ic_summary


# In[146]:


# ---------------------------------------------------------
# Plot monthly Rank IC
# ---------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(13, 5)
)

monthly_rank_ic.plot(
    ax=ax,
    color="navy",
    linewidth=1.1,
    label="Monthly Rank IC"
)

ax.axhline(
    y=0,
    color="black",
    linestyle="--",
    linewidth=0.9
)

ax.axhline(
    y=rank_ic_mean,
    color="darkred",
    linestyle="--",
    linewidth=1.2,
    label=f"Mean Rank IC = {rank_ic_mean:.3f}"
)

ax.set_title(
    "Out-of-Sample Cross-Sectional Rank IC"
)
ax.set_xlabel("Return Month")
ax.set_ylabel("Spearman Rank Correlation")
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()


# In[147]:


# ---------------------------------------------------------
# Newey-West test for the mean monthly Rank IC
# ---------------------------------------------------------

rank_ic_regression_data = monthly_rank_ic.dropna()

rank_ic_constant = np.ones(
    len(rank_ic_regression_data)
)

rank_ic_mean_model = sm.OLS(
    rank_ic_regression_data.values,
    rank_ic_constant
).fit(
    cov_type="HAC",
    cov_kwds={
        "maxlags": 6
    }
)

newey_west_mean_rank_ic = (
    rank_ic_mean_model.params[0]
)

newey_west_t_statistic = (
    rank_ic_mean_model.tvalues[0]
)

newey_west_p_value = (
    rank_ic_mean_model.pvalues[0]
)

newey_west_rank_ic_results = pd.Series({
    "Mean Monthly Rank IC":
        newey_west_mean_rank_ic,

    "Newey-West t-statistic":
        newey_west_t_statistic,

    "Newey-West p-value":
        newey_west_p_value
})

newey_west_rank_ic_results


# In[148]:


# ---------------------------------------------------------
# Save rolling out-of-sample results
# ---------------------------------------------------------

rolling_predictions.to_csv(
    "rolling_predicted_excess_returns_2013_2025.csv"
)

rolling_realized_returns.to_csv(
    "rolling_realized_excess_returns_2013_2025.csv"
)

monthly_rank_ic.to_csv(
    "monthly_rank_ic_2013_2025.csv"
)

print("Saved rolling prediction results.")


# In[ ]:




