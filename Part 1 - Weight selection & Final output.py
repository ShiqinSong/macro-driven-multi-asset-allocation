#!/usr/bin/env python
# coding: utf-8

# # 1. Data Collection and Sample Design
# 
# ## 1.1 Investment Universe
# 
# This study uses seven liquid exchange-traded funds (ETFs) to represent major risky asset classes and the three-month U.S. Treasury bill rate to represent cash.
# 
# | Asset Class | Ticker / Series | Description | Data Source |
# |---|---|---|---|
# | U.S. Equity | SPY | SPDR S&P 500 ETF | Yahoo Finance |
# | International Equity | EFA | iShares MSCI EAFE ETF | Yahoo Finance |
# | U.S. Treasury | IEF | iShares 7–10 Year Treasury Bond ETF | Yahoo Finance |
# | Investment-Grade Credit | LQD | iShares Investment Grade Corporate Bond ETF | Yahoo Finance |
# | High-Yield Credit | HYG | iShares High Yield Corporate Bond ETF | Yahoo Finance |
# | Commodities | DBC | Invesco DB Commodity Index Tracking Fund | Yahoo Finance |
# | Gold | GLD | SPDR Gold Shares | Yahoo Finance |
# | Cash | TB3MS | Three-Month Treasury Bill Rate | FRED |
# 
# QQQ is excluded because it substantially overlaps with SPY and represents an equity-style tilt rather than an independent asset class. Excluding QQQ provides a cleaner asset-class structure for tactical allocation and Brinson attribution.
# 
# ## 1.2 Sample Period
# 
# The complete sample covers:
# 
# - Start date: January 2008
# - End date: December 2025
# - Data frequency: Monthly
# - Portfolio rebalancing frequency: Monthly
# 
# The sample is divided into two periods:
# 
# | Period | Purpose |
# |---|---|
# | January 2008 – December 2012 | Initial training and parameter-estimation period |
# | January 2013 – December 2025 | Out-of-sample backtesting period |
# 
# The 2008 financial crisis is retained in the dataset. It represents genuine market risk and provides useful information for estimating asset volatility, correlations, and macroeconomic sensitivities.
# 
# The initial five-year period is not included in the reported out-of-sample portfolio performance. Instead, it provides the historical observations required to estimate the first set of model parameters.
# 
# ## 1.3 Price and Return Construction
# 
# Daily adjusted ETF prices are downloaded from Yahoo Finance using `yfinance`.
# 
# Adjusted prices are used because they account for stock splits and cash distributions.
# 
# Daily observations are downloaded first and then converted to month-end observations. This approach provides greater control over month-end alignment than downloading monthly observations directly.
# 
# ## 1.4 Data Quality Controls
# 
# Before portfolio construction, the data will be checked for:
# 
# 1. Missing observations;
# 2. Duplicate dates;
# 3. Different ETF inception dates;
# 4. Extreme or potentially erroneous returns;
# 5. Consistent month-end alignment;
# 6. Look-ahead bias;
# 7. Availability of all assets during the out-of-sample period.

# 
# 
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import yfinance as yf
# 
# from pandas_datareader import data as pdr
# 
# pd.set_option("display.max_columns", 20)
# pd.set_option("display.width", 120)
# pd.options.display.float_format = "{:,.6f}".format

# In[2]:


ASSET_TICKERS = {
    "US_Equity": "SPY",
    "International_Equity": "EFA",
    "Treasury": "IEF",
    "Investment_Grade_Credit": "LQD",
    "High_Yield_Credit": "HYG",
    "Commodities": "DBC",
    "Gold": "GLD",
}

TICKER_TO_ASSET = {
    ticker: asset
    for asset, ticker in ASSET_TICKERS.items()
}

TICKERS = list(ASSET_TICKERS.values())


DOWNLOAD_START = "2007-12-01"
DOWNLOAD_END = "2026-01-01"

SAMPLE_START = "2008-01-01"
SAMPLE_END = "2025-12-31"

TRAINING_END = "2012-12-31"
BACKTEST_START = "2013-01-01"


# In[3]:


raw_etf_data = yf.download(
    tickers=TICKERS,
    start=DOWNLOAD_START,
    end=DOWNLOAD_END,
    interval="1d",
    auto_adjust=True,
    repair=True,
    actions=False,
    group_by="column",
    progress=False,
    threads=True,
)

print("Raw data shape:", raw_etf_data.shape)
print("First date:", raw_etf_data.index.min())
print("Last date:", raw_etf_data.index.max())

raw_etf_data.head()


# In[4]:


# Extract adjusted closing prices


daily_prices = raw_etf_data["Close"].copy()

# Convert Yahoo tickers into descriptive asset-class names.
daily_prices = daily_prices.rename(columns=TICKER_TO_ASSET)

# Ensure the columns appear in the intended order.
daily_prices = daily_prices[list(ASSET_TICKERS.keys())]

# Sort the index and remove any duplicated dates.
daily_prices = daily_prices.sort_index()
daily_prices = daily_prices.loc[
    ~daily_prices.index.duplicated(keep="last")
]

daily_prices.head()


# In[5]:


# ---------------------------------------------------------
# Convert daily prices to month-end prices
# ---------------------------------------------------------

monthly_prices = daily_prices.resample("M").last()

# Retain the complete sample plus December 2007,
# which is required to calculate the January 2008 return.
monthly_prices = monthly_prices.loc[
    "2007-12-31":"2025-12-31"
]

print("Monthly price shape:", monthly_prices.shape)
monthly_prices.head()


# In[6]:


# ---------------------------------------------------------
# Calculate monthly ETF returns

monthly_returns = monthly_prices.pct_change(fill_method=None)

# Keep January 2008 through December 2025.
monthly_returns = monthly_returns.loc[
    SAMPLE_START:SAMPLE_END
]

print("Monthly return shape:", monthly_returns.shape)
monthly_returns.head()


# In[7]:


# ---------------------------------------------------------
# Download the three-month Treasury bill rate from FRED

tbill_yield = pdr.DataReader(
    "TB3MS",
    "fred",
    start=DOWNLOAD_START,
    end=SAMPLE_END,
)

tbill_yield = tbill_yield.rename(
    columns={"TB3MS": "TBill_Annual_Yield"}
)

tbill_yield.head()
# Convert the FRED monthly index to calendar month-end.
tbill_yield.index = tbill_yield.index.to_period("M").to_timestamp("M")

# Make sure the observations are ordered correctly.
tbill_yield = tbill_yield.sort_index()

tbill_yield.head()


# In[8]:


# ---------------------------------------------------------
# Convert the annualized T-bill yield into a monthly return
# ---------------------------------------------------------

annual_tbill_decimal = tbill_yield["TBill_Annual_Yield"] / 100

# Lag the yield by one month to avoid using information
# unavailable at the beginning of the return month.
cash_returns = (
    (1 + annual_tbill_decimal.shift(1)) ** (1 / 12) - 1
).rename("Cash")

cash_returns = cash_returns.loc[
    SAMPLE_START:SAMPLE_END
]

cash_returns.head()


# In[9]:


# ---------------------------------------------------------
# Combine risky-asset returns and cash returns
# ---------------------------------------------------------

all_asset_returns = monthly_returns.join(
    cash_returns,
    how="inner",
)

print("Combined return shape:", all_asset_returns.shape)
all_asset_returns.head()


# In[10]:


# ---------------------------------------------------------
# Calculate risky-asset excess returns
# ---------------------------------------------------------

risky_assets = list(ASSET_TICKERS.keys())

excess_returns = all_asset_returns[risky_assets].sub(
    all_asset_returns["Cash"],
    axis=0,
)

excess_returns.columns = [
    f"{column}_Excess"
    for column in excess_returns.columns
]

excess_returns.head()


# In[11]:


# ---------------------------------------------------------
# Divide the sample into training and backtesting periods
# ---------------------------------------------------------

training_returns = all_asset_returns.loc[
    SAMPLE_START:TRAINING_END
].copy()

backtest_returns = all_asset_returns.loc[
    BACKTEST_START:SAMPLE_END
].copy()

training_excess_returns = excess_returns.loc[
    SAMPLE_START:TRAINING_END
].copy()

backtest_excess_returns = excess_returns.loc[
    BACKTEST_START:SAMPLE_END
].copy()

print(
    "Training period:",
    training_returns.index.min(),
    "to",
    training_returns.index.max(),
)

print(
    "Backtest period:",
    backtest_returns.index.min(),
    "to",
    backtest_returns.index.max(),
)

print("Training observations:", len(training_returns))
print("Backtest observations:", len(backtest_returns))


# In[12]:


# ---------------------------------------------------------
# Basic data-quality checks
# ---------------------------------------------------------

coverage = pd.DataFrame({
    "First valid observation": all_asset_returns.apply(
        lambda x: x.first_valid_index()
    ),
    "Last valid observation": all_asset_returns.apply(
        lambda x: x.last_valid_index()
    ),
    "Number of observations": all_asset_returns.count(),
    "Missing observations": all_asset_returns.isna().sum(),
})

coverage

# Check duplicated dates.
print(
    "Number of duplicated dates:",
    all_asset_returns.index.duplicated().sum(),
)

# Check missing values.
print("\nMissing observations:")
print(all_asset_returns.isna().sum())

# Examine descriptive statistics.
return_summary = all_asset_returns.describe().T

return_summary


# In[13]:


# ---------------------------------------------------------
# Identify unusually large monthly returns
# ---------------------------------------------------------

extreme_return_threshold = 0.30

extreme_return_flags = (
    monthly_returns.abs() > extreme_return_threshold
)

extreme_return_observations = monthly_returns.where(
    extreme_return_flags
).dropna(how="all")

extreme_return_observations


# In[14]:


# ---------------------------------------------------------
# Plot cumulative asset returns
# ---------------------------------------------------------

cumulative_returns = (
    1 + all_asset_returns.fillna(0)
).cumprod()

fig, ax = plt.subplots(figsize=(12, 7))

cumulative_returns.plot(
    ax=ax,
    linewidth=1.5,
)

ax.set_title(
    "Cumulative Growth of $1: January 2008–December 2025"
)
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value")
ax.grid(alpha=0.3)
ax.legend(
    title="Asset Class",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)

plt.tight_layout()
plt.show()


# In[15]:


final_values = cumulative_returns.iloc[-1]

total_returns = final_values - 1

asset_final_results = pd.DataFrame({
    "Final Value of $1": final_values,
    "Total Return": total_returns,
})

asset_final_results = asset_final_results.sort_values(
    by="Total Return",
    ascending=False,
)

asset_final_results.style.format({
    "Final Value of $1": "{:.3f}",
    "Total Return": "{:.2%}",
})


# In[16]:


# ---------------------------------------------------------
# Save the cleaned datasets
# ---------------------------------------------------------

monthly_prices.to_csv("monthly_asset_prices_2007_2025.csv")
all_asset_returns.to_csv("monthly_asset_returns_2008_2025.csv")
excess_returns.to_csv("monthly_excess_returns_2008_2025.csv")
tbill_yield.to_csv("three_month_tbill_yield_2007_2025.csv")


# In[17]:


training_risky_returns = training_returns[
    [
        "US_Equity",
        "International_Equity",
        "Treasury",
        "Investment_Grade_Credit",
        "High_Yield_Credit",
        "Commodities",
        "Gold",
    ]
]

annualized_return = training_risky_returns.mean() * 12
annualized_volatility = training_risky_returns.std() * np.sqrt(12)
correlation_matrix = training_risky_returns.corr()
annualized_covariance = training_risky_returns.cov() * 12

risk_summary = pd.DataFrame({
    "Annualized Return": annualized_return,
    "Annualized Volatility": annualized_volatility,
})

risk_summary


# In[18]:


correlation_matrix


# In[19]:


annualized_covariance


# ## Equal Risk Contribution and Portfolio Weights
# 
# Equal Risk Contribution（ERC）的目标不是让所有资产拥有相同的资金权重，而是让每项资产对整个portfolio贡献相同的风险。
# 
# 本项目包含7项风险资产，因此每项资产的目标risk contribution为：
# 
# $$
# \frac{1}{7}=14.286\%.
# $$
# 
# 也就是说，我们希望：
# 
# $$
# PRC_1=PRC_2=\cdots=PRC_7=14.286\%.
# $$
# 
# 这里需要区分两个概念：
# 
# - **Portfolio weight**：有多少资金投资在这项资产中；
# - **Risk contribution**：这项资产贡献了多少portfolio risk。
# 
# 因此：
# 
# $$
# \text{Portfolio Weight}
# \neq
# \text{Risk Contribution}.
# $$
# 
# 例如，Treasury的波动率较低，所以需要分配更多资金，才能贡献与其他资产相同的风险。International Equity的波动率较高，而且与其他股票类资产相关性较高，因此只需要较低的资金权重，就可能贡献相同的风险。
# 
# ### Portfolio Risk
# 
# 假设portfolio weight vector为：
# 
# $$
# \mathbf{w}
# =
# \begin{bmatrix}
# w_1\\
# w_2\\
# \vdots\\
# w_7
# \end{bmatrix},
# $$
# 
# 资产收益的covariance matrix为：
# 
# $$
# \Sigma.
# $$
# 
# Portfolio variance为：
# 
# $$
# \sigma_P^2
# =
# \mathbf{w}^{\top}\Sigma\mathbf{w}.
# $$
# 
# Portfolio volatility为：
# 
# $$
# \sigma_P
# =
# \sqrt{\mathbf{w}^{\top}\Sigma\mathbf{w}}.
# $$
# 
# Covariance matrix同时包含：
# 
# 1. 每项资产自身的variance；
# 2. 不同资产之间的covariance。
# 
# 所以ERC weights不是只根据单个资产的volatility计算，而是根据完整的covariance matrix计算。
# 
# ### Risk Contribution
# 
# 资产 \(i\) 的marginal risk contribution为：
# 
# $$
# MRC_i
# =
# \frac{(\Sigma\mathbf{w})_i}{\sigma_P}.
# $$
# 
# 它表示当资产 \(i\) 的weight稍微增加时，整个portfolio volatility会增加多少。
# 
# 资产 \(i\) 的total risk contribution为：
# 
# $$
# RC_i
# =
# w_iMRC_i.
# $$
# 
# 代入marginal risk contribution：
# 
# $$
# RC_i
# =
# w_i
# \frac{(\Sigma\mathbf{w})_i}{\sigma_P}.
# $$
# 
# Percentage risk contribution为：
# 
# $$
# PRC_i
# =
# \frac{RC_i}{\sigma_P}.
# $$
# 
# 因为：
# 
# $$
# \sigma_P^2
# =
# \mathbf{w}^{\top}\Sigma\mathbf{w},
# $$
# 
# 所以：
# 
# $$
# PRC_i
# =
# \frac{
# w_i(\Sigma\mathbf{w})_i
# }{
# \mathbf{w}^{\top}\Sigma\mathbf{w}
# }.
# $$
# 
# ### ERC Optimization
# 
# ERC要求每项资产的percentage risk contribution都等于：
# 
# $$
# \frac{1}{7}.
# $$
# 
# 因此，优化器需要寻找一组weights，使：
# 
# $$
# \frac{
# w_i(\Sigma\mathbf{w})_i
# }{
# \mathbf{w}^{\top}\Sigma\mathbf{w}
# }
# =
# \frac{1}{7},
# \qquad i=1,\ldots,7.
# $$
# 
# 优化器最小化以下目标函数：
# 
# $$
# \min_{\mathbf{w}}
# \sum_{i=1}^{7}
# \left[
# \frac{
# w_i(\Sigma\mathbf{w})_i
# }{
# \mathbf{w}^{\top}\Sigma\mathbf{w}
# }
# -
# \frac{1}{7}
# \right]^2.
# $$
# 
# 同时满足：
# 
# $$
# \sum_{i=1}^{7}w_i=1,
# $$
# 
# 以及long-only constraint：
# 
# $$
# w_i\geq0.
# $$
# 
# 优化器首先从equal weights开始。它计算当前每项资产的risk contribution：
# 
# - 如果某项资产的risk contribution高于14.286%，就降低它的weight；
# - 如果某项资产的risk contribution低于14.286%，就提高它的weight；
# - 重复调整，直到所有资产的risk contribution都接近14.286%。
# 
# 使用2008–2012年的monthly returns估计covariance matrix后，得到以下ERC weights：
# 
# | Asset | Capital Weight | Target Risk Contribution |
# |---|---:|---:|
# | U.S. Equity | 9.05% | 14.286% |
# | International Equity | 6.38% | 14.286% |
# | U.S. Treasury | 42.36% | 14.286% |
# | Investment-Grade Credit | 15.40% | 14.286% |
# | High-Yield Credit | 9.88% | 14.286% |
# | Commodities | 7.57% | 14.286% |
# | Gold | 9.36% | 14.286% |
# 
# Treasury虽然获得了42.36%的capital weight，但它只贡献大约14.286%的portfolio risk。International Equity虽然只有6.38%的capital weight，也贡献大约14.286%的portfolio risk。
# 
# 整个计算过程可以总结为：
# 
# $$
# \text{Historical Returns}
# \rightarrow
# \text{Covariance Matrix}
# \rightarrow
# \text{Portfolio Risk}
# \rightarrow
# \text{Risk Contributions}
# \rightarrow
# \text{ERC Weights}.
# $$
# 
# 当前权重属于volatility-based ERC weights，不是直接根据Value at Risk计算的权重。VaR可以在后续作为portfolio risk management metric。

# In[20]:


from scipy.optimize import minimize


# In[21]:


def calculate_risk_contributions(
    weights,
    covariance_matrix
):
    """
    Calculate each asset's percentage contribution
    to total portfolio variance.
    """

    weights = np.asarray(weights)
    covariance_matrix = np.asarray(
        covariance_matrix
    )

    # Portfolio variance
    portfolio_variance = (
        weights.T
        @ covariance_matrix
        @ weights
    )

    # Marginal contribution to portfolio variance
    marginal_variance = (
        covariance_matrix @ weights
    )

    # Component contribution to portfolio variance
    component_variance = (
        weights * marginal_variance
    )

    # Percentage contribution to portfolio variance
    percentage_risk_contribution = (
        component_variance / portfolio_variance
    )

    return percentage_risk_contribution


# In[22]:


def erc_objective(
    weights,
    covariance_matrix,
    target_risk_contribution
):
    """
    Minimize the squared differences between actual
    and target risk contributions.
    """

    actual_risk_contribution = (
        calculate_risk_contributions(
            weights,
            covariance_matrix
        )
    )

    return np.sum(
        (
            actual_risk_contribution
            - target_risk_contribution
        ) ** 2
    )


# In[23]:


# Names of the seven risky assets
asset_names = (
    annualized_covariance.columns.tolist()
)

number_of_assets = len(asset_names)

# Equal target risk contribution: 1/7 per asset
target_risk_contribution = np.repeat(
    1 / number_of_assets,
    number_of_assets
)

# Use equal capital weights as the initial guess
initial_weights = np.repeat(
    1 / number_of_assets,
    number_of_assets
)

# Risky-asset weights must sum to 100%
constraints = {
    "type": "eq",
    "fun": lambda weights: np.sum(weights) - 1
}

# Long-only portfolio
bounds = [
    (0.0001, 1.0)
    for _ in range(number_of_assets)
]

erc_result = minimize(
    fun=erc_objective,
    x0=initial_weights,
    args=(
        annualized_covariance.values,
        target_risk_contribution
    ),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={
        "maxiter": 10000,
        "ftol": 1e-12
    }
)

print("Optimization successful:", erc_result.success)
print("Solver message:", erc_result.message)
print("Objective value:", erc_result.fun)


# In[24]:


erc_risky_sleeve_weights = pd.Series(
    erc_result.x,
    index=asset_names,
    name="ERC_Weight"
)

erc_risky_sleeve_weights


# In[25]:


# Calculate the actual percentage risk contribution
# produced by the optimized ERC weights

actual_risk_contribution = pd.Series(
    calculate_risk_contributions(
        erc_result.x,
        annualized_covariance.values
    ),
    index=asset_names,
    name="Risk_Contribution"
)

actual_risk_contribution


# In[26]:


erc_result.x


# In[27]:


actual_risk_contribution.apply(
    lambda value: f"{value:.4%}"
)


# In[ ]:





# In[28]:


# ---------------------------------------------------------
# Add a fixed 5% cash allocation
# ---------------------------------------------------------

RISKY_ASSET_ALLOCATION = 0.95
CASH_ALLOCATION = 0.05

# Scale the seven risky-asset weights from 100% to 95%
strategic_risky_weights = (
    erc_risky_sleeve_weights
    * RISKY_ASSET_ALLOCATION
)

# Add the fixed 5% cash allocation
strategic_benchmark_weights = pd.concat([
    strategic_risky_weights,
    pd.Series({"Cash": CASH_ALLOCATION})
])

strategic_benchmark_weights.name = (
    "Strategic_Benchmark_Weight"
)

strategic_benchmark_weights


# In[29]:


strategic_benchmark_weights.sum()


# In[30]:


strategic_benchmark_weights.apply(
    lambda value: f"{value:.2%}"
)


# In[31]:


# ---------------------------------------------------------
# Extract monthly asset returns for 2013
# ---------------------------------------------------------

returns_2013 = all_asset_returns.loc[
    "2013-01-01":"2013-12-31",
    strategic_benchmark_weights.index
].copy()

returns_2013


# In[32]:


# ---------------------------------------------------------
# Calculate monthly strategic benchmark returns for 2013
# ---------------------------------------------------------

benchmark_returns_2013 = (
    returns_2013
    .mul(strategic_benchmark_weights, axis=1)
    .sum(axis=1)
)

benchmark_returns_2013.name = (
    "Strategic_Benchmark_Return"
)

benchmark_returns_2013


# In[33]:


benchmark_total_return_2013 = (
    (1 + benchmark_returns_2013).prod() - 1
)

print(
    "2013 Strategic Benchmark Return:",
    f"{benchmark_total_return_2013:.2%}"
)


# In[34]:


benchmark_cumulative_2013 = (
    1 + benchmark_returns_2013
).cumprod()

benchmark_cumulative_2013.name = (
    "Cumulative_Portfolio_Value"
)

benchmark_cumulative_2013


# In[35]:


# ---------------------------------------------------------
# Function for calculating ERC benchmark weights
# ---------------------------------------------------------

def solve_erc_weights(
    covariance_matrix,
    risky_allocation=0.95,
    cash_allocation=0.05
):
    """
    Calculate Equal Risk Contribution weights for risky
    assets and then add a fixed cash allocation.
    """

    covariance_matrix = pd.DataFrame(
        covariance_matrix
    )

    asset_names_local = (
        covariance_matrix.columns.tolist()
    )

    number_of_assets = len(asset_names_local)

    # Equal target risk contribution
    target_risk_contribution = np.repeat(
        1 / number_of_assets,
        number_of_assets
    )

    # Equal weights as the initial guess
    initial_weights = np.repeat(
        1 / number_of_assets,
        number_of_assets
    )

    constraints = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }

    bounds = [
        (0.0001, 1.0)
        for _ in range(number_of_assets)
    ]

    result = minimize(
        fun=erc_objective,
        x0=initial_weights,
        args=(
            covariance_matrix.values,
            target_risk_contribution
        ),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 10000,
            "ftol": 1e-12
        }
    )

    if not result.success:
        raise ValueError(
            f"ERC optimization failed: {result.message}"
        )

    # ERC weights within the risky sleeve
    risky_sleeve_weights = pd.Series(
        result.x,
        index=asset_names_local
    )

    # Scale risky assets from 100% to 95%
    total_portfolio_risky_weights = (
        risky_sleeve_weights * risky_allocation
    )

    # Add the fixed cash allocation
    complete_weights = pd.concat([
        total_portfolio_risky_weights,
        pd.Series({"Cash": cash_allocation})
    ])

    return complete_weights


# In[36]:


# ---------------------------------------------------------
# Generate annual strategic benchmark weights
# ---------------------------------------------------------

BACKTEST_START_YEAR = 2013
BACKTEST_END_YEAR = 2025
ESTIMATION_WINDOW_YEARS = 5

annual_weight_list = []

for portfolio_year in range(
    BACKTEST_START_YEAR,
    BACKTEST_END_YEAR + 1
):
    # Example:
    # 2013 portfolio uses 2008-2012 returns.
    estimation_start_year = (
        portfolio_year - ESTIMATION_WINDOW_YEARS
    )

    estimation_end_year = portfolio_year - 1

    estimation_returns = all_asset_returns.loc[
        f"{estimation_start_year}-01-01":
        f"{estimation_end_year}-12-31",
        asset_names
    ].copy()

    # Confirm that exactly 60 monthly observations are used.
    if len(estimation_returns) != 60:
        raise ValueError(
            f"{portfolio_year}: expected 60 observations, "
            f"but received {len(estimation_returns)}"
        )

    if estimation_returns.isna().any().any():
        raise ValueError(
            f"{portfolio_year}: estimation data contains "
            "missing values"
        )

    annual_covariance = (
        estimation_returns.cov() * 12
    )

    weights_for_year = solve_erc_weights(
        covariance_matrix=annual_covariance,
        risky_allocation=0.95,
        cash_allocation=0.05
    )

    weights_for_year.name = portfolio_year

    annual_weight_list.append(
        weights_for_year
    )

annual_benchmark_weights = pd.DataFrame(
    annual_weight_list
)

annual_benchmark_weights.index.name = "Portfolio_Year"

annual_benchmark_weights


# In[37]:


annual_benchmark_weights.sum(axis=1)


# In[38]:


# ---------------------------------------------------------
# Expand annual weights into monthly weights
# ---------------------------------------------------------

benchmark_return_data = all_asset_returns.loc[
    "2013-01-01":"2025-12-31",
    annual_benchmark_weights.columns
].copy()

monthly_benchmark_weights = pd.DataFrame(
    index=benchmark_return_data.index,
    columns=annual_benchmark_weights.columns,
    dtype=float
)

for date in monthly_benchmark_weights.index:
    portfolio_year = date.year

    monthly_benchmark_weights.loc[date] = (
        annual_benchmark_weights.loc[
            portfolio_year
        ]
    )

monthly_benchmark_weights.head(15)


# In[39]:


monthly_benchmark_weights.loc[
    "2013-12-31":"2014-01-31"
].style.format("{:.2%}")


# In[40]:


# ---------------------------------------------------------
# Calculate monthly return contributions
# ---------------------------------------------------------

monthly_asset_contributions = (
    benchmark_return_data
    * monthly_benchmark_weights
)

monthly_asset_contributions.head()


# In[41]:


# ---------------------------------------------------------
# Calculate monthly benchmark returns
# ---------------------------------------------------------

strategic_benchmark_returns = (
    monthly_asset_contributions.sum(axis=1)
)

strategic_benchmark_returns.name = (
    "Strategic_ERC_Benchmark"
)

strategic_benchmark_returns.head()


# In[42]:


# ---------------------------------------------------------
# Calculate cumulative portfolio value
# ---------------------------------------------------------

strategic_benchmark_cumulative = (
    1 + strategic_benchmark_returns
).cumprod()

strategic_benchmark_cumulative.name = (
    "Strategic_ERC_Benchmark"
)

strategic_benchmark_cumulative


# In[43]:


# ---------------------------------------------------------
# Plot the strategic benchmark cumulative return
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 7))

strategic_benchmark_cumulative.plot(
    ax=ax,
    color="navy",
    linewidth=2.2,
    label="Strategic ERC Benchmark"
)

ax.axhline(
    y=1,
    color="black",
    linestyle="--",
    linewidth=1,
    alpha=0.7
)

ax.set_title(
    "Strategic ERC Benchmark: Growth of $1 "
    "(2013–2025)"
)
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value")
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()


# In[44]:


# ---------------------------------------------------------
# Calculate benchmark performance statistics
# ---------------------------------------------------------

number_of_months = len(
    strategic_benchmark_returns
)

number_of_years = number_of_months / 12

final_portfolio_value = (
    strategic_benchmark_cumulative.iloc[-1]
)

total_return = (
    final_portfolio_value - 1
)

cagr = (
    final_portfolio_value
    ** (1 / number_of_years)
    - 1
)

annualized_volatility = (
    strategic_benchmark_returns.std()
    * np.sqrt(12)
)

# Sharpe ratio using the monthly cash return
benchmark_excess_returns = (
    strategic_benchmark_returns
    - benchmark_return_data["Cash"]
)

sharpe_ratio = (
    benchmark_excess_returns.mean() * 12
) / (
    benchmark_excess_returns.std()
    * np.sqrt(12)
)

# Drawdown calculation
running_peak = (
    strategic_benchmark_cumulative.cummax()
)

drawdown = (
    strategic_benchmark_cumulative
    / running_peak
    - 1
)

maximum_drawdown = drawdown.min()

performance_summary = pd.Series({
    "Initial Portfolio Value": 1.0,
    "Final Portfolio Value": final_portfolio_value,
    "Total Return": total_return,
    "CAGR": cagr,
    "Annualized Volatility": annualized_volatility,
    "Sharpe Ratio": sharpe_ratio,
    "Maximum Drawdown": maximum_drawdown
})

performance_summary


# In[45]:


fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(13, 9),
    sharex=True,
    gridspec_kw={
        "height_ratios": [2, 1]
    }
)

strategic_benchmark_cumulative.plot(
    ax=axes[0],
    color="navy",
    linewidth=2,
    label="Strategic ERC Benchmark"
)

axes[0].axhline(
    1,
    color="black",
    linestyle="--",
    linewidth=1
)

axes[0].set_title(
    "Strategic ERC Benchmark Performance "
    "(2013–2025)"
)
axes[0].set_ylabel("Growth of $1")
axes[0].grid(alpha=0.3)
axes[0].legend()

drawdown.plot(
    ax=axes[1],
    color="darkred",
    linewidth=1.5
)

axes[1].fill_between(
    drawdown.index,
    drawdown.values,
    0,
    color="darkred",
    alpha=0.25
)

axes[1].set_title("Portfolio Drawdown")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Drawdown")
axes[1].yaxis.set_major_formatter(
    plt.FuncFormatter(
        lambda value, position: f"{value:.0%}"
    )
)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# In[ ]:





# In[46]:


strategic_benchmark_weights


# In[47]:


# ---------------------------------------------------------
# Fixed ERC benchmark
# Use the initial 2013 weights throughout 2013-2025
# ---------------------------------------------------------

fixed_benchmark_return_data = all_asset_returns.loc[
    "2013-01-01":"2025-12-31",
    strategic_benchmark_weights.index
].copy()

fixed_erc_returns = (
    fixed_benchmark_return_data
    .mul(strategic_benchmark_weights, axis=1)
    .sum(axis=1)
)

fixed_erc_returns.name = "Fixed_ERC_Benchmark"

fixed_erc_returns.head()


# In[48]:


fixed_erc_cumulative = (
    1 + fixed_erc_returns
).cumprod()

fixed_erc_cumulative.name = (
    "Fixed ERC Benchmark"
)


# In[49]:


strategic_benchmark_cumulative


# In[50]:


benchmark_comparison = pd.concat(
    [
        fixed_erc_cumulative,
        strategic_benchmark_cumulative.rename(
            "Rolling ERC Benchmark"
        )
    ],
    axis=1
)

benchmark_comparison.head()


# In[51]:


fig, ax = plt.subplots(figsize=(13, 7))

benchmark_comparison.plot(
    ax=ax,
    linewidth=2
)

ax.axhline(
    y=1,
    color="black",
    linestyle="--",
    linewidth=1,
    alpha=0.7
)

ax.set_title(
    "Fixed versus Annually Rebalanced ERC Benchmark "
    "(2013–2025)"
)
ax.set_xlabel("Date")
ax.set_ylabel("Growth of $1")
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()


# In[52]:


comparison_final_value = (
    benchmark_comparison.iloc[-1]
)

comparison_total_return = (
    comparison_final_value - 1
)

comparison_results = pd.DataFrame({
    "Final Value of $1": comparison_final_value,
    "Total Return": comparison_total_return
})

comparison_results.style.format({
    "Final Value of $1": "${:.4f}",
    "Total Return": "{:.2%}"
})


# In[53]:


def calculate_basic_performance(return_series):
    number_of_years = len(return_series) / 12

    final_value = (
        (1 + return_series).prod()
    )

    total_return = final_value - 1

    cagr = (
        final_value ** (1 / number_of_years) - 1
    )

    annualized_volatility = (
        return_series.std() * np.sqrt(12)
    )

    cumulative_value = (
        1 + return_series
    ).cumprod()

    drawdown = (
        cumulative_value
        / cumulative_value.cummax()
        - 1
    )

    maximum_drawdown = drawdown.min()

    return pd.Series({
        "Final Value": final_value,
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Volatility": annualized_volatility,
        "Maximum Drawdown": maximum_drawdown
    })
benchmark_performance_comparison = pd.DataFrame({
    "Fixed ERC": calculate_basic_performance(
        fixed_erc_returns
    ),

    "Rolling ERC": calculate_basic_performance(
        strategic_benchmark_returns
    )
}).T

benchmark_performance_comparison.style.format({
    "Final Value": "${:.4f}",
    "Total Return": "{:.2%}",
    "CAGR": "{:.2%}",
    "Annualized Volatility": "{:.2%}",
    "Maximum Drawdown": "{:.2%}"
})


# In[54]:


# ---------------------------------------------------------
# Export Part 1 results for tactical allocation
# ---------------------------------------------------------

monthly_benchmark_weights.to_csv(
    "monthly_rolling_erc_weights_2013_2025.csv"
)

strategic_benchmark_returns.to_csv(
    "rolling_erc_returns_2013_2025.csv"
)

fixed_erc_returns.to_csv(
    "fixed_erc_returns_2013_2025.csv"
)

benchmark_return_data.to_csv(
    "monthly_all_asset_returns_2013_2025.csv"
)

print("Saved: monthly_rolling_erc_weights_2013_2025.csv")
print("Saved: rolling_erc_returns_2013_2025.csv")
print("Saved: fixed_erc_returns_2013_2025.csv")
print("Saved: monthly_all_asset_returns_2013_2025.csv")


# In[55]:


print("Monthly Rolling ERC weight shape:")
print(monthly_benchmark_weights.shape)

print("\nFirst date:")
print(monthly_benchmark_weights.index.min())

print("\nLast date:")
print(monthly_benchmark_weights.index.max())

print("\nWeight sums:")
print(monthly_benchmark_weights.sum(axis=1).describe())

monthly_benchmark_weights.head()

