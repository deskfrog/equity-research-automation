import wrds
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
from scipy.stats import linregress
from scipy.optimize import minimize

# ==========================================================
# INPUTS 
# ==========================================================

tickers = ["^GSPC", "PLTR", "NVDA", "GE", "AVGO", "LLY", "CAT", "XOM", "GS", "MS"]

MARKET_TABLE = True 
COMPS_TABLE = True
PORTFOLIO_TABLE = True
MACRO_TABLE = True
FUNDAMENTALS_TABLE = True 

TICKERS_FILTER = False

start_year = 2020
weight_cap = 0.45
interval = "1d"

macro_factors = ["^VIX", "GC=F", "^TNX"]
treasury_yield = "^FVX"

if interval == "1d":
    assume_trading_days = 252
elif interval == "1wk":
    assume_trading_days = 52
elif interval == "1mo":
    assume_trading_days = 12

console_formating = 180

# S&P 100
#tickers = ["^GSPC", "AAPL","ABBV","ABT","ACN","ADBE","AIG","AMD","AMGN","AMT","AMZN",
#"AVGO","AXP","BA","BAC","BK","BKNG","BLK","BMY","C",
#"CAT","CL","CMCSA","COF","COP","CRM","CSCO","CVS","CVX",
#"DE","DHR","DIS","DUK","EMR","FDX","GD","GE","GILD","GM",
#"GOOG","GOOGL","GS","HD","HON","IBM","INTC","ISRG","JNJ","JPM",
#"KO","LIN","LLY","LMT","LOW","MA","MCD","MDLZ","MDT","MET",
#"META","MMM","MO","MRK","MS","MSFT","NEE","NFLX","NKE","NOW",
#"NVDA","ORCL","PEP","PFE","PG","PLTR","PM","PYPL","QCOM","RTX",
#"SBUX","SCHW","SO","SPG","T","TGT","TMO","TMUS","TSLA","TXN",
#"UBER","UNH","UNP","UPS","USB","V","VZ","WFC","WMT","XOM"]

# ==========================================================
# FETCH PRICES AND CALCULATIONS 
# ==========================================================

today = dt.datetime.today()
end = today - dt.timedelta(days=1)
start = dt.datetime(start_year, 1, 1)
df_prices = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=False).dropna()
rf = float(yf.download(treasury_yield, start=start, end=end, interval=interval, auto_adjust=False)["Adj Close"].iloc[-1]) / 100 # Treasury Yield 5 Years (US)
#print("\nAdjusted Close Prices:", df_prices.loc[:, ["Adj Close"]])

#-----# RETURNS #-----#

df_log_returns = np.log(df_prices["Adj Close"]).diff().dropna()
df_mean_returns = df_log_returns.mean()
df_variance_returns = df_log_returns.var()
df_stdev_returns = df_log_returns.std()
#print("\nInterval Log Returns:", df_log_returns)
#print("\nInterval Mean:", df_mean_returns)
#print("\n Total Return:", (df_prices["Adj Close"].iloc[-1] / df_prices["Adj Close"].iloc[0]) - 1)
#print("\nInterval Variance:", df_variance_returns)
#print("\nInterval Standard Deviation:", df_stdev_returns)

#-----# BETA AND ALPHA #-----#

index_returns = df_log_returns["^GSPC"]
beta_dict = {}
alpha_dict = {}

for stock in tickers:
    if stock != "^GSPC":
        stock_returns = df_log_returns[stock]
        slope, intercept, r_value, p_value, std_err = linregress(index_returns, stock_returns)
        beta_dict[stock] = slope
        alpha_dict[stock] = intercept
#print("\nBeta:", beta_dict)
#print("\nAlpha:", alpha_dict)

#-----# ANNUALIZED RETURNS METRICS #-----#

annual_mean_returns = df_mean_returns * assume_trading_days
annual_variance_returns = df_variance_returns * assume_trading_days
annual_stdev_returns = df_stdev_returns * np.sqrt(assume_trading_days)
annual_alpha_dict = {stock: alpha * assume_trading_days for stock, alpha in alpha_dict.items()}

#-----# CAPM #-----#

annual_capm_dict = {stock: rf + beta_dict[stock] * (annual_mean_returns["^GSPC"] - rf) for stock in beta_dict.keys()}

#-----# VARIANCE COVARIANCE MATRIX #-----#

stock_tickers = [ticker for ticker in tickers if ticker != "^GSPC"]
df_stock_returns = df_log_returns[stock_tickers]
excess_returns = df_stock_returns - df_stock_returns.mean()
n_obs = len(excess_returns)
variance_covariance_matrix = (excess_returns.T @ excess_returns) / (n_obs - 1)
annualized_variance_covariance_matrix = variance_covariance_matrix * assume_trading_days
#print("\n=== ANNUALIZED COVARIANCE MATRIX ===\n", pd.DataFrame(annualized_variance_covariance_matrix, index=stock_tickers, columns=stock_tickers))

#-----# CORRELATION MATRIX #-----#

correlation_matrix = df_stock_returns.corr()
#print("\n=== CORRELATION MATRIX ===\n", correlation_matrix.to_string())

#-----# CORRELATION WITH MACRO FACTORS #-----#

try:
    df_macro_prices = yf.download(macro_factors, start=start, end=end, interval=interval, auto_adjust=False).dropna()
except Exception:
    df_macro_prices = pd.DataFrame()

if not df_macro_prices.empty:
    df_macro_returns = np.log(df_macro_prices["Adj Close"]).diff().dropna()
else:
    df_macro_returns = pd.DataFrame()

if not df_macro_returns.empty:
    combined_returns = pd.concat([df_stock_returns, df_macro_returns], axis=1)
    correlation_macro = combined_returns.corr().loc[stock_tickers, macro_factors]
    correlation_macro.columns = ["VIX", "Gold", "Interest"]
else:
    correlation_macro = pd.DataFrame(index=stock_tickers, columns=["VIX", "Gold", "Interest"])

# ==========================================================
# PORTFOLIO OPTIMALIZATION 
# ==========================================================

#-----# MINIMUM VARIANCE PORTFOLIO #-----#

def portfolio_variance(weights, cov_matrix):
    return weights @ cov_matrix @ weights

annualized_returns = np.array([annual_mean_returns[ticker] for ticker in stock_tickers])
initial_weights = np.array([1 / len(stock_tickers)] * len(stock_tickers))
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}        # Weight summarize to 1
bounds = tuple((0, weight_cap) for _ in stock_tickers)              # No shorts (negative positions allowed), Not all in one stock

result = minimize(
    portfolio_variance,
    initial_weights,
    args=(annualized_variance_covariance_matrix,),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

min_var_weights = result.x
min_var_portfolio_variance = result.fun
min_var_portfolio_sd = np.sqrt(min_var_portfolio_variance)
min_var_portfolio_return = np.sum(min_var_weights * annualized_returns)
min_var_sharpe_ratio = (min_var_portfolio_return - rf) / min_var_portfolio_sd if min_var_portfolio_sd != 0 else 0

min_var_df = pd.DataFrame({
    "Stock": stock_tickers,
    "Weight": min_var_weights,
    "var(P)": min_var_portfolio_variance,
    "SD(P)": min_var_portfolio_sd,
    "E[R_P]": min_var_portfolio_return,
    "S_P": min_var_sharpe_ratio
})
min_var_df = min_var_df.set_index("Stock")
#print("\n=== MINIMUM VARIANCE PORTFOLIO (Annualized) ===\n", min_var_df.to_string())

#-----# TANGENT PORTFOLIO #-----#

def negative_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
    port_return = np.sum(weights * returns)
    port_variance = weights @ cov_matrix @ weights
    port_sd = np.sqrt(port_variance)
    sharpe = (port_return - risk_free_rate) / port_sd if port_sd != 0 else 0
    return -sharpe      # Negative because we want to maximize

result_tangent = minimize(
    negative_sharpe_ratio,
    initial_weights,
    args=(annualized_returns, annualized_variance_covariance_matrix, rf),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

tangent_weights = result_tangent.x
tangent_portfolio_variance = tangent_weights @ annualized_variance_covariance_matrix @ tangent_weights
tangent_portfolio_sd = np.sqrt(tangent_portfolio_variance)
tangent_portfolio_return = np.sum(tangent_weights * annualized_returns)
tangent_sharpe_ratio = (tangent_portfolio_return - rf) / tangent_portfolio_sd if tangent_portfolio_sd != 0 else 0

tangent_port_df = pd.DataFrame({
    "Stock": stock_tickers,
    "Weight": tangent_weights,
    "var(P)": tangent_portfolio_variance,
    "SD(P)": tangent_portfolio_sd,
    "E[R_P]": tangent_portfolio_return,
    "S_P": tangent_sharpe_ratio
})
tangent_port_df = tangent_port_df.set_index("Stock")
#print("\n=== TANGENT PORTFOLIO (Max Sharpe Ratio - Annualized) ===\n", tangent_port_df.to_string())

#-----# LEAST CORRELATION PORTFOLIO #-----#

def least_correlation_objective(weights, corr_matrix):
    port_corr = weights @ corr_matrix @ weights
    return port_corr

result_least_corr = minimize(
    least_correlation_objective,
    initial_weights,
    args=(correlation_matrix.values,),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)
least_corr_weights = result_least_corr.x
least_corr_portfolio_corr = least_correlation_objective(least_corr_weights, correlation_matrix.values)
least_corr_portfolio_return = np.sum(least_corr_weights * annualized_returns)
least_corr_portfolio_variance = least_corr_weights @ annualized_variance_covariance_matrix @ least_corr_weights
least_corr_portfolio_sd = np.sqrt(least_corr_portfolio_variance)
least_corr_sharpe_ratio = (least_corr_portfolio_return - rf) / least_corr_portfolio_sd if least_corr_portfolio_sd != 0 else 0

least_corr_df = pd.DataFrame({
    "Stock": stock_tickers,
    "Weight": least_corr_weights,
    "Corr(P)": least_corr_portfolio_corr,
    "E[R_P]": least_corr_portfolio_return,
    "var(P)": least_corr_portfolio_variance,
    "SD(P)": least_corr_portfolio_sd,
    "S_P": least_corr_sharpe_ratio
})
least_corr_df = least_corr_df.set_index("Stock")
#print("\n=== LEAST CORRELATION PORTFOLIO (Annualized) ===\n", least_corr_df.to_string())

#-----# MAX RETURN PORTFOLIO #-----#

def negative_return(weights, returns):
    return -np.sum(weights * returns)  # Negative because we maximize

result_max_return = minimize(
    negative_return,
    initial_weights,
    args=(annualized_returns,),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

max_return_weights = result_max_return.x
max_return_portfolio_return = np.sum(max_return_weights * annualized_returns)
max_return_portfolio_variance = max_return_weights @ annualized_variance_covariance_matrix @ max_return_weights
max_return_portfolio_sd = np.sqrt(max_return_portfolio_variance)
max_return_sharpe_ratio = (
    (max_return_portfolio_return - rf) / max_return_portfolio_sd
    if max_return_portfolio_sd != 0 else 0
)

max_return_df = pd.DataFrame({
    "Stock": stock_tickers,
    "Weight": max_return_weights,
    "E[R_P]": max_return_portfolio_return,
    "var(P)": max_return_portfolio_variance,
    "SD(P)": max_return_portfolio_sd,
    "S_P": max_return_sharpe_ratio
}).set_index("Stock")

# ==========================================================
# SUMMARY TABLE 
# ==========================================================

# Calculate total returns for each stock
total_returns = {}
for stock in stock_tickers:
    total_returns[stock] = (df_prices["Adj Close"][stock].iloc[-1] / df_prices["Adj Close"][stock].iloc[0]) - 1

# Calculate market total return
market_total_return = (df_prices["Adj Close"]["^GSPC"].iloc[-1] / df_prices["Adj Close"]["^GSPC"].iloc[0]) - 1

# Calculate Year-To-Date returns (from Jan 1 of current year to last close)
def _get_price_on_or_after(series, dt_obj):
    s = series[series.index >= pd.Timestamp(dt_obj)]
    return s.iloc[0] if not s.empty else series.iloc[0]

start_of_year = dt.datetime(today.year, 1, 1)
ytd_returns = {}
for t in stock_tickers:
    series = df_prices["Adj Close"][t]
    start_price = _get_price_on_or_after(series, start_of_year)
    ytd_returns[t] = (series.iloc[-1] / start_price) - 1
# market ytd
market_start_price = _get_price_on_or_after(df_prices["Adj Close"]["^GSPC"], start_of_year)
market_ytd_return = (df_prices["Adj Close"]["^GSPC"].iloc[-1] / market_start_price) - 1

# Fetch latest prices and market caps for each stock (from Yahoo)
latest_prices = {}
market_caps = {}
for t in stock_tickers:
    # latest adjusted close price
    latest_prices[t] = df_prices["Adj Close"][t].iloc[-1]
    try:
        info = yf.Ticker(t).info
        market_caps[t] = info.get("marketCap", np.nan)
    except Exception:
        market_caps[t] = np.nan

total_market_cap = sum(v for v in market_caps.values() if not pd.isna(v)) if len(market_caps) > 0 else np.nan

summary_data = []

summary_data.append({
    "Ticker": "S&P500",
    "Total Return (%)": market_total_return * 100,
    "YTD Return (%)": market_ytd_return * 100,
    "Mean Return (%)": annual_mean_returns["^GSPC"] * 100,
    "σ (%)": annual_stdev_returns["^GSPC"] * 100,
    "Sharpe Ratio": (annual_mean_returns["^GSPC"] - rf) / annual_stdev_returns["^GSPC"] if annual_stdev_returns["^GSPC"] != 0 else 0,
    "β": 1.0,
    "CAPM (%)": annual_mean_returns["^GSPC"] * 100,
    "α (%)": 0,
    "Market Price": df_prices["Adj Close"]["^GSPC"].iloc[-1],
    "MC ($B)": (total_market_cap / 1e9) if not pd.isna(total_market_cap) else np.nan,
    "MC Proportion (%)": 100.0
})

for stock in stock_tickers:
    mcap = market_caps.get(stock, np.nan)
    pct = (mcap / total_market_cap * 100) if (not pd.isna(mcap) and total_market_cap and total_market_cap > 0) else np.nan
    summary_data.append({
        "Ticker": stock,
        "Total Return (%)": total_returns[stock] * 100,
        "YTD Return (%)": ytd_returns.get(stock, np.nan) * 100,
        "Mean Return (%)": annual_mean_returns[stock] * 100,
        "σ (%)": annual_stdev_returns[stock] * 100,
        "Sharpe Ratio": (annual_mean_returns[stock] - rf) / annual_stdev_returns[stock] if annual_stdev_returns[stock] != 0 else 0,
        "β": beta_dict[stock],
        "CAPM (%)": annual_capm_dict[stock] * 100,
        "α (%)": annual_alpha_dict[stock] * 100,
        "Market Price": latest_prices.get(stock, np.nan),
        "MC ($B)": (mcap / 1e9) if not pd.isna(mcap) else np.nan,
        "MC Proportion (%)": pct
    })
summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.set_index("Ticker")

# Create combined portfolio table with performance metrics first
portfolio_data = {
    "Min Variance": [],
    "Tangent": [],
    "Min Correlation": [],
    "Max Return": []
}

# Add performance metrics
portfolio_data["Min Variance"].append(f"{min_var_portfolio_return*100:.2f} %")
portfolio_data["Tangent"].append(f"{tangent_portfolio_return*100:.2f} %")
portfolio_data["Min Correlation"].append(f"{least_corr_portfolio_return*100:.2f} %")

portfolio_data["Min Variance"].append(f"{min_var_portfolio_sd*100:.2f} %")
portfolio_data["Tangent"].append(f"{tangent_portfolio_sd*100:.2f} %")
portfolio_data["Min Correlation"].append(f"{least_corr_portfolio_sd*100:.2f} %")

portfolio_data["Min Variance"].append(f"{min_var_portfolio_variance:.2f}")
portfolio_data["Tangent"].append(f"{tangent_portfolio_variance:.2f}")
portfolio_data["Min Correlation"].append(f"{least_corr_portfolio_variance:.2f}")

portfolio_data["Min Variance"].append(f"{min_var_sharpe_ratio:.2f}")
portfolio_data["Tangent"].append(f"{tangent_sharpe_ratio:.2f}")
portfolio_data["Min Correlation"].append(f"{least_corr_sharpe_ratio:.2f}")

portfolio_data["Max Return"].append(f"{max_return_portfolio_return*100:.2f} %")
portfolio_data["Max Return"].append(f"{max_return_portfolio_sd*100:.2f} %")
portfolio_data["Max Return"].append(f"{max_return_portfolio_variance:.2f}")
portfolio_data["Max Return"].append(f"{max_return_sharpe_ratio:.2f}")

# Add separator
portfolio_data["Min Variance"].append("---")
portfolio_data["Tangent"].append("---")
portfolio_data["Min Correlation"].append("---")
portfolio_data["Max Return"].append("---")

# Determine which tickers have any non-zero allocation
non_zero_mask = []

for i in range(len(stock_tickers)):
    if (
        min_var_weights[i] > 1e-6 or
        tangent_weights[i] > 1e-6 or
        least_corr_weights[i] > 1e-6 or
        max_return_weights[i] > 1e-6
    ):
        non_zero_mask.append(True)
    else:
        non_zero_mask.append(False)

# Add weights for each stock (ONLY non-zero ones)
for i, stock in enumerate(stock_tickers):

    if not non_zero_mask[i]:
        continue

    w_min = min_var_weights[i]
    w_tan = tangent_weights[i]
    w_corr = least_corr_weights[i]
    w_max = max_return_weights[i]

    portfolio_data["Min Variance"].append(f"{w_min*100:.2f} %")
    portfolio_data["Tangent"].append(f"{w_tan*100:.2f} %")
    portfolio_data["Min Correlation"].append(f"{w_corr*100:.2f} %")
    portfolio_data["Max Return"].append(f"{w_max*100:.2f} %")

# Create index with performance metrics and stock weights
index_labels = [
    "E[R]",
    "σ",
    "Variance",
    "Sharpe Ratio",
    ""
] + [f"Weight: {stock}" for i, stock in enumerate(stock_tickers) if non_zero_mask[i]]

combined_portfolio_df = pd.DataFrame(portfolio_data, index=index_labels)

# ==========================================================
# FETCH FUNDAMENTAL VALUES (WRDS) 
# ==========================================================

conn = wrds.Connection(wrds_username="tobiasnhh")  # create pgpass file for automated login

gvkeys = conn.raw_sql(f"""
    SELECT DISTINCT gvkey, tic
    FROM comp.funda
    WHERE tic IN {tuple(tickers)}
""")

gvkey_list = tuple(gvkeys.gvkey.unique())

data = conn.raw_sql(f"""
    SELECT gvkey, tic, fyear,
           sale, ebit, ni,
           oancf,
           dltt, dlc, ceq
    FROM comp.funda
    WHERE gvkey IN {gvkey_list}
      AND indfmt = 'INDL'
      AND datafmt = 'STD'
      AND popsrc = 'D'
      AND consol = 'C'
      AND fyear >= {start_year - 1}
    ORDER BY gvkey, fyear
""")

data = data.sort_values(['tic', 'fyear'])

# Revenue Growth (YoY)
data['Revenue_Growth'] = data.groupby('tic')['sale'].pct_change()

# EBIT Margin
data['EBIT_Margin'] = data['ebit'] / data['sale']

# Net Profit Growth & Margin
data['Net_Profit_Growth'] = data.groupby('tic')['ni'].pct_change()
data['Net_Profit_Margin'] = data['ni'] / data['sale']

# Operating Cash Flow Growth & Margin
data['OCF_Growth'] = data.groupby('tic')['oancf'].pct_change()
data['OCF_Margin'] = data['oancf'] / data['sale']

# Debt to Equity
data['Debt_to_Equity'] = (data['dltt'] + data['dlc']) / data['ceq']

def default_rating(x):
    if pd.isna(x):
        return "Risky"
    elif x > 0.247:
        return "Aaa"
    elif x > 0.188:
        return "Aa"
    elif x > 0.142:
        return "A"
    elif x > 0.128:
        return "Baa"
    elif x > 0.118:
        return "Ba"
    elif x > 0.095:
        return "B"
    elif x > 0.045:
        return "C"
    else:
        return "Risky"

data['Default_Risk'] = data['EBIT_Margin'].apply(default_rating)

data = data[data['fyear'] >= start_year]

metrics = [
    'Revenue_Growth',
    'EBIT_Margin',
    'Net_Profit_Growth',
    'Net_Profit_Margin',
    'OCF_Growth',
    'OCF_Margin',
    'Debt_to_Equity',
    'Default_Risk'
]

final = data[['tic', 'fyear'] + metrics].copy()
final = final.sort_values(['tic', 'fyear'])
final = final.set_index(['tic', 'fyear'])

pd.options.display.float_format = '{:.2%}'.format

# ==========================================================
# COMPARABLES TABLE 
# ==========================================================

# Get latest fiscal year per firm
latest_fundamentals = data.loc[data.groupby('tic')['fyear'].idxmax()].copy()

comps_data = []

for stock in stock_tickers:

    firm = latest_fundamentals[latest_fundamentals['tic'] == stock]
    if firm.empty:
        continue

    firm = firm.iloc[0]

    try:
        sector = yf.Ticker(stock).info.get("sector", "N/A")
    except Exception:
        sector = "N/A"

    # Fundamentals (Compustat in millions → USD)
    revenue = firm['sale'] * 1e6
    ebit = firm['ebit'] * 1e6
    net_income = firm['ni'] * 1e6
    total_debt = (firm['dltt'] + firm['dlc']) * 1e6

    price = latest_prices.get(stock, np.nan)
    market_cap = market_caps.get(stock, np.nan)

    if pd.isna(price) or pd.isna(market_cap):
        continue

    enterprise_value = market_cap + total_debt
    shares = market_cap / price if price > 0 else np.nan

    pe = (price / (net_income / shares)) if shares and net_income > 0 else np.nan
    ev_sales = enterprise_value / revenue if revenue > 0 else np.nan
    ev_ebit = enterprise_value / ebit if ebit > 0 else np.nan

    comps_data.append({
        "Company": stock,
        "Sector": sector,
        "P/E": pe,
        "EV/Sales": ev_sales,
        "EV/EBIT": ev_ebit
    })

comps_numeric = pd.DataFrame(comps_data).set_index("Company")

industry_median = comps_numeric.groupby("Sector")[["P/E", "EV/Sales", "EV/EBIT"]].median().round(2)

def add_signal(row, column):
    value = row[column]
    median = industry_median.loc[row["Sector"], column]

    if pd.isna(value) or pd.isna(median):
        return ""
    elif value > median:
        return " (^)"
    elif value < median:
        return " (⌄)"
    else:
        return ""

comps_display = comps_numeric.copy()

for col in ["P/E", "EV/Sales", "EV/EBIT"]:
    comps_display[col] = comps_numeric.apply(
        lambda row: f"{round(row[col],2)}{add_signal(row,col)}"
        if not pd.isna(row[col]) else np.nan,
        axis=1
    )

# ==========================================================
# PRINT TO CONSOLE 
# ==========================================================

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

#-----# PRINT MARKET ANALYSIS #-----#

if MARKET_TABLE:
    print("\n" + "="*console_formating)
    print("MARKET (Annualized)".center(console_formating))
    print("="*console_formating + "\n")

    #print("\n" + "="*console_formating)
    #print("CORRELATION MATRIX - STOCK TO STOCK".center(console_formating))
    #print("="*console_formating)

    def _print_df_with_bars(df):
        try:
            from tabulate import tabulate
            # center-align both strings and numbers
            print(tabulate(df, headers="keys", tablefmt="github", showindex=True, stralign="center", numalign="center"))
            return
        except Exception:
            pass
        try:
            # use markdown table with github format for vertical separators
            print(df.to_markdown(tablefmt="github"))
            return
        except Exception:
            pass
        import re
        s = df.to_string()
        s = re.sub(r' {2,}', ' | ', s)
        print(s)

    _print_df_with_bars(summary_df.round(2).sort_values("Mean Return (%)", ascending=False))

#-----# PRINT COMPS #-----#

if COMPS_TABLE:
    print("\n" + "="*console_formating)
    print("COMPS".center(console_formating))
    print("="*console_formating + "\n")
    _print_df_with_bars(industry_median) 
    print("\n")
    _print_df_with_bars(comps_display.sort_values(["Sector", "Company"])) 

#-----# PRINT PORTFOLIOS #-----#

if PORTFOLIO_TABLE: 
    # Print individual analysis with vertical separators
    print("\n" + "="*console_formating)
    print("PORTFOLIOS".center(console_formating))
    print("="*console_formating + "\n")
    _print_df_with_bars(combined_portfolio_df)
    #print("="*console_formating + "\n")

#-----# PRINT MACRO CORRELATION TABLE #-----#

if MACRO_TABLE:
    print("\n" + "="*console_formating)
    print("MACRO FACTORS CORRELATION".center(console_formating))
    print("="*console_formating + "\n")

    if not correlation_macro.empty:

        # ====== CALCULATE MACRO YTD CHANGE ======
        macro_ytd = {}

        for macro in macro_factors:
            series = df_macro_prices["Adj Close"][macro]
            start_price = _get_price_on_or_after(series, start_of_year)
            ytd_change = (series.iloc[-1] / start_price) - 1
            macro_ytd[macro] = ytd_change * 100

        macro_ytd_formatted = {
            "VIX": f"{macro_ytd['^VIX']:+.0f}%",
            "Gold": f"{macro_ytd['GC=F']:+.0f}%",
            "Interest": f"{macro_ytd['^TNX']:+.0f}%"
        }

        change_row = pd.DataFrame(
            [macro_ytd_formatted],
            index=["CHANGE"]
        )

        # ====== SEPARATOR ROW ======
        # ====== SEPARATOR ROW (only factor columns) ======
        separator_row = pd.DataFrame(
            [["---", "---", "---"]],
            columns=["VIX", "Gold", "Interest"],
            index=[""]   # <- blank index so nothing appears under CHANGE
        )

        # Combine: CHANGE → separator → tickers
        macro_table = pd.concat([
            change_row,
            separator_row,
            correlation_macro.round(2)
        ])

        _print_df_with_bars(macro_table)

    else:
        print("No macro data available to compute correlations.")

#-----# PRINT FUNDAMENTALS #-----#

if FUNDAMENTALS_TABLE:
    print("\n" + "="*console_formating) 
    print("FUNDAMENTALS".center(console_formating)) 
    print("="*console_formating + "\n") 

    for ticker in final.index.get_level_values(0).unique():
        print(f"{ticker}")
        print("--------------------------")
        print(final.loc[ticker])
        print("\n")

# ==========================================================
# TICKER FILTER
# ==========================================================

if TICKERS_FILTER:

    print("\n" + "="*console_formating)
    print("TICKER FILTER".center(console_formating))
    print("="*console_formating + "\n")

#-----# MARKET ANALYSIS FILTER #-----#


    market_df = summary_df.loc[stock_tickers].copy()

    market_df["Rank_Return"] = market_df["Mean Return (%)"].rank(ascending=False)
    market_df["Rank_Sharpe"] = market_df["Sharpe Ratio"].rank(ascending=False)

    market_df["Market_Score"] = (
        0.5 * market_df["Rank_Return"] +
        0.5 * market_df["Rank_Sharpe"]
    )

    cutoff_market = max(1, int(len(market_df) / 3))
    top_market = market_df.sort_values("Market_Score").head(cutoff_market)
    top_market_tickers = top_market.index.tolist()

    print("Top Market Analysis (1/3):")
    print(top_market_tickers)
    print("\n")

#-----# COMPS FILTER #-----#

    comps_filter_df = comps_numeric.loc[
        comps_numeric.index.intersection(top_market_tickers)
    ].copy()

    undervaluation_scores = []

    for ticker in comps_filter_df.index:

        row = comps_filter_df.loc[ticker]
        sector = row["Sector"]

        if sector not in industry_median.index:
            continue

        sector_med = industry_median.loc[sector]
        diffs = []

        for metric in ["P/E", "EV/Sales", "EV/EBIT"]:
            if pd.notna(row[metric]) and pd.notna(sector_med[metric]):
                diffs.append((sector_med[metric] - row[metric]) / sector_med[metric])

        if len(diffs) > 0:
            undervaluation_scores.append({
                "Ticker": ticker,
                "Score": np.mean(diffs)
            })

    undervaluation_df = pd.DataFrame(undervaluation_scores).set_index("Ticker")

    cutoff_comps = max(1, int(len(undervaluation_df) / 3))
    top_comps = undervaluation_df.sort_values("Score", ascending=False).head(cutoff_comps)
    top_comps_tickers = top_comps.index.tolist()

    print("Top COMPS (1/3 of Market):")
    print(top_comps_tickers)
    print("\n")

#-----# FUNDEMENTALS FILTER #-----#

    fundamentals_latest = latest_fundamentals.set_index("tic")

    fundamentals_filter = fundamentals_latest.loc[
        fundamentals_latest.index.intersection(top_comps_tickers)
    ].copy()

    fundamentals_filter["Rank_Revenue"] = fundamentals_filter["Revenue_Growth"].rank(ascending=False)
    fundamentals_filter["Rank_OCF"] = fundamentals_filter["OCF_Growth"].rank(ascending=False)

    fundamentals_filter["Fundamental_Score"] = (
        0.5 * fundamentals_filter["Rank_Revenue"] +
        0.5 * fundamentals_filter["Rank_OCF"]
    )

    top_fundamentals = fundamentals_filter.sort_values("Fundamental_Score").head(5)
    top_fundamental_tickers = top_fundamentals.index.tolist()

    print("Top Fundamentals (5 of Market and COMPS):")
    print(top_fundamental_tickers)
    print("\n")