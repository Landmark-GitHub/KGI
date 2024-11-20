import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

#'C:/Users/LandMark/Desktop/KGI/competition_api'
#'~/Desktop/competition_api'
output_dir = '~/Desktop/competition_api'

# ฟังก์ชันสำหรับโหลดข้อมูลเก่า
def load_previous(file_type, teamName):
    global prev_portfolio_df
    folder_path = output_dir + f"/Previous/{file_type}"
    file_path = folder_path + f"/{teamName}_{file_type}.csv"

    if os.path.exists(file_path):
        prev_portfolio_df = pd.read_csv(file_path)
        print(f"Loaded '{file_type}' Previous")
    else:
        print(f"Previous '{file_type}' file not found")

# ฟังก์ชันสำหรับบันทึกผลลัพธ์
def save_output(data, file_type, teamName):
    folder_path = output_dir + f"/Result/{file_type}"
    file_path = folder_path + f"/{teamName}_{file_type}.csv"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"Directory created: '{folder_path}'")

    # Save CSV
    data.to_csv(file_path, index=False)
    print(f"{file_type} saved at {file_path}")

# C:/Users/LandMark/Desktop/Daily_Ticks_081124.csv
#'~/Desktop/Daily_Ticks.csv'
file_path =  '~/Desktop/Daily_Ticks.csv'  
data = pd.read_csv(file_path)
team_name = "079_Loki"

# Set display options
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(df, window=20, num_std_dev=2):
    df['Rolling Mean'] = df['LastPrice'].rolling(window=window).mean() # ค่าเฉลี่ยเคลื่อนที่ 20 วัน
    df['Rolling STD'] = df['LastPrice'].rolling(window=window).std() # ส่วนเบี่ยงเบนมาตรฐาน 20 วัน
    df['Bollinger Upper'] = df['Rolling Mean'] + num_std_dev * df['Rolling STD'].rolling(window=window).std()
    df['Bollinger Lower'] = df['Rolling Mean'] - num_std_dev * df['Rolling STD'].rolling(window=window).std()
    return df


# Function to calculate Stochastic Oscillator
def calculate_stochastic_oscillator(df, k_window=14, d_window=3):
    df['Low Min'] = df['LastPrice'].rolling(window=k_window).min()
    df['High Max'] = df['LastPrice'].rolling(window=k_window).max()
    df['%K'] = (df['LastPrice'] - df['Low Min']) / (df['High Max'] - df['Low Min']) * 100
    df['%D'] = df['%K'].rolling(window=d_window).mean()
    return df


# Function to generate buy/sell signals based on Bollinger Bands and Stochastic Oscillator
def generate_signals(df):
    df['Buy Signal'] = (df['LastPrice'] < df['Bollinger Lower']) & (df['%K'] < 20)
    df['Sell Signal'] = (df['LastPrice'] > df['Bollinger Upper']) & (df['%K'] > 80)
    return df

# Apply technical indicators and signals to each stock
data = calculate_bollinger_bands(data)
data = calculate_stochastic_oscillator(data)
data = generate_signals(data)

# Portfolio and trade statement DataFrame preparation
portfolio_df = pd.DataFrame({
    'Table Name': 'Portfolio (Table)',
    'File Name': 'Daily_Ticks.csv',
    'Stock Name': data['ShareCode'],
    'Start Vol': data['Volume'],
    'Actual Vol': data['Volume'],
    'Avg Cost': data['LastPrice'] * 0.95,  # Placeholder for Avg Cost calculation
    'Market price': data['LastPrice'],
    'Amount Cost': (data['LastPrice'] * 0.95) * data['Volume'],  # Placeholder, calculate if Avg Cost available
    'Market Value': data['LastPrice'] * data['Volume'],
    'Unrealized P/L': (data['LastPrice'] * data['Volume']) - ((data['LastPrice'] * 0.95) * data['Volume']),  # Placeholder, calculate based on cost and current price
    '%Unrealized P/L': ((data['LastPrice'] * data['Volume']) - ((data['LastPrice'] * 0.95) * data['Volume']) / ((data['LastPrice'] * 0.95) * data['Volume']) * 100),  # Placeholder for % Unrealized P/L
    'Realized P/L': np.nan  # Placeholder for Realized P/L calculation
})

statement_df = pd.DataFrame({
    'Table Name': 'Statement',
    'File Name': 'Daily_Ticks.csv',
    'Stock Name': data['ShareCode'],
    'Date': data['TradeDateTime'],  # Placeholder, fill with actual time if available
    'Side': np.where(data['Buy Signal'], 'Buy', 'Sell'),
    'Volume': data['Volume'],
    'Price': data['LastPrice'],
    'Amount cost': data['LastPrice'] * data['Volume'],
    'End_line_available': np.nan  # Placeholder for available balance
})

summary_df = pd.DataFrame({
    'Table Name': ['Summary'],
    'File Name': ['Daily_Ticks.csv'],
    'Portfolio value': [portfolio_df['Market Value'].sum()],
    'End Line available': [portfolio_df['Market Value'].sum()],  # Placeholder for end line available
    'Start Line available': [portfolio_df['Amount Cost'].sum()],  # Placeholder for start line available
    'Number of wins': [(portfolio_df['Unrealized P/L'] > 0).sum()],  # Placeholder for win count
    'Number of matched trades': [statement_df['Side'].value_counts().min()],  # Placeholder for matched trades
    'Number of transactions': [len(statement_df)],
    'Sum of Unrealized P/L': [portfolio_df['Unrealized P/L'].sum()],
    'Sum of %Unrealized P/L': [portfolio_df['%Unrealized P/L'].mean()],
    'Sum of Realized P/L': [portfolio_df['Realized P/L'].sum()],
    'Maximum value': [portfolio_df['Market Value'].max()],
    'Minimum value': [portfolio_df['Market Value'].min()],
    'Win rate': ((portfolio_df['Unrealized P/L'] > 0).sum() / len(portfolio_df)) * 100,  # Placeholder for win rate calculation
    'Calmar Ratio': [np.nan],  # Placeholder for Calmar Ratio
    'Relative Drawdown': [np.nan],  # Placeholder for Relative Drawdown
    'Maximum Drawdown': [np.nan],  # Placeholder for Maximum Drawdown
    '%Return': ((portfolio_df['Market Value'].sum() - portfolio_df['Amount Cost'].sum()) / portfolio_df['Amount Cost'].sum()) * 100  # Placeholder for %Return
})

# # Save each table to Excel for reporting
# portfolio_df.to_excel('Portfolio.xlsx', index=False)
# print('Export Portfolio.xlsx to Success')
# statement_df.to_excel('Statement.xlsx', index=False)
# print('Export Statemen.xlsx to Success')
# summary_df.to_excel('Summary.xlsx', index=False)
# print('Export Summary.xlsx to Success')


# Save each table to files
save_output(portfolio_df, "Portfolio", team_name)
save_output(statement_df, "Statement", team_name)
save_output(summary_df, "Summary", team_name)


