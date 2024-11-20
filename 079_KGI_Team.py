import pandas as pd
import numpy as np
from IPython.display import display

#C:/Users/LandMark/Downloads
#'~/Desktop/Daily_Ticks.csv'
file_path =  'C:/Users/LandMark/Downloads/Daily_Ticks.csv'
# กำหนดค่าการแสดงผลให้แสดงข้อมูลทั้งหมดของ DataFrame
pd.set_option('display.max_columns', None)  # แสดงทุกคอลัมน์
pd.set_option('display.max_rows', None)    
data = pd.read_csv(file_path) 

data.columns =['Name', 'Date', 'LastPrice', 'Volume','Value','Flag']
symbol = data['Name'].unique() # list all symbol set 50
symbol_list = ['TTB', 'BTS', 'HMPRO', 'TRUE', 'MTC', 'BH', 'TOP', 'TIDLOR', 'OSP', 'SCC']
data_filter = data[data['Name'].isin(symbol_list)].copy()

# ฟังก์ชันคำนวณ Bollinger Bands
def bollinger_bands(data, window=20, no_of_std=2):
    data.loc[:,'SMA'] = data['LastPrice'].rolling(window=window).mean() # ค่าเฉลี่ยเคลื่อนที่ 20 วัน
    data.loc[:,'STD'] = data['LastPrice'].rolling(window=window).std() # ส่วนเบี่ยงเบนมาตรฐาน 20 วัน
    data.loc[:,'Upper Band'] = data['SMA'] + (data['STD'] * no_of_std) # ขอบบนของ Bollinger Band
    data.loc[:,'Lower Band'] = data['SMA'] - (data['STD'] * no_of_std) # ขอบล่างของ Bollinger Band
    return data
# ฟังก์ชันคำนวณ Stochastic Oscillator
def stochastic_oscillator(data, period=14):
    data.loc[:,'Low14'] = data['Volume'].rolling(period).min() # จุดต่ำสุดในช่วง 14 วัน
    data.loc[:,'High14'] = data['Volume'].rolling(period).max() # จุดสูงสุดในช่วง 14 วัน
    data.loc[:,'%K'] = 100 * (data['Volume'] - data['Low14']) / (data['High14'] - data['Low14']) # สูตร %K
    data.loc[:,'%D'] = data['%K'].rolling(3).mean() # %D คือค่าเฉลี่ยเคลื่อนที่ 3 วันของ %K
    data.fillna(0, inplace=True)
    return data
# ฟังก์ชันสร้างสัญญาณซื้อ/ขาย
def generate_signals(data):
    data.loc[:,'Buy Signal'] = np.where((data['LastPrice'] < data['Lower Band']) & (data['%K'] < 20), 1, 0)
    data.loc[:,'Sell Signal'] = np.where((data['LastPrice'] > data['Upper Band']) & (data['%K'] > 80), -1, 0)
    return data

data_filter = bollinger_bands(data_filter)
data_filter = stochastic_oscillator(data_filter)
data_filter = generate_signals(data_filter)

# ฟังก์ชันคำนวณผลตอบแทนจากสัญญาณซื้อขาย 
def calculate_returns(data, investment): 
    # ตรวจสอบว่า DataFrame ว่างเปล่าหรือไม่
    if data.empty:
        return 0, investment, data  # คืนค่าศูนย์หากไม่มีข้อมูล

    data['Position'] = np.where(data['Buy Signal'] == 1, 1, np.where(data['Sell Signal'] == -1, -1, 0))
    data['Returns'] = data['LastPrice'].pct_change()  # ผลตอบแทนรายวัน 
    data['Strategy Returns'] = data['Position'].shift(1) * data['Returns']  # ผลตอบแทนตามกลยุทธ์จากสัญญาณซื้อขาย 
    data['Cumulative Strategy Returns'] = (1 + data['Strategy Returns']).cumprod() 

    # ตรวจสอบว่า Cumulative Strategy Returns มีข้อมูลหรือไม่
    if data['Cumulative Strategy Returns'].empty:
        return 0, investment, data  # คืนค่าศูนย์หากไม่มี Cumulative Strategy Returns

    final_value = investment * data['Cumulative Strategy Returns'].iloc[-1]
    profit_loss = final_value - investment 
    data.fillna(0, inplace=True)
    return profit_loss, final_value, data 

initial_investment = 10000000
# สุ่มค่าในช่วง 0 ถึง 1
random_weights = np.random.rand(len(symbol_list))
# ปรับค่าดังกล่าวให้มีผลรวมเป็น 1
normalized_weights = random_weights / np.sum(random_weights)
# สร้างพอร์ตโฟลิโอ
portfolio = {stock: weight for stock, weight in zip(symbol_list, normalized_weights)}

TTB_profit_loss, TTB_final_value, TTB_data = calculate_returns(data_filter[data_filter['Name'] == 'TTB'], initial_investment * portfolio['TTB'])
# WHA_profit_loss, WHA_final_value, WHA_data = calculate_returns(data_filter[data_filter['Name'] == 'WHA'], initial_investment * portfolio['WHA'])
BTS_profit_loss, BTS_final_value, BTS_data = calculate_returns(data_filter[data_filter['Name'] == 'BTS'], initial_investment * portfolio['BTS'])
HMPRO_profit_loss, HMPRO_final_value, HMPRO_data = calculate_returns(data_filter[data_filter['Name'] == 'HMPRO'], initial_investment * portfolio['HMPRO'])
TRUE_profit_loss, TRUE_final_value, TRUE_data = calculate_returns(data_filter[data_filter['Name'] == 'TRUE'], initial_investment * portfolio['TRUE'])
MTC_profit_loss, MTC_final_value, MTC_data = calculate_returns(data_filter[data_filter['Name'] == 'MTC'], initial_investment * portfolio['MTC'])
BH_profit_loss, BH_final_value, BH_data = calculate_returns(data_filter[data_filter['Name'] == 'BH'], initial_investment * portfolio['BH'])
TOP_profit_loss, TOP_final_value, TOP_data = calculate_returns(data_filter[data_filter['Name'] == 'TOP'], initial_investment * portfolio['TOP'])
TIDLOR_profit_loss, TIDLOR_final_value, TIDLOR_data = calculate_returns(data_filter[data_filter['Name'] == 'TIDLOR'], initial_investment * portfolio['TIDLOR'])
OSP_profit_loss, OSP_final_value, OSP_data = calculate_returns(data_filter[data_filter['Name'] == 'OSP'], initial_investment * portfolio['OSP'])
SCC_profit_loss, SCC_final_value, SCC_data = calculate_returns(data_filter[data_filter['Name'] == 'SCC'], initial_investment * portfolio['SCC'])

def calculate_performance_metrics(data):       
    net_profit_pct = (data['Cumulative Strategy Returns'] - 1) * 100
    # Calculate cumulative maximum for drawdown calculations
    data['Cumulative Max'] = data['Cumulative Strategy Returns'].cummax()
    data['Drawdown'] = data['Cumulative Max'] - data['Cumulative Strategy Returns']
    data['Drawdown %'] = data['Drawdown'] / data['Cumulative Max']
    max_drawdown = data['Drawdown %'].max() * 100
    # Change in this line: Use any() to check if any value in the Series is not 0
    relative_drawdown = (data['Drawdown'].max() / data['Cumulative Max'].max()) * 100 if (data['Cumulative Max'] != 0).any() else 0  
    wins = data[data['Strategy Returns'] > 0].shape[0]
    total_trades = data[data['Position'] != 0].shape[0]
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    # Calculate Calmar Ratio
    calmar_ratio = net_profit_pct / abs(max_drawdown) if max_drawdown != 0 else np.nan
    # Sharpe Ratio is calculated separately if needed
    sharpe_ratio = data['Strategy Returns'].mean() / data['Strategy Returns'].std() * np.sqrt(252)
    return net_profit_pct, win_rate, max_drawdown, relative_drawdown, calmar_ratio, sharpe_ratio

TTB_metrics = calculate_performance_metrics(TTB_data) 
# WHA_metrics = calculate_performance_metrics(WHA_data)
BTS_metrics = calculate_performance_metrics(BTS_data)
HMPRO_metrics = calculate_performance_metrics(HMPRO_data)
TRUE_metrics = calculate_performance_metrics(TRUE_data)
MTC_metrics = calculate_performance_metrics(MTC_data)
BH_metrics = calculate_performance_metrics(BH_data)
TOP_metrics = calculate_performance_metrics(TOP_data)
TIDLOR_metrics = calculate_performance_metrics(TIDLOR_data)
OSP_metrics = calculate_performance_metrics(OSP_data)
SCC_metrics = calculate_performance_metrics(SCC_data)

TTB_profit_pct = (TTB_profit_loss / (initial_investment * portfolio['TTB'])) * 100
# WHA_profit_pct = (WHA_profit_loss / (initial_investment * portfolio['WHA'])) * 100
BTS_profit_pct = (BTS_profit_loss / (initial_investment * portfolio['BTS'])) * 100
HMPRO_profit_pct = (HMPRO_profit_loss / (initial_investment * portfolio['HMPRO'])) * 100
TRUE_profit_pct = (TRUE_profit_loss / (initial_investment * portfolio['TRUE'])) * 100
MTC_profit_pct = (MTC_profit_loss / (initial_investment * portfolio['MTC'])) * 100
BH_profit_pct = (BH_profit_loss / (initial_investment * portfolio['BH'])) * 100
TOP_profit_pct = (TOP_profit_loss / (initial_investment * portfolio['TOP'])) * 100
TIDLOR_profit_pct = (TIDLOR_profit_loss / (initial_investment * portfolio['TIDLOR'])) * 100
OSP_profit_pct = (OSP_profit_loss / (initial_investment * portfolio['OSP'])) * 100
SCC_profit_pct = (SCC_profit_loss / (initial_investment * portfolio['SCC'])) * 100

# คำนวณกำไรรวมของพอร์ต
total_final_value = TTB_final_value + BTS_final_value + HMPRO_final_value + TRUE_final_value + MTC_final_value + BH_final_value + TOP_final_value + TIDLOR_final_value + OSP_final_value + SCC_final_value
total_investment = initial_investment * sum(portfolio.values())
total_profit_loss = total_final_value - total_investment

# คำนวณตัวชี้วัดสำหรับพอร์ต
def calculate_portfolio_metrics(metrics):
    net_profit_pct = (metrics[0][0] + metrics[1][0] + metrics[2][0]) / 3
    win_rate = (metrics[0][1] + metrics[1][1] + metrics[2][1]) / 3
    max_drawdown = (metrics[0][2] + metrics[1][2] + metrics[2][2]) / 3
    sharpe_ratio = (metrics[0][3] + metrics[1][3] + metrics[2][3]) / 3
    return net_profit_pct, win_rate, max_drawdown, sharpe_ratio

# คำนวณตัวชี้วัดของพอร์ต
portfolio_metrics = calculate_portfolio_metrics([TTB_metrics , BTS_metrics , HMPRO_metrics , TRUE_metrics , MTC_metrics , BH_metrics , TOP_metrics , TIDLOR_metrics , OSP_metrics , SCC_metrics])

# สร้าง DataFrame สำหรับแสดงผลลัพธ์ในรูปตาราง

portfolio_data = {
'Stock': ['TTB', 'BTS', 'HMPRO', 'TRUE', 'MTC', 'BH', 'TOP', 'TIDLOR', 'OSP', 'SCC','Portfolio'],
'Initial Investment': [initial_investment * portfolio['TTB'],
                        initial_investment * portfolio['BTS'],
                        initial_investment * portfolio['HMPRO'],
                        initial_investment * portfolio['TRUE'],
                        initial_investment * portfolio['MTC'],
                        initial_investment * portfolio['BH'],
                        initial_investment * portfolio['TOP'],
                        initial_investment * portfolio['TIDLOR'],
                        initial_investment * portfolio['OSP'],
                        initial_investment * portfolio['SCC'],
                        total_investment],
'Final Value': [ TTB_final_value ,
                BTS_final_value ,
                HMPRO_final_value ,
                TRUE_final_value ,
                MTC_final_value ,
                BH_final_value ,
                TOP_final_value ,
                TIDLOR_final_value ,
                OSP_final_value ,
                SCC_final_value ,
                total_final_value],
'Profit/Loss': [ TTB_profit_loss ,
                BTS_profit_loss ,
                HMPRO_profit_loss ,
                TRUE_profit_loss ,
                MTC_profit_loss ,
                BH_profit_loss ,
                TOP_profit_loss ,
                TIDLOR_profit_loss ,
                OSP_profit_loss ,
                SCC_profit_loss ,
                total_profit_loss],
'Profit %': [    TTB_profit_pct ,
                BTS_profit_pct ,
                HMPRO_profit_pct ,
                TRUE_profit_pct ,
                MTC_profit_pct ,
                BH_profit_pct ,
                TOP_profit_pct ,
                TIDLOR_profit_pct ,
                OSP_profit_pct ,
                SCC_profit_pct ,
                portfolio_metrics[0]],
'Win Rate %': [  TTB_metrics[1] ,
                BTS_metrics[1] ,
                HMPRO_metrics[1] ,
                TRUE_metrics[1] ,
                MTC_metrics[1] ,
                BH_metrics[1] ,
                TOP_metrics[1] ,
                TIDLOR_metrics[1] ,
                OSP_metrics[1] ,
                SCC_metrics[1] ,
                portfolio_metrics[1]],
'Max Drawdown %': [TTB_metrics[2] ,
                BTS_metrics[2] ,
                HMPRO_metrics[2] ,
                TRUE_metrics[2] ,
                MTC_metrics[2] ,
                BH_metrics[2] ,
                TOP_metrics[2] ,
                TIDLOR_metrics[2] ,
                OSP_metrics[2] ,
                SCC_metrics[2] ,
                portfolio_metrics[2]],
'Sharpe Ratio': [TTB_metrics[3] ,
                BTS_metrics[3] ,
                HMPRO_metrics[3] ,
                TRUE_metrics[3] ,
                MTC_metrics[3] ,
                BH_metrics[3] ,
                TOP_metrics[3] ,
                TIDLOR_metrics[3] ,
                OSP_metrics[3] ,
                SCC_metrics[3] ,
                portfolio_metrics[3]]
}


# สร้างตารางผลลัพธ์ด้วย pandas DataFrame

portfolio_df = pd.DataFrame(portfolio_data)



# ตั้งค่าแสดงผลเป็นทศนิยม 2 ตำแหน่ง

pd.options.display.float_format = '{:.2f}'.format



# แสดงผลลัพธ์ในรูปแบบตาราง

# print(portfolio_df)


# คำนวณกำไรรวมของพอร์ต
total_final_value = TTB_final_value + BTS_final_value + HMPRO_final_value + TRUE_final_value + MTC_final_value + BH_final_value + TOP_final_value + TIDLOR_final_value + OSP_final_value + SCC_final_value
total_investment = initial_investment * sum(portfolio.values())
total_profit_loss = total_final_value - total_investment

# Get the last value of total_final_value Series for printing
total_final_value_to_print = total_final_value  # Access the last element

# แสดงผลกำไรรวมของพอร์ตโฟลิโอ
print(f"\nTotal Investment: {total_investment:.2f}")
print(f"Total Final Value: {total_final_value_to_print:.2f}")  # Change here
print(f"Total Profit/Loss: {total_profit_loss}")  # Chan

# # Convert values to DataFrames for export
dfPortfolio = pd.DataFrame(portfolio_df)

# # Save to Excel
dfPortfolio.to_excel('Portfolio.xlsx', index=False)