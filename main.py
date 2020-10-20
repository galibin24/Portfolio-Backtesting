# three states 
import yfinance as yf
import pandas as pd 
import numpy as np
import functools 

# assumptions:
    # When stop-loss exceeds 10% we sell at the last price on the same day(hence cash loss exceeds 1000)
    # When profit exceeds 100% we sell 1/2 at the last price on the same day(hence the cash gain can exceed 10000)



# 1 = 100%, 0.5 = 50%
gain_percent_required_before_cashing = 1
# 0.5 = 1/2
# proportion added back to stop loss
proportion_reinvested = 0.1
# other proportions
proportion_cashedout = 0.6  
proportion_kept_after_cashing = 0.4 
# -0.1 = -10%
stop_loss_percent = -0.05


cash = 0
# when buy issued stock deleted from all and added here
bought_stocks_full = []
bought_stocks_mem = []
# when stock is 100% above initial we sell 1/2 and keep 1/2
bought_stocks_third = []
bougth_prop = []
# when stock drops 10% below initial 
stop_loss_sold = []
stop_loss_sold_dates = []
stop_loss_sold_selldate = []
# recomendations
stocks = pd.read_excel('Bert_stocks.xlsx', )
# list of all tickers
Tickers = list(stocks['Stock'])
# stocks bought history 
stock_bought = {}
for stock in Tickers:
    stock_bought[stock] = 0
# price for every ticker
all_prices = pd.read_excel('Bert_Prices.xlsx', index_col = 0)

# historicaly sorted recomendations
stocks['Buy_Day'] = pd.to_datetime(stocks.Buy_Day)
his_stocks = stocks.sort_values(by='Buy_Day')
print(his_stocks)
his_stocks = his_stocks[his_stocks.iloc[ : , 1] >= '2018-07-01']
print(his_stocks)
his_stocks.index = his_stocks['Stock']

his_stocks.drop(['Stock'], axis=1, inplace = True )

# gets the nearest date 
def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


# if price for nearest date is nan pick the first price to find 
def check_for_nan(price, date, stock, buy_prices):
    df = buy_prices
    print(price)
    if str(price) == 'nan':
        print('price is nan')
        print(date)
        print(stock)
        
        new_price = df.loc[df.loc[date:, stock ].first_valid_index(), stock]
        print('new price is:' + str(new_price))
    else:
        return price



def get_date_of_buy(stock, recomendations, buy_prices, stock_Bougth):
    recomend_buy = recomendations
    
    print()
    rec_before = stock_Bougth[stock]
    # all recomendations for particular stock 
    all_buy_dates = recomend_buy.loc[stock].values
    try:
        if isinstance(all_buy_dates[0], np.ndarray):
            date = all_buy_dates[rec_before][0]
            stock_intial_price = buy_prices.loc[date, stock]
        else:
            date = all_buy_dates[0]
            stock_intial_price = buy_prices.loc[date, stock]

    # occurs when price for the date doesn't exist 
    # solved by finding nearest date to buy date
    except KeyError:
        if isinstance(all_buy_dates[0], np.ndarray):
            date = nearest(buy_prices.index.values, all_buy_dates[rec_before][0])
            stock_intial_price = buy_prices.loc[date, stock]
        else:
            date = nearest(buy_prices.index.values, all_buy_dates[0])
            stock_intial_price = buy_prices.loc[date, stock]
    
    return date, stock_intial_price

# calculates return from the day of buy
def Return(stock, date, initial_price, prices, slp = stop_loss_percent, 
    gprbf = gain_percent_required_before_cashing, cashed =proportion_cashedout, re_stop = proportion_reinvested):
    global cash
    df = prices[[stock]]
    prices_onw = df.loc[date:]

    returns_until = []
    for price, date in zip(prices_onw.values, prices_onw.index.values):
        if str(initial_price) != 'nan':
            returns = (float(price) - float(initial_price)) / float(initial_price)
            # if price is nan 
            if str(price[0]) == 'nan':
                # date +1 day from last date
                # last_date = list(returns_until[-1].keys())[0] + np.timedelta64(1,'D')
                last_return = list(returns_until[-1].values())[0] 
                returns_until.append({date: last_return})
            else:
                date2 = date
                returns_until.append({date2: returns})
                if returns > gprbf:
                    date1 = date
                    date_of_end = date1 + np.timedelta64(1,'D')
                    state = "gain reached"
                    # print()
                    last_ret = list(returns_until[-1].values())[0] 
                    # total cash 
                    cash = 10000 * last_ret + 10000
                    # cash used for reinvestment 
                    cash2 = cash
                    # cash used to add back to Stock Loss(10%)
                    cash_for_stop = cash * re_stop
                    # Cash added to main cash(60%)
                    cash = cash * cashed
                    return price[0], date1, stock, state, returns_until, cash, date_of_end, cash_for_stop, cash2
                if returns < slp:
                    date1 = date
                    state = 'stop-loss reached'
                    last_ret = list(returns_until[-1].values())[0]
                    # gain
                    cash = 10000 * last_ret
                    return price[0], date1, stock, state, returns_until, cash
        else:
            return 'initial price is nan'

# return if stock is still hold
def Return_holding(stock, date, initial_price, prices):
    df = prices[[stock]]
    prices_onw = df.loc[date:]
    returns_until = []
    for price, date in zip(prices_onw.values, prices_onw.index.values):
        if str(initial_price) != 'nan':
            returns = (float(price) - float(initial_price)) / float(initial_price)
            # check here
            if str(price[0]) == 'nan':
                # last_date = list(returns_until[-1].keys())[0] + np.timedelta64(1,'D')
                last_return = list(returns_until[-1].values())[0] 
                returns_until.append({date: last_return})
            else:
       
                date2 = date
                returns_until.append({date2: returns})
                state = 'holding'

    return price[0], date2, stock, state, returns_until

# makes Dataframe from returns
def make_df(all_prices, returns):
    returns = returns

    # take index from all prices and inject the returns leaving all other values nan 
    # construct dataframe and concat them 
    indexxx = [list(item.keys())[0] for item in returns[4]]
    data = [item.values() for item in returns[4]]
    df2 = pd.DataFrame.from_records(data = data, index = indexxx)
    df2.rename(columns={df2.columns.values[0]: returns[2]}, inplace= True)
    df2.index.name = 'Date'
    df2['Cash Changes ' + returns[2]] = 10000

    df2['Cash Changes ' + returns[2]] = df2['Cash Changes ' + returns[2]] * df2[returns[2]] + 10000 
    df2 = df2[['Cash Changes ' + returns[2]]]

    df2 = df2.loc[~df2.index.duplicated(keep='first')]

    return df2

# proportion of stock is sold and rest is kept 
def make_df_proportional(all_prices, returns, t_cash, kept = proportion_kept_after_cashing):
    returns = returns
    if returns[2] == 'ANET':
        print(t_cash)
    t_cash = t_cash * kept
    indexxx = [list(item.keys())[0] for item in returns[4]]
    data = [item.values() for item in returns[4]]
    df2 = pd.DataFrame.from_records(data = data, index = indexxx)
    df2.rename(columns={df2.columns.values[0]: returns[2]}, inplace= True)
    df2.index.name = 'Date'
    # print('')
    if returns[2] == 'ANET':
        print(t_cash)
    df2['Cash Changes ' + returns[2]] = t_cash
    df2['Cash Changes ' + returns[2]] = df2['Cash Changes ' + returns[2]] * df2[returns[2]] + t_cash 
    df2 = df2[['Cash Changes ' + returns[2]]]
    df2 = df2.loc[~df2.index.duplicated(keep='first')]
    return df2

# makes a secured portfolio
def sec_df(sec_list):
    if len(sec_list) == 0:
        return 
    df_final = pd.concat(sec_list, sort= False, join='outer', axis = 1)
    df_final['Portfolio value'] = df_final.sum(axis=1)

    df_sec = df_final[['Portfolio value']]
    return df_sec

# calculate cash flows
def cash_df(cash_list):
    indexxx = [list(item.keys())[0] for item in cash_list]
    
    data = [item.values() for item in cash_list]
    df2 = pd.DataFrame.from_records(data = data, index = indexxx)
    df2.rename(columns={df2.columns.values[0]: 'Incrementral Cash Change'}, inplace= True)
    df2.index.name = 'Date'
    df2 = df2.sort_values(by='Date')

    cash3 = 0
    cash_3_l = []
    for i in df2['Incrementral Cash Change'].values:
        cash3 = cash3 + i
        cash_3_l.append(cash3)
    
    df2['Cash'] = cash_3_l

    # for all cash flows
    # fill mising dates with values 
    df2.index = pd.DatetimeIndex(df2.index)
    df2 = df2.loc[~df2.index.duplicated(keep='last')]
    all_days = pd.date_range(df2.index.min(), df2.index.max(), freq='D')
    df2 = df2.loc[all_days]
    df2.fillna(method='ffill', inplace = True)
    df2 = df2[['Cash']]

    return df2

def loss_cash_df(cash_list, loss_re):

    new_dict = dict((key,d[key]) for d in cash_list for key in d)

    c_list = []
    
    # for stock in cash_list
    for key, value in new_dict.items():
        if int(value) == int(-10000):
            
            dic_c = {key: (value + 10000)}
            c_list.append(dic_c)
        if 0 < value < 10000:
            
            dic_c = {key: (value - 10000)}
            c_list.append(dic_c)
    

    c_list = c_list + loss_re
    indexxx = [list(item.keys())[0] for item in c_list]
    
    data = [item.values() for item in c_list]
    df2 = pd.DataFrame.from_records(data = data, index = indexxx)
    df2.rename(columns={df2.columns.values[0]: 'Incrementral Cash Change'}, inplace= True)
    df2.index.name = 'Date'
    df2 = df2.sort_values(by='Date')
    cash3 = 0
    cash_3_l = []
    for i in df2['Incrementral Cash Change'].values:
        cash3 = cash3 + i
        cash_3_l.append(cash3)
    
    df2['Cash'] = cash_3_l

    # for all cash flows
    # fill mising dates with values 
    df2.index = pd.DatetimeIndex(df2.index)
    df2 = df2.loc[~df2.index.duplicated(keep='last')]
    all_days = pd.date_range(df2.index.min(), df2.index.max(), freq='D')
    df2 = df2.loc[all_days]
    df2.fillna(method='ffill', inplace = True)
    df2 = df2[['Cash']]
    # df_final['Cash'] = df_Cash['Cash']

    return df2


def date_conv(date):
    ts = pd.to_datetime(str(date)) 
    d = ts.strftime('%Y.%m.%d')
    return d

# Memorandum for stop loss 
def mem_stop_loss(sold_stocks, date_of_ac, date_of_sell):

    sold_stocks = sold_stocks
    date_of_ac = [i[1] for i in date_of_ac]
    date_of_ac = [date_conv(i) for i in date_of_ac]
    # print(len(date_of_ac), len(date_of_sell), len(sold_stocks))
    date_of_sell = [date_conv(i[1]) for i in date_of_sell]
    date_of_sell = [ 'Stop-loss_reached_on ' + str(i) for i in date_of_sell]
    all_l = [sold_stocks, date_of_ac, date_of_sell]
    df = pd.DataFrame(index = sold_stocks, columns = ['Date of Acquisition'], data = date_of_ac )
    # df.rename(index={df.index.name: 'Stocks'}, inplace= True)
    df.index.name = 'Name of stock'
    df['Final status'] = date_of_sell
    # print(df)
    
    return df 

# Memorandum for holding
def mem_holding(date_of_ac):
    
    stocks = [i[2] for i in date_of_ac]
    returns = [i[1][0] for i in date_of_ac]
    date_of_ac =  [i[0] for i in date_of_ac]
    date_of_ac = [date_conv(i) for i in date_of_ac]
    returns = [i*100 for i in returns ]
    returns = ["%.2f" % round(i,2) for i in returns]
    returns = ["presently holding with " + str(i) + "%" for i in returns]

    df = pd.DataFrame(index = stocks, columns = ['Date of Acquisition'], data = date_of_ac )
    df.rename(index={df.index.name: 'Stocks'}, inplace= True)
    df.index.name = 'Name of stock'
    df['Final status'] = returns

    return df

# Memorandum for gains
def mem_portion(all_l):
    print(all_l)
    stocks = [i[0] for i in all_l]
    returns = [i[2][0] for i in all_l]
    date_of_ac = [i[1] for i in all_l]
    date_of_ac = [date_conv(i) for i in date_of_ac]
    print(returns)

    returns = [i*100 for i in returns ]
    returns = ["%.2f" % round(i,2) for i in returns]
    returns = [r"(1/2) sold after 100% increase and holding return = " + str(i) + "%" for i in returns]

    df = pd.DataFrame(index = stocks, columns = ['Date of Acquisition'], data = date_of_ac )
    df.rename(index={df.index.name: 'Stocks'}, inplace= True)
    df.index.name = 'Name of stock'
    df['Final status'] = returns
    # print(df)
    return df



# dealing with abnormalities shit
dic_bouth_sold = {}

def main(Tickers, prices, stock_bought):
    global cash, stop_loss_sold_selldate, stop_loss_sold_dates, bought_stocks_mem, bougth_prop
    recomend_buy = Tickers
    buy_prices = prices
    stocks_bought = stock_bought
    # adding stock to portfolio 
    prices_check = []
    df_list = []
    df_cash_list = []
    df_cash_reinv_stop = []
    df_secured = []
    for stock in recomend_buy.index:
        # add stock to portfolio for first time
        if stock not in bought_stocks_full and stock not in bought_stocks_third: 
            
            date_buy = get_date_of_buy(stock, recomend_buy, buy_prices, stocks_bought) 
            date = date_buy[0]
            try:
                sold_date = dic_bouth_sold[stock]
                if sold_date > date:
                    continue
                else:
                    pass
            except:
                pass
            # in case I need to show only recent results 
            # if stock in stop_loss_sold:
                # stop_loss_sold.remove(stock)
                # stop_loss_sold_dates = list(filter(lambda sub_list: stock not in sub_list, stop_loss_sold_dates))
                # print(stop_loss_sold_dates)
                # stop_loss_sold_selldate =  list(filter(lambda sub_list: stock not in sub_list, stop_loss_sold_selldate))
            bought_stocks_full.append(stock)
            
            # print(dic_bouth_sold)

            # function returns date and initial price            
            stock_intial_price = date_buy[1]
            # borrow cash to buy 
            Inc_cash_1  = -10000
            # add that recomendations have been used
            stocks_bought[stock] = stocks_bought[stock] + 1
            # calculate returns
            return1 = Return(stock = stock, date = date, initial_price= stock_intial_price, prices = buy_prices)

            df_cash_list.append({date: Inc_cash_1}) 
            # stock which are holded until today
            if str(return1) == 'None':
                return_hold = Return_holding(stock = stock,date = date,initial_price= stock_intial_price, prices = buy_prices)
                hold_df = make_df(buy_prices, return_hold)
                bought_stocks_mem.append([date, list(return_hold[4][-1].values()), stock])

                df_list.append(hold_df)
                print('holding')
                continue
            # stocks with no price
            if return1 == 'initial price is nan':
                print('no prices' + str(date))
                continue
            else:
                # stop-loss then we buy again
                if return1[3] == 'stop-loss reached':
                
                    bought_stocks_full.remove(stock)

                    stop_loss_sold.append(stock)
                    stop_loss_sold_dates.append([stock, date])
                    
                    stop_loss_df = make_df(buy_prices, return1)
                    # add losses 
                    inc_cash_2 = 10000 + return1[5]
                    df_cash_list.append({return1[1]: inc_cash_2}) 
                    
                    df_list.append(stop_loss_df)

                    stop_loss_sold_selldate.append([stock, return1[1]])
                    dic_bouth_sold[stock] = return1[1]


                # 1/2 then we 
                if return1[3] == 'gain reached':
                    print('YEAAAAAAAAAAAA')
                    
                    bought_stocks_full.remove(stock)
                    # calculate returns from before
                    returns_before = make_df(buy_prices, return1)
                    df_list.append(returns_before)  

                    inc_cash_3 = return1[5]
                    # print(inc_cash_3)
                    df_cash_list.append({return1[1]: inc_cash_3}) 

                    # cash to divide in proportions
                    t_cash = return1[8]
                    bought_stocks_third.append(stock)
                    # day when 100% reached and from where we count new returns 
                    date_reached = return1[6]
                    # price from which we count new returns 
                    price_of_reach = return1[0]
                    
                    # returns until holding 
                    returns_af = Return_holding(stock = stock, date = date_reached, initial_price= price_of_reach, prices = buy_prices)
                    # include to reinvest in stop_loss
                    df_cash_reinv_stop.append({return1[1]: return1[7]}) 
                    # proportional df 
                    bougth_prop.append([stock, date, list(returns_af[4][-1].values())])
                    df_prop = make_df_proportional(buy_prices, returns_af, t_cash)
                    df_list.append(df_prop)
                    df_secured.append(df_prop)

    # finalizing main df
    df_Cash = cash_df(df_cash_list)
    df_Cash_stop_loss = loss_cash_df(df_cash_list, df_cash_reinv_stop)
    df_Secured_f = sec_df(df_secured)

    df_final = pd.concat(df_list, sort= False, join='outer', axis = 1)
    df_final['Portfolio value'] = df_final.sum(axis=1)
    df_final['Secured'] = df_Secured_f

    # cash manipulations
    df_final['Cash'] = df_Cash['Cash']
    df_final_cash = df_final[['Cash']]
    df_final_cash.fillna(method='ffill', inplace = True)
    df_final['Cash'] = df_final_cash['Cash']

    df_final['Cash loss only'] = df_Cash_stop_loss['Cash']
    df_final_cash = df_final[['Cash loss only']]
    df_final_cash.fillna(method='ffill', inplace = True)
    df_final['Cash loss only'] = df_final_cash['Cash loss only']

    df_final['Secured'] = df_final['Secured'].fillna(0)
    print(df_final['Secured'])
    df_final['Liquidative value'] = df_final['Cash'] + df_final['Portfolio value']
    df_final['MAIN'] = df_final['Portfolio value'] - df_final['Secured']



    df_final = df_final.groupby(level=0, axis=1, sort = False).sum()
    df_final.replace(0, np.nan, inplace=True)
    # rename 
    df_final['MAIN - STOCKS VALUE'] = df_final['MAIN']
    df_final['MAIN - CASH LOSS ONLY'] = df_final['Cash loss only']     
    df_final['TOTAL LIQUIDATIVE VALUE'] = df_final['Liquidative value']
    df_final['MAIN - CASH'] = df_final['Cash']
    df_final['MAIN - LIQUIDATIVE VALUE'] = df_final['MAIN - STOCKS VALUE'] + df_final['MAIN - CASH']
    df_final['SECURED - STOCK VALUE'] = df_final['Secured']
    df_final['TOTAL 2 PORTFOLIO'] = df_final['Portfolio value']
    df_final.drop(['Portfolio value', 'Secured', 'Cash', 'Liquidative value', 'Cash loss only', 'MAIN' ],inplace= True, axis=1)
    df_final.to_excel('Graph2.xlsx', 'Main')
    
    # memorandum table
    sold_stocks_df = mem_stop_loss(stop_loss_sold,stop_loss_sold_dates, stop_loss_sold_selldate)
    hold_stocks_df = mem_holding(bought_stocks_mem)
    portion_stocks = mem_portion(bougth_prop)
    df_list_V2 = [sold_stocks_df, hold_stocks_df, portion_stocks ]

    Mem_tabel = pd.concat(df_list_V2, sort= False, join='outer', axis = 0, ignore_index= False)
    # Mem_tabel = Mem_tabel.groupby(Mem_tabel.columns, axis=1, sort = False)
    Mem_tabel.to_excel('Memorandum_Table2.xlsx', 'Memorandum Table')
    # print(Mem_tabel)




# Return(stock = 'FSLY', date = '2020-01-06', initial_price= '78.8', prices = all_prices)
# returns = Return(stock = 'UI', date = '2018-02-01 00:00:00', initial_price= '78.8', prices = all_prices)
# returns2 = Return_holding(stock = 'UI', date = '2018-02-01 00:00:00', initial_price= '78.8', prices = all_prices)
# make_df(all_prices, returns)
# print(returns)
main(his_stocks, all_prices, stock_bought)

# print(all_prices.loc['2015-02-05 00:00    :00', 'UI'])

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.dates import MonthLocator, YearLocator
from matplotlib.dates import AutoDateFormatter, AutoDateLocator
import matplotlib.pyplot as plt
import matplotlib 
import matplotlib.dates as mdates


# # # making graph
graph = pd.read_excel("First_graph2.xlsx", parse_dates=True, sheet_name = "Graph", index_col = 0)

fig, ax = plt.subplots()

graph.plot( ax=ax)

ax.grid(b=True, which='major', color='r', linestyle='-')
ax.grid(b=True, which='minor',linestyle='--')
# months = mdates.MonthLocator(())

# months = mdates.MonthLocator(interval= 3)
# monthsFmt = mdates.DateFormatter('%Y-%m') 

# years = mdates.YearLocator()   
# yearFmt = mdates.DateFormatter('%Y-%m')

# ax.xaxis.set_major_locator(years)
# ax.xaxis.set_major_formatter(yearFmt)

# ax.xaxis.set_major_locator(months)
# ax.xaxis.set_major_formatter(monthsFmt)

# # plt.setp(ax.get_xmajorticklabels(), visible=False)
# matplotlib.rcParams.update({'font.size': 5})

# plt.yscale('log')
# ax.minorticks_on()
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

# ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
# fig.autofmt_xdate()
import matplotlib.pyplot as mplot
# mplot.savefig("foo.pdf", bbox_inches="tight")
# yloc = YearLocator()
# mloc = MonthLocator()
# ax.xaxis.set_major_locator(yloc)
# ax.xaxis.set_minor_locator(mloc)
# ax.xticks(rotation=70)
plt.title('Stop loss 5%')
ax.set_ylim(-300000, 350000)
ax.legend(loc = 'upper left')
mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
plt.show()
# # plt.savefig('graph_prog.png')