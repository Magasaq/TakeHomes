import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_results(res):
    mid = (res['ask_rate'].apply(lambda x: x[0]) + res['bid_rate'].apply(lambda x: x[0])) * 0.5
    plt.figure(figsize=(15, 10))
    n_plots = 3
    plt.subplot(n_plots, 1, 1)
    plt.plot(res['pnl'], label='PnL')
    plt.plot(res['pnl_after_fee'], label='PnL after fee')
    plt.grid()
    plt.legend(loc="best")

    plt.subplot(n_plots, 1, 2)
    plt.plot(mid, label='Midprice')
    plt.grid()
    plt.legend(loc="best")

    plt.subplot(n_plots, 1, 3)
    plt.plot(res['pos'], label='pos change')
    plt.grid()
    plt.legend(loc="best")
    plt.show()
    return


def make_action(amounts, price, size, commission, max_depth):
    real_trade_size = min(size, sum(amounts[:max_depth]))
    commission_paid = price * real_trade_size * commission
    return price, real_trade_size, commission_paid


def calc_trade_profit(buys, sells):
    price_diff = sells[0][0] - buys[0][0]
    curr_size = min(sells[0][1], buys[0][1])
    profit = price_diff * curr_size
    sells[0][1] -= curr_size
    buys[0][1] -= curr_size
    if buys[0][1] == 0:
        buys = buys[1:]
    if sells[0][1] == 0:
        sells = sells[1:]
    return profit, buys, sells


def backtest(df, strategy, strategy_params, viz=False):
    '''
    Описание
    --------------
    Функция для симуляции работы стратегии. В каждый тик биржи функция обращается к стратегии и получает от неё объем сделки.
    После чего совершает сделку.
    
    Параметры
    --------------
    df : DataFrame
        DataFrame, содержащий информацию о стакане и сигнал от модели
    strategy : function
        Стратегия
    strategy_params : dict
        Параметры стратегии
    viz : bool
        Нужно ли рисовать график

    Результат
    --------------
    summary : dict
        Результат работы стратегии: pnl, pnl после комиссии, количество сделок
    result : DataFrame
        DataFrame, содержащий информацию о стакане, сигнал модели, pnl, pnl_after_fee, decisions, pos в каждый момент времени
    '''
    data_dict = df.to_dict('list')
    max_ticks = df.shape[0]
    
    buys = []
    sells = []
    pnl = [0.]
    pnl_after_fee = [0.]
    decisions = []
    pos = [0.]
    actions = 0
    profits = []
    # backtest params
    max_depth = 2
    commission = 0.0001
    
    for index in range(max_ticks):
        pos_change = 0
        curr_pnl = 0
        curr_pnl_after_fee = 0
        depth_index = 0
        curr_pos = pos[-1]
        pred = data_dict['signal'][index]
        ask_rate = data_dict['ask_rate'][index]
        ask_amount = data_dict['ask_amount'][index]
        bid_rate = data_dict['bid_rate'][index]
        bid_amount = data_dict['bid_amount'][index]
        
        orderbook_state = [ask_rate, ask_amount, bid_rate, bid_amount]
        strategy_state = [index, curr_pos, pred]
        
        des = strategy(strategy_state, orderbook_state, **strategy_params)  # make decision
        
        if index == max_ticks - 1:  # force close on last tick
            des = -pos[-1]
        ideal_size = abs(des)
        
        if des > 0:  # buy
            if pos[-1] >= 1:
                print(f"Warning: position limit! Index: {index}. Order size: {des}. Current position: {pos[-1]}. Max position: {1}")
                continue
            price, real_trade_size, commission_paid = make_action(ask_amount, ask_rate[0], ideal_size, commission, max_depth)
            buys.append([price, real_trade_size])
            curr_pnl_after_fee -= commission_paid
            pos_change += real_trade_size
            actions += 1
        if des < 0:  # sell
            if pos[-1] <= -1:
                print(f"Warning: position limit! Index: {index}. Order size: {des}. Current position: {pos[-1]}. Max position: {1}")
                continue
            price, real_trade_size, commission_paid = make_action(bid_amount, bid_rate[0], ideal_size, commission, max_depth)
            sells.append([price, real_trade_size])
            curr_pnl_after_fee -= commission_paid
            pos_change -= real_trade_size
            actions += 1
        while len(sells) > 0 and len(buys) > 0:  # calc profit
            profit, buys, sells = calc_trade_profit(buys, sells)
            curr_pnl += profit
            curr_pnl_after_fee += profit
            
        # save state
        pos.append(pos[-1] + pos_change)
        pnl.append(pnl[-1] + curr_pnl)
        pnl_after_fee.append(pnl_after_fee[-1] + curr_pnl_after_fee)
        decisions.append('buy' if pos_change > 0 else ('sell' if pos_change < 0 else 'hold'))
        
    # final results
    result = pd.DataFrame.from_dict(data_dict)
    result['pnl'] = pnl[1:]
    result['pnl_after_fee'] = pnl_after_fee[1:]
    result['decisions'] = decisions
    result['pos'] = pos[1:]
    result.index = df.index
    summary = {}
    summary['pnl'] = pnl[-1]
    summary['pnl_after_fee'] = pnl_after_fee[-1]
    summary['actions'] = actions
    
    if viz:
        print(summary)
        plot_results(result)
    
    return summary, result
    
