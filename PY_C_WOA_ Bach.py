import numpy as np
from matplotlib import pyplot as plt
import time 
import ctypes 

############# Helper funtions  ##########################
lib = ctypes.cdll.LoadLibrary('/Users/williamsorensen/Desktop/bach-pyFiler/WOA_Bach_lib.so')
game = lib.game
game.restype = ctypes.c_int
game.argtypes = [
    ctypes.c_int,  # periods
    ctypes.c_int,  # n_prices
    np.ctypeslib.ndpointer(dtype=np.float64),  # prices
    ctypes.c_double,  # delta
    ctypes.c_double,  # alpha
    ctypes.c_double,  # theta
    np.ctypeslib.ndpointer(dtype=np.float64),  # P1_priser
    np.ctypeslib.ndpointer(dtype=np.float64),  # P2_priser
    np.ctypeslib.ndpointer(dtype=np.float64),  # profit_1_arr
    np.ctypeslib.ndpointer(dtype=np.float64),  # profit_2_arr
]


def prof_means(prof_arr1, prof_arr2):
    return np.mean(prof_arr1, axis=0), np.mean(prof_arr2, axis=0)


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def moving_avg(fst_arr, snd_arr, window_size):
    return running_mean(fst_arr, window_size), running_mean(snd_arr, window_size)
     


############# Game simulator ##########################
def init_game(prices, periods, alpha, theta, delta):
    n_prices = len(prices)

    n_price_matrix = int(periods/2)-1
    P1_priser = np.zeros(n_price_matrix)
    P2_priser = np.zeros(n_price_matrix)

    n_profit_matrix = int(periods-2)
    profit_1_arr = np.zeros(n_profit_matrix)
    profit_2_arr = np.zeros(n_profit_matrix)

    # Call the function
    result = game(periods, n_prices, prices, delta, alpha, theta, P1_priser, P2_priser, profit_1_arr, profit_2_arr)

    # Check the result
    if result != 1:
        print('Error occurred')

    # Q2_table, player0_name, player1_name 
    return profit_1_arr, profit_2_arr, P1_priser, P2_priser


#simulating multiple runs and averaging profit
def many_games(prices, periods, alpha, theta, learners,delta):
    total_pro_arr = np.zeros((learners,periods-2),dtype=np.ndarray)
    total_pro_arr2 = np.zeros((learners,periods-2),dtype=np.ndarray)
    
    avg_profit = np.zeros(learners)
    avg_profit2 = np.zeros(learners)


    for i in range(learners):
        print('run #',i+1 ,'of ', learners , 'runs')
        profit_1_arr, profit_2_arr, P1_priser, P2_priser = init_game(prices, periods, alpha, theta, delta)
        total_pro_arr[i] = profit_1_arr
        total_pro_arr2[i] = profit_2_arr
    
        avg_profit[i] = np.mean(profit_1_arr[-10000:])
        avg_profit2[i] = np.mean(profit_2_arr[-10000:])

    return total_pro_arr, total_pro_arr2, avg_profit, avg_profit2 



def main(): 
    price = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1])
    n_periods = 500000
    alpha = 0.3
    theta = 0.0000276306393827805
    delta = 0.95
    runs = 100
    window_size = 1000
    t0 = time.time()

    many_profs, many_profs2, avg_profit, avg_profit2 = many_games(price, n_periods, alpha, theta, runs, delta)

    samlet_prof, samlet_prof2= prof_means(many_profs, many_profs2)
    profitability_arr1, profitability_arr0 = moving_avg(samlet_prof, samlet_prof2, window_size)

    print('Runtime: ', time.time()-t0)

    plt.plot(np.arange(0,n_periods-window_size-1),profitability_arr1,'-',label="player1_name")
    plt.plot(np.arange(0,n_periods-window_size-1),profitability_arr0, '-', label="player0_name")
    plt.axhline(y=0.125, color='k', linestyle = '--')
    plt.axhline(y=0.061, color='k', linestyle = '--')
    plt.xlabel("Time")
    plt.ylabel("Profitability")
    plt.ylim(0.00,0.15)
    plt.legend()
    plt.show()

    combi_arr = np.mean((np.vstack((profitability_arr1, profitability_arr0))), axis=0)
    plt.plot(np.arange(0,n_periods-window_size-1),combi_arr,'-',label='Average profit')
    plt.axhline(y=0.125, color='k', linestyle = '--')
    plt.axhline(y=0.0611, color='k', linestyle = '--')
    plt.xlabel("Time")
    plt.ylabel("Profitability")
    plt.ylim(0.00,0.15)
    plt.legend()
    plt.show()



main()

