import random
import numpy as np
from matplotlib import pyplot as plt
import time 
from threading import Thread

############# Helper funtions  ##########################
def find_argmax(numbers):
    # Return index og max val in list 
    max_number = numbers[0]
    argmax = 0 
    for i,number in enumerate(numbers):
        if number > max_number:
            max_number = number
            argmax = i
    return argmax

def uniform(a, b):
    return random.random() * (b - a) + a

def zeros_matrix(size):
    return [[0] * size for _ in range(size)]

def get_column(matrix, x):
    column = []
    for i in range(len(matrix)):
        column.append(matrix[i][x])
    return column

def prof_means(prof_arr1, prof_arr2):
    return np.mean(prof_arr1, axis=0), np.mean(prof_arr2, axis=0)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def moving_avg(fst_arr, snd_arr, window_size):
    return running_mean(fst_arr, window_size), running_mean(snd_arr, window_size)
     
def profit(pris1, pris2):
    return pris1 * demand(pris1,pris2)

def demand(p1,p2):
        if p1 < p2:
            d = 1 - p1
        elif p1 == p2:
            d = 0.5 * (1 - p1)
        else:
            d = 0
        return d
    

def zeros_list(n): 
    return [0 for x in range(n)]


def Q_Player(prices, Q, epsilon, p2):
    if random.uniform(0,1) < epsilon:
        p3 = int(np.random.choice(len(prices)))
    else:
        _q = get_column(Q, p2)
        p3 = int(find_argmax(_q))
    return p3, 'Q_Player'

def tit4tat(prev, player_index):
    pt = prev[player_index][1]
    return pt, 'tit4tat'


 
def update(Q_table, previous_play, alpha, delta, prices, index):
    if index == 1: 
        p1 = prices[previous_play[0][0]]
        p2 = prices[previous_play[1][0]]
        p22 = prices[previous_play[1][1]]
        pe1 = Q_table[previous_play[0][0]][previous_play[1][0]]

        _q = get_column(Q_table, previous_play[1][1])
        ne1 = p1*demand(p1,p2) + delta * p1 * demand(p1,p22) + delta**2 * Q_table[int(find_argmax(_q))][previous_play[1][1]]
        Q_table[previous_play[0][0]][previous_play[1][0]] = (1-alpha) * pe1 + alpha * ne1
    elif index == 0: 
        p1 = prices[previous_play[1][0]]
        p2 = prices[previous_play[0][0]]
        p22 = prices[previous_play[0][1]]
        pe2 = Q_table[previous_play[1][0]][previous_play[0][0]]

        _q = get_column(Q_table, previous_play[0][1])
        ne2 = p1*demand(p1,p2) + delta* p1*demand(p1,p22) + delta**2 * Q_table[int(find_argmax(_q))][previous_play[0][1]]
        Q_table[previous_play[1][0]][previous_play[0][0]] = (1-alpha) * pe2 + alpha * ne2
    else: 
        raise ValueError("Player index does not match! must be 1 or 0")


def init_matrix(prices): 
    previous_play = zeros_matrix(2)  
    for i in range(len(previous_play)):
        for j in range(len(previous_play)):
            previous_play[i][j] = np.random.choice(len(prices))
    return previous_play 



class simulations(): 
    def __init__(self, learners, periods) -> None:
        self.proi = []
        self.proi2 = [] 
        self.arri = []
        self.arr1i = []
        self.Q_ti = []
        self.player0_name = ""
        self.player1_name = ""

        self.total_pro_arr = np.zeros((learners,periods-2),dtype=np.ndarray)
        self.total_pro_arr2 = np.zeros((learners,periods-2),dtype=np.ndarray)
        
        self.avg_profit = np.zeros(learners)
        self.avg_profit2 = np.zeros(learners)

        self.main()

    ############# Game simulator ##########################
    def game(self, prices, periods, alpha, theta, delta, gameindex):
        n_prices = len(prices)

        Q1_table = zeros_matrix(n_prices)
        Q2_table = zeros_matrix(n_prices)

        # Representere priserne som 'felter' 0-6  (len=7) som man kan spille
        # Værdien indeholder sidste spillede priser
        
        previous_play = init_matrix(prices)

        n_price_matrix = int(periods/2)-1
        P1_priser = zeros_list(n_price_matrix)
        P2_priser = zeros_list(n_price_matrix)

        n_profit_matrix = int(periods-2)
        profit_1_arr = zeros_list(n_profit_matrix)
        profit_2_arr = zeros_list(n_profit_matrix)


        # Skubber tidsperiodens start for at vi representere det fra t=0
        # Og ikke får index out of range
        t = 3 

        # Counters til at styre matrix-placering
        i_counter = 0
        j_counter = 0

        
        for t in range(t, periods+1):
            epsilon = (1-theta)**t
            if t % 2 != 0: 
                # Defining the player 
                player_index = 1

                # Myopic Q-learner -> delta = 0 -> meaning it prefers today rather than tomorrow(Implement by chaning Updates delta -> Myopic_delta)
                Myopic_delta = 0

                update(Q1_table, previous_play, alpha, delta, prices, player_index)

                # Computing the index of price for the player 
                # price_index_player_1, player1_name = tit4tat(previous_play, player_index)
                price_index_player_1, player1_name = Q_Player(prices, Q1_table, epsilon, previous_play[player_index][1])

                # Updating the Q-table with indexs of previous plays
                previous_play[0][0] = previous_play[0][1]
                previous_play[0][1] = price_index_player_1
                previous_play[1][0] = previous_play[1][1]

                # Adding the price of the play to storage-list
                P1_priser[i_counter] = (prices[price_index_player_1])

                # Increment player counter
                i_counter += 1

                # Adding the profit of the play to storage-list
                profit_2_arr[t-3] = profit(prices[previous_play[1][1]], prices[price_index_player_1])
                profit_1_arr[t-3] = profit(prices[price_index_player_1], prices[previous_play[1][1]])

            else: 
                # Defining the player 
                player_index = 0

                update(Q2_table, previous_play, alpha, delta, prices, player_index)

                # Computing the index of price for the player 
                p_j, player0_name = Q_Player(prices, Q2_table, epsilon, previous_play[player_index][1])

                # Updating the Q-table with indexs of previous plays
                previous_play[1][0] = previous_play[1][1]
                previous_play[1][1] = p_j
                previous_play[0][0] = previous_play[0][1]

                # Adding the price of the play to storage-list
                P2_priser[j_counter] = (prices[p_j])

                # increment player counter
                j_counter += 1

                # Adding the profit of the play to storage-list
                profit_1_arr[t-3] = profit(prices[previous_play[0][1]], prices[p_j])
                profit_2_arr[t-3] = profit(prices[p_j], prices[previous_play[0][1]])
            
        # self.proi = profit_1_arr
        # self.proi2 = profit_2_arr
        # self.arri = P1_priser
        # self.arr1i = P2_priser
        # self.Q_ti = Q2_table
        self.player0_name = player0_name
        self.player1_name = player1_name
        self.total_pro_arr[gameindex] = profit_1_arr
        self.total_pro_arr2[gameindex] = profit_2_arr
    
        self.avg_profit[gameindex] = np.mean(profit_1_arr[-10000:])
        self.avg_profit2[gameindex] = np.mean(profit_2_arr[-10000:])


    #simulating multiple runs and averaging profit
    def many_games(self, prices, periods, alpha, theta, learners,delta):

        threadlist = []
        for gameindex in range(learners):
            print('Init #',gameindex+1 ,'of ', learners , 'runs')
            t1 = Thread(target = self.game, args=[prices, periods, alpha, theta, delta, gameindex])
            t1.setDaemon(True)
            threadlist.append((gameindex, t1))


        for i,t in threadlist: 
            print('run #',i+1 ,'of ', learners , 'runs')
            t.start()


        for i,t in threadlist: 
            print('Completed #',i+1 ,'of ', learners , 'runs')
            t.join()

    
    def main(self): 
        price = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1])
        alpha = 0.3
        theta = 0.0000276306393827805
        delta = 0.95
        window_size = 1000
        t0 = time.time()

        # many_profs, many_profs2, avg_profit, avg_profit2, player0_name, player1_name = 
        self.many_games(price, n_periods, alpha, theta, runs, delta)
        samlet_prof, samlet_prof2= prof_means(self.total_pro_arr, self.total_pro_arr2)
        profitability_arr1, profitability_arr0 = moving_avg(samlet_prof, samlet_prof2, window_size)

        print('Runtime: ', time.time()-t0)

        plt.plot(np.arange(0,n_periods-window_size-1),profitability_arr1,'-',label=self.player1_name)
        plt.plot(np.arange(0,n_periods-window_size-1),profitability_arr0, '-', label=self.player0_name)
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






n_periods = 500000
runs = 10
sim = simulations(runs, n_periods)


















# def one_game(): 
#     prices = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]
#     n_periods = 500000
#     alpha = 0.3
#     theta = 0.0000276306393827805
#     delta = 0.95
#     proi, proi2, arri, arr1i, Q_ti = game(prices, n_periods, alpha, theta, delta)



