#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Compile to .so: 
// gcc -fPIC -shared -o WOA_Bach_lib.so WOA_Bach_lib.c 


// compile and run C: 
// gcc  WOA_Bach_lib.c -o WOA_Bach_lib  && ./WOA_Bach_lib 



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int indexOfMax(double *numbers, int size) {
    int maxIndex = 0; // Assume the first element is the max
    for (int i = 1; i < size; i++) {
        if (numbers[i] > numbers[maxIndex]) {
            maxIndex = i;
        }
    }
    return maxIndex;
}

double demand(double p1, double p2) {
    double d;
    if (p1 < p2) {
        d = 1.0 - p1;
    } else if (p1 == p2) {
        d = 0.5 * (1.0 - p1);
    } else {
        d = 0.0;
    }
    return d;
}

double profit(double pris1, double pris2) {
    double d = demand(pris1, pris2);
    return pris1 * d;
}

int row_major_index(int row, int col, int length) {
    return (row * length) + col;
}


double* zeros_matrix(int length){
    double* Q_table = malloc(length* length * sizeof(double));
    for (int i = 0; i < length*length; i++) {
        Q_table[i] = 0.0; 
    }
    return Q_table; 
}

double* getColumn(double* matrix, int column_index, int matrix_height, int matrix_length) {
    double* result = malloc(matrix_height * sizeof(double));
    for (int i = 0; i < matrix_height; i++) {
        result[i] = matrix[row_major_index(i, column_index, matrix_length)];
    }
    return result;
}


void seedRandom() {
    clock_t t;
    t = clock();
    srand(t);
}


int randomIntBetween(int min, int max) {
    int random = (rand() % (max - min + 1)) + min; 
    return random ;  // Generate a random integer between min and max (inclusive)
}


int * init_matrix(int lengthOfMatrix, int n_prices) {
    int* Q_table = malloc(lengthOfMatrix * sizeof(int));
    for (int i = 0; i < lengthOfMatrix; i++) {
        Q_table[i] = randomIntBetween(0, n_prices-1);  
    }
    return Q_table; 
}

double* zeros_list(int length){
    double* list = malloc(length * sizeof(double));
    for (int i = 0; i < length; i++) {
        list[i] = 0.0; 
    }
    return list; 
}

int tit4tat(int* prev, int player_index, int prev_length ){
    return prev[row_major_index(player_index, 1, prev_length )]; 
}

double power(double base, int exponent) {
    if (exponent == 0) {
        return 1.0;
    }
    double temp = power(base, exponent/2);
    if (exponent % 2 == 0) {
        return temp * temp;
    } else {
        if (exponent > 0) {
            return base * temp * temp;
        } else {
            return (temp * temp) / base;
        }
    }
}


double random_uniform_double(double min_val, double max_val) {
    double range = (max_val - min_val);
    double rand_val = ((double)rand() / RAND_MAX) * range + min_val;
    return rand_val;
}

int Q_Player(double* prices, double* Q, int Q_length, int Q_height, double epsilon, int p2){
    int p3; 
    if (random_uniform_double(0,1) < epsilon){
        p3 = randomIntBetween(0,Q_length-1);  
    }
    else{
        double* columnRes = getColumn(Q, p2, Q_height, Q_length);
        p3 = indexOfMax(columnRes, Q_length); 
    }
    return p3;
}



void update(double* Q_table,int Q_length,  double* prices, double delta, double alpha,  int* previous_play, int previous_playlength, int playerIndex){
    double p1,p2, p22, pe1, pe2; 
    int _argMaxIndex; 
    if (playerIndex == 1) {
        p1 = prices[previous_play[row_major_index(0,0,previous_playlength)]]; 
        p2 = prices[previous_play[row_major_index(1,0,previous_playlength)]]; 
        p22 = prices[previous_play[row_major_index(1,1,previous_playlength)]]; 
        pe1 = Q_table[row_major_index(previous_play[row_major_index(0,0,previous_playlength)], previous_play[row_major_index(1,0,previous_playlength)],Q_length )]; 

        // Finds col 
        double* columnRes = getColumn(Q_table, previous_play[row_major_index(1,1,previous_playlength)], Q_length, Q_length);
        
        // Compute Q_val to insert into Q_table
        double _Q_table_val = Q_table[row_major_index(indexOfMax(columnRes, Q_length ),previous_play[row_major_index(1,1,previous_playlength)], Q_length )]; 
        double ne1 =  (p1*demand(p1,p2)) + (delta * p1 * demand(p1,p22)) + (power(delta,2) * _Q_table_val ); 
        Q_table[row_major_index( previous_play[row_major_index(0,0,previous_playlength)],previous_play[row_major_index(1,0,previous_playlength)], Q_length)] = ((1-alpha) * pe1) + (alpha * ne1); 
        
        free(columnRes); 
    }
    if (playerIndex == 0) {
        p1 = prices[previous_play[row_major_index(1,0,previous_playlength)]]; 
        p2 = prices[previous_play[row_major_index(0,0,previous_playlength)]]; 
        p22 = prices[previous_play[row_major_index(0,1,previous_playlength)]]; 
        pe2 = Q_table[row_major_index(previous_play[row_major_index(1,0,previous_playlength)], previous_play[row_major_index(0,0,previous_playlength)],Q_length )]; 
  
        // Finds col 
        double* columnRes = getColumn(Q_table, previous_play[row_major_index(0,1,previous_playlength)], Q_length,Q_length);

        // Compute Q_val to insert into Q_table
        double _Q_table_val = Q_table[row_major_index(indexOfMax(columnRes, Q_length ),previous_play[row_major_index(0,1,previous_playlength)], Q_length )]; 
        double ne2 =  (p1*demand(p1,p2)) + (delta * p1 * demand(p1,p22)) + (power(delta,2) * _Q_table_val) ; 
        Q_table[row_major_index( previous_play[row_major_index(1,0,previous_playlength)],previous_play[row_major_index(0,0,previous_playlength)], Q_length)] = ((1-alpha) * pe2) + (alpha * ne2); ; 
    
        free(columnRes); 
    }
}


int game(int periods, int n_prices, double* prices, double delta, double alpha, double theta , double* P1_priser, double* P2_priser, double* profit_1_arr, double* profit_2_arr){
    seedRandom(); 
    int playerIndex, price_index_player_1, price_index_player_2;
    
    double epsilon; 
    double* Q1_table = zeros_matrix(n_prices); 
    double* Q2_table = zeros_matrix(n_prices); 

    int previous_playlength = 2 ; 

    int* previous_play = init_matrix(previous_playlength*previous_playlength, n_prices); 
  
    // Counters til at styre matrix-placering
    int i_counter = 0; 
    int j_counter = 0;  
    int myopic_delta = 0 ; 
    for (int t = 3; t < periods+1 ; t++) {
        epsilon = power((1-theta),t); 

        if (t % 2 != 0){
            playerIndex = 1 ; 



            update(Q1_table, n_prices, prices,  myopic_delta, alpha, previous_play, previous_playlength, playerIndex ); 

            // Computing the index of price for the player 
            price_index_player_1 = Q_Player(prices, Q1_table, n_prices, n_prices, epsilon, previous_play[row_major_index(playerIndex,1,previous_playlength)]); 
            // price_index_player_1 = tit4tat( previous_play, playerIndex, previous_playlength ); 
        


            previous_play[row_major_index(0,0, previous_playlength)] = previous_play[row_major_index(0,1, previous_playlength)]; 
            previous_play[row_major_index(0,1, previous_playlength)] = price_index_player_1; 
            previous_play[row_major_index(1,0, previous_playlength)] = previous_play[row_major_index(1,1, previous_playlength)]; 

            P1_priser[i_counter] = prices[price_index_player_1]; 

            // Increment player counter
            i_counter += 1; 

            // Adding the profit of the play to storage-list
            profit_2_arr[t-3] = profit(prices[previous_play[row_major_index(1,1, previous_playlength)]], prices[price_index_player_1]); 
            profit_1_arr[t-3] = profit(prices[price_index_player_1], prices[previous_play[row_major_index(1,1, previous_playlength)]]); 

        }

        else{
            playerIndex = 0; 
            update(Q2_table, n_prices, prices,  delta, alpha, previous_play, previous_playlength, playerIndex ); 

            // Computing the index of price for the player 
            price_index_player_2 = Q_Player(prices, Q2_table, n_prices, n_prices, epsilon, previous_play[row_major_index(playerIndex,1,previous_playlength)]); 
            // price_index_player_2 = tit4tat( previous_play, playerIndex, previous_playlength ); 
            
            previous_play[row_major_index(1,0, previous_playlength)] = previous_play[row_major_index(1,1, previous_playlength)]; 
            previous_play[row_major_index(1,1, previous_playlength)] = price_index_player_2; 
            previous_play[row_major_index(0,0, previous_playlength)] = previous_play[row_major_index(0,1, previous_playlength)]; 

            P2_priser[j_counter] = prices[price_index_player_2]; 

            // Increment player counter
            j_counter += 1; 

            // Adding the profit of the play to storage-list
            profit_1_arr[t-3] = profit(prices[previous_play[row_major_index(0,1, previous_playlength)]], prices[price_index_player_2]); 
            profit_2_arr[t-3] = profit(prices[price_index_player_2], prices[previous_play[row_major_index(0,1, previous_playlength)]]); 
        }
    }

    free(previous_play); 
    free(Q1_table); 
    free(Q2_table); 
    return 1 ; 
}
