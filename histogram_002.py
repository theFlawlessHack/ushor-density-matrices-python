from __future__ import division
#
# s_978 is running again
#
# to run this code type:
# "import histogram_002", then
# "histogram_002.z(1)"
#
# initialize python console width and height
#import os
#os.system("mode con: cols=88 lines=256")

# 1956 is a tough one 

import os
os.system('mode con: cols=88 lines=256')

fileout = open('workfile', 'w')

def z(n):    #
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import scipy
    import pdb
    from scipy.stats import norm

    #globals()[]NUMBER_STABLE_STATES = 7
    globals()['NUMBER_STABLE_STATES'] = 7
    index_election_high = 52 #8 is a hard(est) one

    cumulative_transition_matrix = np.zeros(shape=(NUMBER_STABLE_STATES,NUMBER_STABLE_STATES))
    T_master = np.zeros(shape=(NUMBER_STABLE_STATES, NUMBER_STABLE_STATES, index_election_high))
    entropy_master = np.zeros(shape=(NUMBER_STABLE_STATES,NUMBER_STABLE_STATES))
    
    HISTORICAL_RECORD = make_historical_data(0)

    historical_averages, historical_std, two_party_averages, two_party_std, predicted_averages, predicted_std = calculateMinimalPoliticalSignificance(HISTORICAL_RECORD)
    fileout.write(' ')
    fileout.write('historical_averages = ')
    fileout.write(str(np.matrix.round(historical_averages*1000)/1000))
    fileout.write('historical_std = ')
    fileout.write(str(np.matrix.round(historical_std*1000)/1000))
    fileout.write('two_party_averages = ')
    fileout.write(str(np.matrix.round(two_party_averages*1000)/1000))
    fileout.write('two_party std = ')
    fileout.write(str(np.matrix.round(two_party_std*1000)/1000))
    fileout.write('predicted_averages = ')
    fileout.write(str(np.matrix.round(predicted_averages*1000)/1000))
    fileout.write('predicted std = ')
    fileout.write(str(np.matrix.round(predicted_std*1000)/1000))

    #return

    for index_election in range(1, index_election_high + 1): # 4th election anhilation;; 7th election has generation ;;; eletions 23, 24 need to be looked at respect to historical record
        
        transition_matrix_temp = make_transition_matrix(NUMBER_STABLE_STATES, HISTORICAL_RECORD, index_election)        
        sum_transition_matrix = sum(sum(transition_matrix_temp))

        cumulative_transition_matrix = cumulative_transition_matrix + transition_matrix_temp

        average_transition_matrix = cumulative_transition_matrix/index_election
        sum_average_transition_matrix = sum(sum(average_transition_matrix))

        if 1 == 1:
            fileout.write('index election = {}'.format(index_election))
            fileout.write('transition_matrix = ')
            fileout.write(str(np.matrix.round(transition_matrix_temp*1000)/1000))
            fileout.write('sum of sum of transition matrix = ')
            fileout.write(str(round(sum_transition_matrix*1000)/1000))
            fileout.write('index_election = {} average transition_matrix = '.format(index_election))
            fileout.write(str(np.matrix.round(average_transition_matrix*100)/100))


        if abs(sum_transition_matrix - 1) > 0.00001:
            fileout.write('{} not normalized correctly 11'.format(index_election))
            return
        elif abs(sum_average_transition_matrix - 1) > 0.00001:
            fileout.write('{} not normalized correctly 22'.format(index_election))
            return
        elif index_election == 23:
            tommy = (1 == 2)
            #return    
        if index_election == 1:
            tommy = (1 == 0)
            #return

        #globals()['T_%s' % index_election] = transition_matrix_temp
        for dex_1 in range(0, 7):
            for dex_2 in range(0, 7):
                T_master[dex_1][dex_2][index_election-1] = transition_matrix_temp[dex_1][dex_2]
                if transition_matrix_temp[dex_1][dex_2] != 0:
                    entropy_master[dex_1][dex_2] = entropy_master[dex_1][dex_2] - transition_matrix_temp[dex_1][dex_2]*np.log(transition_matrix_temp[dex_1][dex_2])

    #fileout.write('Averaged transition_matrix = ')
    #fileout.write(np.matrix.round(average_transition_matrix*1000)/1000)
    #fileout.write(sum(sum(average_transition_matrix)))

    signal = np.zeros(shape=(NUMBER_STABLE_STATES, NUMBER_STABLE_STATES))
    noise = np.zeros(shape=(NUMBER_STABLE_STATES, NUMBER_STABLE_STATES))
    for dex_1 in range(0, 7):
        for dex_2 in range(0, 7):
            signal[dex_1][dex_2] = np.mean(T_master[dex_1][dex_2])
            noise[dex_1][dex_2] = np.std(T_master[dex_1][dex_2])

    fileout.write(' ')
    fileout.write(' ')
    fileout.write('TAKE I ')
    fileout.write(' ')
    fileout.write('Signal')
    fileout.write(str(np.matrix.round(signal*1000)/1000))
    fileout.write(' ')
    conf_low = signal - 3.496*noise/np.sqrt(52)
    fileout.write('99.9% conf_low @ 50 samples  =')
    fileout.write(str(np.round(conf_low*1000)/1000))
    fileout.write(' ')
    fileout.write('Entropy  =')
    fileout.write(str(np.round(entropy_master*100)/100))
    fileout.write(' ')
    fileout.write('Total Entropy  =')
    fileout.write(str(np.round(np.sum(np.sum(entropy_master))*1000)/1000))
    

    matrix_temp = np.zeros(shape=(NUMBER_STABLE_STATES, NUMBER_STABLE_STATES))
    matrix_temp[0][0] = 1
    matrix_temp[1][1] = 1
    matrix_temp[2][2] = 1
    matrix_temp[3][2] = 1
    matrix_temp[4][2] = 1
    matrix_temp[5][2] = 1
    matrix_temp[6][6] = 1
    #fileout.write(matrix_temp)
    
    signal, noise = chunk1(matrix_temp, T_master)
    fileout.write(' ')
    fileout.write(' ')
    fileout.write('TAKE II ')
    fileout.write(' ')
    fileout.write('Signal')
    fileout.write(np.matrix.round(signal*1000)/1000)
    fileout.write(' ')
    conf_low = signal - 3.496*noise/np.sqrt(52)
    fileout.write('99.9% conf_low @ 50 samples  =')
    fileout.write(str(np.round(conf_low*1000)/1000))

    signal, noise = chunk2(matrix_temp, T_master)
    fileout.write(' ')
    fileout.write(' ')
    fileout.write('TAKE III ')
    fileout.write(' ')
    fileout.write('Signal')
    fileout.write(str(np.matrix.round(signal*1000)/1000))
    fileout.write(' ')
    conf_low = signal - 3.496*noise/np.sqrt(52)
    fileout.write('99.9% conf_low @ 50 samples  =')
    fileout.write(str(np.round(conf_low*1000)/1000))

    signal, noise = chunk3(matrix_temp, T_master)
    fileout.write(' ')
    fileout.write(' ')
    fileout.write('TAKE IV ')
    fileout.write(' ')
    fileout.write('Signal')
    fileout.write(str(np.matrix.round(signal*1000)/1000))
    fileout.write(' ')
    conf_low = signal - 3.496*noise/np.sqrt(52)
    fileout.write('99.9% conf_low @ 50 samples  =')
    fileout.write(str(np.round(conf_low*1000)/1000))
    

def calculateMinimalPoliticalSignificance(HISTORICAL_RECORD):
    import numpy as np
    
    number_elections = 52
    
    first_place = np.zeros(shape=(number_elections, 1))
    second_place = np.zeros(shape=(number_elections, 1))
    third_place = np.zeros(shape=(number_elections, 1))
    fourth_place = np.zeros(shape=(number_elections, 1))
    fifth_place = np.zeros(shape=(number_elections, 1))
    two_party_first = np.zeros(shape=(number_elections, 1))
    two_party_second = np.zeros(shape=(number_elections, 1))
    minimal_first = np.zeros(shape=(number_elections, 1))
    minimal_second = np.zeros(shape=(number_elections, 1))
    minimal_third = np.zeros(shape=(number_elections, 1))
    historical_averages = np.zeros(shape=(5, 1))
    historical_std = np.zeros(shape=(5, 1))
    two_party_averages = np.zeros(shape=(2, 1))
    two_party_std = np.zeros(shape=(2, 1))
    predicted_averages = np.zeros(shape=(3, 1))
    predicted_std = np.zeros(shape=(3, 1))
    
    for dex in range(1, number_elections):
        first_place[dex-1] = np.sum(HISTORICAL_RECORD[(dex-1)*8 + 1][3])/HISTORICAL_RECORD[(dex-1)*8][3]
        second_place[dex-1] = np.sum(HISTORICAL_RECORD[(dex-1)*8 + 2][3])/HISTORICAL_RECORD[(dex-1)*8][3]
        third_place[dex-1] = np.sum(HISTORICAL_RECORD[(dex-1)*8 + 3][3])/HISTORICAL_RECORD[(dex-1)*8][3]
        fourth_place[dex-1] = np.sum(HISTORICAL_RECORD[(dex-1)*8 + 4][3])/HISTORICAL_RECORD[(dex-1)*8][3]
        fifth_place[dex-1] = np.sum(HISTORICAL_RECORD[(dex-1)*8 + 5][3])/HISTORICAL_RECORD[(dex-1)*8][3]

        two_party_first[dex-1] = np.sum(HISTORICAL_RECORD[(dex-1)*8 + 1][3])/HISTORICAL_RECORD[(dex-1)*8][3]
        two_party_second[dex-1] = np.sum(HISTORICAL_RECORD[(dex-1)*8 + 2][3])/HISTORICAL_RECORD[(dex-1)*8][3] + np.sum(HISTORICAL_RECORD[(dex-1)*8 + 3][3])/HISTORICAL_RECORD[(dex-1)*8][3] + np.sum(HISTORICAL_RECORD[(dex-1)*8 + 4][3])/HISTORICAL_RECORD[(dex-1)*8][3] + np.sum(HISTORICAL_RECORD[(dex-1)*8 + 5][3])/HISTORICAL_RECORD[(dex-1)*8][3]
        
        minimal_first[dex-1] =  (first_place[dex-1])/(1 + min([(first_place[dex-1] - second_place[dex-1]), second_place[dex-1], first_place[dex-1]]))
        minimal_second[dex-1] =  (second_place[dex-1])/(1 + min([(first_place[dex-1] - second_place[dex-1]), second_place[dex-1], first_place[dex-1]]))
        minimal_third[dex-1] =  min([(first_place[dex-1] - second_place[dex-1]), second_place[dex-1], first_place[dex-1]])/(1 + min([(first_place[dex-1] - second_place[dex-1]), second_place[dex-1], first_place[dex-1]]))

    historical_averages[0] = np.mean(first_place)
    historical_averages[1] = np.mean(second_place)
    historical_averages[2] = np.mean(third_place)
    historical_averages[3] = np.mean(fourth_place)
    historical_averages[4] = np.mean(fifth_place)

    historical_std[0] = np.std(first_place)
    historical_std[1] = np.std(second_place)
    historical_std[2] = np.std(third_place)
    historical_std[3] = np.std(fourth_place)
    historical_std[4] = np.std(fifth_place)

    two_party_averages[0] = np.mean(two_party_first)
    two_party_averages[1] = np.mean(two_party_second)
    
    two_party_std[0] = np.std(two_party_first)
    two_party_std[1] = np.std(two_party_second)

    predicted_averages[0] = np.mean(minimal_first)
    predicted_averages[1] = np.mean(minimal_second)
    predicted_averages[2] = np.mean(minimal_third)
    
    predicted_std[0] = np.std(minimal_first)
    predicted_std[1] = np.std(minimal_second)
    predicted_std[2] = np.std(minimal_third)

    return historical_averages, historical_std, two_party_averages, two_party_std, predicted_averages, predicted_std

def chunk3(matrix_temp, T_master):
    import numpy as np

    rank_out = 2
    
    signal_out = np.zeros(shape=(rank_out, rank_out))
    noise_out = np.zeros(shape=(rank_out, rank_out))
        
    # there are ___ sections of this chunk.
    # part 1)
    d1 = 0
    d2 = 0
    signal_out[d1][d2] = np.mean(T_master[d1][d2])
    noise_out[d1][d2] = np.std(T_master[d1][d2])
    # part 2)
    d1 = 0
    d2 = 1
    signal_out[d1][d2] = np.mean(T_master[d1][1]+  T_master[d1][2] + T_master[d1][3] + T_master[d1][4] + T_master[d1][5] + T_master[d1][6])
    noise_out[d1][d2] = np.std(T_master[d1][1] + T_master[d1][2] + T_master[d1][3] + T_master[d1][4] + T_master[d1][5] + T_master[d1][6])
    # part 3)
    d1 = 1
    d2 = 0
    signal_out[d1][d2] = np.mean(T_master[1][d2] + T_master[2][d2] + T_master[3][d2] + T_master[4][d2] + T_master[5][d2] + T_master[6][d2])
    noise_out[d1][d2] = np.std(T_master[1][d2] + T_master[2][d2] + T_master[3][d2] + T_master[4][d2] + T_master[5][d2] + T_master[6][d2])
    # part 4)
    d1 = 1
    d2 = 1
    signal_out[d1][d2] = np.mean(T_master[1][1] + T_master[1][2] + T_master[1][3] + T_master[1][4] + T_master[1][5] + T_master[1][6] + T_master[2][1] + T_master[2][2] + T_master[2][3] + T_master[2][4] + T_master[2][5] + T_master[2][6] + T_master[3][1] + T_master[3][2] + T_master[3][3] + T_master[3][4] + T_master[3][5] + T_master[3][6] + T_master[4][1] + T_master[4][2] + T_master[4][3] + T_master[4][4] + T_master[4][5] + T_master[4][6] + T_master[5][1] + T_master[5][2] + T_master[5][3] + T_master[5][4] + T_master[5][5] + T_master[5][6] + T_master[6][1] + T_master[6][2] + T_master[6][3] + T_master[6][4] + T_master[6][5] + T_master[6][6])
    noise_out[d1][d2] = np.std(T_master[1][1] + T_master[1][2] + T_master[1][3] + T_master[1][4] + T_master[1][5] + T_master[1][6] + T_master[2][1] + T_master[2][2] + T_master[2][3] + T_master[2][4] + T_master[2][5] + T_master[2][6] + T_master[3][1] +  T_master[3][2] + T_master[3][3] + T_master[3][4] + T_master[3][5] + T_master[3][6] + T_master[4][1] + T_master[4][2] + T_master[4][3] + T_master[4][4] + T_master[4][5] + T_master[4][6] + T_master[5][1] + T_master[5][2] + T_master[5][3] + T_master[5][4] + T_master[5][5] + T_master[5][6] + T_master[6][1] + T_master[6][2] + T_master[6][3] + T_master[6][4] + T_master[6][5] + T_master[6][6])

    return signal_out, noise_out

def chunk2(matrix_temp, T_master):
    import numpy as np

    rank_out = 3
    
    signal_out = np.zeros(shape=(rank_out, rank_out))
    noise_out = np.zeros(shape=(rank_out, rank_out))
    SNR_out = np.zeros(shape=(rank_out, rank_out))
    aggregate_noise = np.zeros(shape=(rank_out, rank_out))

        
    # there are ___ sections of this chunk.
    # part 1)
    d1 = 0
    d2 = 0
    signal_out[d1][d2] = np.mean(T_master[d1][d2])
    noise_out[d1][d2] = np.std(T_master[d1][d2])
    if noise_out[d1][d2] != 0:
        SNR_out[d1][d2] = (signal_out[d1][d2])/(noise_out[d1][d2])
    d1 = 0
    d2 = 1
    signal_out[d1][d2] = np.mean(T_master[d1][d2])
    noise_out[d1][d2] = np.std(T_master[d1][d2])
    if noise_out[d1][d2] != 0:
        SNR_out[d1][d2] = (signal_out[d1][d2])/(noise_out[d1][d2])
    d1 = 1
    d2 = 0
    signal_out[d1][d2] = np.mean(T_master[d1][d2])
    noise_out[d1][d2] = np.std(T_master[d1][d2])
    if noise_out[d1][d2] != 0:
        SNR_out[d1][d2] = (signal_out[d1][d2])/(noise_out[d1][d2])
    d1 = 1
    d2 = 1
    signal_out[d1][d2] = np.mean(T_master[d1][d2])
    noise_out[d1][d2] = np.std(T_master[d1][d2])
    if noise_out[d1][d2] != 0:
        SNR_out[d1][d2] = (signal_out[d1][d2])/(noise_out[d1][d2])
    # part 2)
    d1 = 0
    d2 = 2
    signal_out[d1][d2] = np.mean(T_master[d1][2] + T_master[d1][3] + T_master[d1][4] + T_master[d1][5] + T_master[d1][6])
    noise_out[d1][d2] = np.std(T_master[d1][2] + T_master[d1][3] + T_master[d1][4] + T_master[d1][5] + T_master[d1][6])
    if noise_out[d1][d2] != 0:
        SNR_out[d1][d2] = (signal_out[d1][d2])/(noise_out[d1][d2])
    d1 = 1
    d2 = 2
    signal_out[d1][d2] = np.mean(T_master[d1][2] + T_master[d1][3] + T_master[d1][4] + T_master[d1][5] + T_master[d1][6])
    noise_out[d1][d2] = np.std(T_master[d1][2] + T_master[d1][3] + T_master[d1][4] + T_master[d1][5] + T_master[d1][6])
    if noise_out[d1][d2] != 0:
        SNR_out[d1][d2] = (signal_out[d1][d2])/(noise_out[d1][d2])
    # part 3)
    d1 = 2
    d2 = 0
    signal_out[d1][d2] = np.mean(T_master[2][d2] + T_master[3][d2] + T_master[4][d2] + T_master[5][d2] + T_master[6][d2])
    noise_out[d1][d2] = np.std(T_master[2][d2] + T_master[3][d2] + T_master[4][d2] + T_master[5][d2] + T_master[6][d2])
    if noise_out[d1][d2] != 0:
        SNR_out[d1][d2] = (signal_out[d1][d2])/(noise_out[d1][d2])
    d1 = 2
    d2 = 1
    signal_out[d1][d2] = np.mean(T_master[2][d2] + T_master[3][d2] + T_master[4][d2] + T_master[5][d2] + T_master[6][d2])
    noise_out[d1][d2] = np.std(T_master[2][d2] + T_master[3][d2] + T_master[4][d2] + T_master[5][d2] + T_master[6][d2])
    if noise_out[d1][d2] != 0:
        SNR_out[d1][d2] = (signal_out[d1][d2])/(noise_out[d1][d2])
    # part 4)
    d1 = 2
    d2 = 2
    signal_out[d1][d2] = np.mean(T_master[2][2] + T_master[2][3] + T_master[2][4] + T_master[2][5] + T_master[2][6] + T_master[3][2] + T_master[3][3] + T_master[3][4] + T_master[3][5] + T_master[3][6] + T_master[4][2] + T_master[4][3] + T_master[4][4] + T_master[4][5] + T_master[4][6] + T_master[5][2] + T_master[5][3] + T_master[5][4] + T_master[5][5] + T_master[5][6] + T_master[6][2] + T_master[6][3] + T_master[6][4] + T_master[6][5] + T_master[6][6])
    noise_out[d1][d2] = np.std(T_master[2][2] + T_master[2][3] + T_master[2][4] + T_master[2][5] + T_master[2][6] + T_master[3][2] + T_master[3][3] + T_master[3][4] + T_master[3][5] + T_master[3][6] + T_master[4][2] + T_master[4][3] + T_master[4][4] + T_master[4][5] + T_master[4][6] + T_master[5][2] + T_master[5][3] + T_master[5][4] + T_master[5][5] + T_master[5][6] + T_master[6][2] + T_master[6][3] + T_master[6][4] + T_master[6][5] + T_master[6][6])
    if noise_out[d1][d2] != 0:
        SNR_out[d1][d2] = (signal_out[d1][d2])/(noise_out[d1][d2])

    SNR_meanOfAggregate = np.zeros(shape=(rank_out, 1))
    for dex_1 in range(0, rank_out):
          for dex_2 in range(0, dex_1):
              fileout.write(dex_1, dex_2)
              SNR_meanOfAggregate[dex_1] = SNR_meanOfAggregate[dex_1] + SNR_out[dex_1][dex_2]
          SNR_meanOfAggregate[dex_1] = SNR_meanOfAggregate[dex_1] + SNR_out[dex_1][dex_1]
          for dex_2 in range(0, dex_1):
              SNR_meanOfAggregate[dex_1] = SNR_meanOfAggregate[dex_1] + SNR_out[dex_2][dex_1]
          SNR_meanOfAggregate[dex_1] = SNR_meanOfAggregate[dex_1]/(2*dex_1+1)

    return signal_out, noise_out

def chunk1(matrix_temp, T_master):
    import numpy as np

    rank_out = np.linalg.matrix_rank(matrix_temp)
    
    signal_out = np.zeros(shape=(rank_out, rank_out))
    noise_out = np.zeros(shape=(rank_out, rank_out))
        
    # there are ___ sections of this chunk.
    # part 1)
    d1 = 0
    d2 = 0
    signal_out[d1][d2] = np.mean(T_master[d1][d2])
    noise_out[d1][d2] = np.std(T_master[d1][d2])
    d1 = 0
    d2 = 1
    signal_out[d1][d2] = np.mean(T_master[d1][d2])
    noise_out[d1][d2] = np.std(T_master[d1][d2])
    d1 = 0
    d2 = 3
    signal_out[d1][d2] = np.mean(T_master[d1][6])
    noise_out[d1][d2] = np.std(T_master[d1][6])
    d1 = 1
    d2 = 0
    signal_out[d1][d2] = np.mean(T_master[d1][d2])
    noise_out[d1][d2] = np.std(T_master[d1][d2])
    d1 = 1
    d2 = 1
    signal_out[d1][d2] = np.mean(T_master[d1][d2])
    noise_out[d1][d2] = np.std(T_master[d1][d2])
    d1 = 1
    d2 = 3
    signal_out[d1][d2] = np.mean(T_master[d1][6])
    noise_out[d1][d2] = np.std(T_master[d1][6])
    d1 = 3
    d2 = 0
    signal_out[d1][d2] = np.mean(T_master[6][d2])
    noise_out[d1][d2] = np.std(T_master[6][d2])
    d1 = 3
    d2 = 1
    signal_out[d1][d2] = np.mean(T_master[6][d2])
    noise_out[d1][d2] = np.std(T_master[6][d2])
    d1 = 3
    d2 = 3
    signal_out[d1][d2] = np.mean(T_master[6][6])
    noise_out[d1][d2] = np.std(T_master[6][6])
    # part 2)
    d1 = 0
    d2 = 2
    signal_out[d1][d2] = np.mean(T_master[d1][2] + T_master[d1][3] + T_master[d1][4] + T_master[d1][5])
    noise_out[d1][d2] = np.std(T_master[d1][2] + T_master[d1][3] + T_master[d1][4] + T_master[d1][5])
    d1 = 1
    d2 = 2
    signal_out[d1][d2] = np.mean(T_master[d1][2] + T_master[d1][3] + T_master[d1][4] + T_master[d1][5])
    noise_out[d1][d2] = np.std(T_master[d1][2] + T_master[d1][3] + T_master[d1][4] + T_master[d1][5])
    d1 = 3
    d2 = 2
    signal_out[d1][d2] = np.mean(T_master[6][2] + T_master[6][3] + T_master[6][4] + T_master[6][5])
    noise_out[d1][d2] = np.std(T_master[6][2] + T_master[6][3] + T_master[6][4] + T_master[6][5])
    # part 3)
    d1 = 2
    d2 = 0
    signal_out[d1][d2] = np.mean(T_master[2][d2] + T_master[3][d2] + T_master[4][d2] + T_master[5][d2])
    noise_out[d1][d2] = np.std(T_master[2][d2] + T_master[3][d2] + T_master[4][d2] + T_master[5][d2])
    d1 = 2
    d2 = 1
    signal_out[d1][d2] = np.mean(T_master[2][d2] + T_master[3][d2] + T_master[4][d2] + T_master[5][d2])
    noise_out[d1][d2] = np.std(T_master[2][d2] + T_master[3][d2] + T_master[4][d2] + T_master[5][d2])
    d1 = 2
    d2 = 3
    signal_out[d1][d2] = np.mean(T_master[2][6] + T_master[3][6] + T_master[4][6] + T_master[5][6])
    noise_out[d1][d2] = np.std(T_master[2][6] + T_master[3][6] + T_master[4][6] + T_master[5][6])
    # part 4)
    d1 = 2
    d2 = 2
    signal_out[d1][d2] = np.mean(T_master[2][2] + T_master[2][3] + T_master[2][4] + T_master[2][5] + T_master[3][2] + T_master[3][3] + T_master[3][4] + T_master[3][5] + T_master[4][2] + T_master[4][3] + T_master[4][4] + T_master[4][5] + T_master[5][2] + T_master[5][3] + T_master[5][4] + T_master[5][5])
    noise_out[d1][d2] = np.std(T_master[2][2] + T_master[2][3] + T_master[2][4] + T_master[2][5] + T_master[3][2] + T_master[3][3] + T_master[3][4] + T_master[3][5] + T_master[4][2] + T_master[4][3] + T_master[4][4] + T_master[4][5] + T_master[5][2] + T_master[5][3] + T_master[5][4] + T_master[5][5])
    
    return signal_out, noise_out

def make_transition_matrix(NUMBER_STABLE_STATES, HISTORICAL_RECORD, index_election):
    index_record_i = 8*(index_election - 1)
    index_record_f = 8*(index_election)
    fileout.write('  ')
    fileout.write('=====================================================================')
    fileout.write("%s%d%s%d%s%d" % ('Election Cycle: ', index_election, '; ', HISTORICAL_RECORD[index_record_i][0], ' -> ', HISTORICAL_RECORD[index_record_f][0]))#
    fileout.write('  ')
    transition_matrix = buildTransitionMatrixFromHistoricalData(HISTORICAL_RECORD, index_record_i, index_record_f)
    return transition_matrix


def buildTransitionMatrixFromHistoricalData(HISTORICAL_RECORD, index_record_i, index_record_f):
    # here there should really be two functions:
    # (1) build the primary matrices that we need (which number three)
    # (2) processing those matrices correctly into the three sub transition matrices
    # (3) adding those three sub transistion matrices
    import numpy as np

    percent_vote_i, percent_vote_f, do_transition_if_generation, do_transition_if_continuation, do_transition_if_annihalation = buildFundamentalMatrices(HISTORICAL_RECORD, index_record_i, index_record_f)
    if 0 == 1:
        fileout.write(' ')
        fileout.write('Report from Build Transition Matrix From Historical Data:')
        fileout.write('index_i = {}; index_f = {}'.format(index_record_i, index_record_f))

    if (1 == 1):
        fileout.write('     percent_vote_i = ')
        fileout.write(str(np.matrix.round(percent_vote_i*1000)/1000))
        fileout.write('     percent_vote_f = ')
        fileout.write(str(np.matrix.round(percent_vote_f*1000)/1000))
        fileout.write('     do_transition_if generation = ')
        fileout.write(str(np.matrix.round(do_transition_if_generation*1000)/1000))
        fileout.write('     do_transition_if continuation = ')
        fileout.write(str(np.matrix.round(do_transition_if_continuation*1000)/1000))
        fileout.write('     do_transition_if annihalation = ')
        fileout.write(str(np.matrix.round(do_transition_if_annihalation*1000)/1000))

    transition_matrix = buildOneSubTransitionMatrices(percent_vote_i, percent_vote_f, do_transition_if_generation, do_transition_if_continuation, do_transition_if_annihalation)
    fileout.write('     transition_matrix =')
    fileout.write(str(np.matrix.round(transition_matrix*1000)/1000))
    
    return transition_matrix
     
    NUMBER_STABLE_STATES = 7
    # before we continue, we need ro ascertain the correct renormalization based on generation possibilities
    transition_matrix = (transition_matrix_generation + transition_matrix_continuation + transition_matrix_annihalation)
    fileout.write('transition_matrix not necessarily normalized =')
    fileout.write(str(np.matrix.round(transition_matrix*1000)/1000))
    product_renormalization_generation = 1/(sum_prob_generation[6] + 1)
    transition_matrix = product_renormalization_generation*(transition_matrix_generation + transition_matrix_continuation + transition_matrix_annihalation)
    fileout.write('transition_matrix yes necessarily normalized =')
    fileout.write(str(np.matrix.round(transition_matrix*1000)/1000))

    sum_transition_matrix_annihalation = sum(sum(transition_matrix_annihalation))
    fileout.write('sum sum transition_matrix_annihalation =')
    fileout.write(str(sum_transition_matrix_annihalation))
    fileout.write('transition_matrix_annihalation =')
    fileout.write(str(np.matrix.round(transition_matrix_annihalation*1000)/1000))
    sum_transition_matrix_annihalation = sum(sum(transition_matrix_annihalation))
    fileout.write('sum sum transition_matrix_annihalation =')
    fileout.write(str(sum_transition_matrix_annihalation))
    fileout.write('transition_matrix_annihalation =')
    fileout.write(str(np.matrix.round(transition_matrix_annihalation*1000)/1000))
    sum_transition_matrix_annihalation = sum(sum(transition_matrix_annihalation))
    fileout.write('sum sum transition_matrix_annihalation =')
    fileout.write(str(sum_transition_matrix_annihalation))




def buildFundamentalMatrices(HISTORICAL_RECORD, index_record_i, index_record_f):
    import numpy as np
    import pdb

    NUMBER_STABLE_STATES = 7

    percent_vote_i = np.zeros(shape=(NUMBER_STABLE_STATES,1))
    percent_vote_f = np.zeros(shape=(1,NUMBER_STABLE_STATES))
    do_transition_if_generation = np.zeros(shape=(NUMBER_STABLE_STATES,NUMBER_STABLE_STATES))
    do_transition_if_continuation = np.zeros(shape=(NUMBER_STABLE_STATES,NUMBER_STABLE_STATES))
    do_transition_if_annihalation = np.zeros(shape=(NUMBER_STABLE_STATES,NUMBER_STABLE_STATES))

    for pointer_i in range(1, NUMBER_STABLE_STATES+1):
        percent_vote_i[pointer_i-1,0] = np.sum(HISTORICAL_RECORD[index_record_i + pointer_i][3])/HISTORICAL_RECORD[index_record_i][3]

    for pointer_f in range(1, NUMBER_STABLE_STATES+1):
        percent_vote_f[0,pointer_f-1] = np.sum(HISTORICAL_RECORD[index_record_f + pointer_f][3])/HISTORICAL_RECORD[index_record_f][3]
        #A = HISTORICAL_RECORD[index_record_f + pointer_f][3]
        #degeneracy_f = len(HISTORICAL_RECORD[index_record_f + pointer_f][3])
        #strs = ["Hi","Bye"] 
        #bools = [ True for s in strs ]
        #if len(HISTORICAL_RECORD[index_record_f + pointer_f][3]) == 1:
        #    percent_vote_f[0,pointer_f-1] = 
        #else:
        #    percent_vote_f[0,pointer_f-1] = sum(HISTORICAL_RECORD[index_record_f + pointer_f][3])/HISTORICAL_RECORD[index_record_f][3]
        #    fileout.write('mnm')
        #    fileout.write(np.matrix.round(percent_vote_f*1000)/1000)

    for pointer_i in range(1, NUMBER_STABLE_STATES+1):
        for pointer_f in range(1, NUMBER_STABLE_STATES+1):
            if (1 == 1):
                if 1 == 0:
                    fileout.write('    ')
                    fileout.write('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
                    fileout.write('Report from Build Fundamental Matrices; loop (_,_) of (_,_)')
                    fileout.write('Test: ({} -> {}): ({} > {}); ({},{}) > ({},{})'.format(HISTORICAL_RECORD[index_record_i + pointer_i][0],HISTORICAL_RECORD[index_record_f + pointer_f][0],pointer_i,pointer_f, HISTORICAL_RECORD[index_record_i + pointer_i][1], HISTORICAL_RECORD[index_record_i + pointer_i][2], HISTORICAL_RECORD[index_record_f + pointer_f][1], HISTORICAL_RECORD[index_record_f + pointer_f][2]))

            degeneracy_i = len(HISTORICAL_RECORD[index_record_i + pointer_i][3])
            degeneracy_f = len(HISTORICAL_RECORD[index_record_f + pointer_f][3])
            
            # GENERERATION CONDITIONS begin here
            # condition_xy_g1: is the name of the initial person we are looking at right now equivalent to None?
            # condition_xy_g2: is the name of the final person that we are considering now equivalent to the name of any of the initial people??
            # condition_xy_g3: is the final persons party affiliation(s) equal to any of those other people's party officialations?
            condition_if_g1 = (pointer_i == 7)#  this means that i am now looking at an initial state in the void
            condition_if_g2 = ('Yes' == 'Yes');
            for jarbon in range(1, NUMBER_STABLE_STATES+1):
                #if HISTORICAL_RECORD[index_record_f + pointer_f][1] == 'Andrew Jackson' and HISTORICAL_RECORD[index_record_i + pointer_i][1] == 'Andrew Jackson':
                 #   pdb.set_trace()
                if (HISTORICAL_RECORD[index_record_f + pointer_f][1] == HISTORICAL_RECORD[index_record_i + jarbon][1]): # is the name of the final person equivalent to the name of any of the initial people?
                    condition_if_g2 = ('Yes' == 'No'); # you are not now, nor at any time in the future, going to be generated
                    
                
            condition_if_g3 = ('Yes' == 'Yes')
            for dex_deg_f in range(0, degeneracy_f):
                for jarbon in range(1, NUMBER_STABLE_STATES+1):
                    #fileout.write('jarbon = ', jarbon)
                    degeneracy_jarbon = len(HISTORICAL_RECORD[index_record_i + jarbon][3])
                    for dex_deg_jarbon in range(0, degeneracy_jarbon):
                        if 0 == 1:
                            fileout.write(index_record_f, pointer_f, dex_deg_f)
                            fileout.write(HISTORICAL_RECORD[index_record_f + pointer_f][2][dex_deg_f])
                            fileout.write(index_record_i, jarbon, dex_deg_jarbon)
                            fileout.write(HISTORICAL_RECORD[index_record_i + jarbon][2][dex_deg_jarbon])
                        if (HISTORICAL_RECORD[index_record_f + pointer_f][2][dex_deg_f] == HISTORICAL_RECORD[index_record_i + jarbon][2][dex_deg_jarbon]): # is the name of the final party that we are considering now equivalent to the name of any of the initial parties??:
                            condition_if_g3 = ('Yes' == 'No')# you are not now, nor at any time in the future, going to be generated
            #fileout.write('G CONDITIONS: >>', condition_if_g1, condition_if_g2, condition_if_g3)
            if (condition_if_g1 and (condition_if_g2 and condition_if_g3)):
                do_transition_if_generation[pointer_i-1,pointer_f-1] = 1
                fileout.write('Generation: (',HISTORICAL_RECORD[index_record_i + pointer_i][0],' ->',HISTORICAL_RECORD[index_record_f + pointer_f][0],'): (',pointer_i,'>',pointer_f,')', HISTORICAL_RECORD[index_record_f + pointer_f][1], '-s', HISTORICAL_RECORD[index_record_f + pointer_f][2], ' Party, is (re)-inaugerated into the historical record.')
                fileout.write(pointer_i, pointer_f)
                # 
            else:
                do_transition_if_generation[pointer_i-1,pointer_f-1] = 0
                # fileout the info regarding this event

            number_continuation_events = 0  # default value
            pointer_if_continuation = [0, 0] # default value
            # CONTINUATION CONDITIONS begin here
            # condition_if_c1: is the name of the initial person being considered equivalent to the name of the final person being considered?
            # condition_if_c2: is the name of the initial party being considered equivalent to the name of the final party being considered?
            # condition_if_c3
            condition_if_c1 = (HISTORICAL_RECORD[index_record_i + pointer_i][1] == HISTORICAL_RECORD[index_record_f + pointer_f][1]) # check by name
            condition_if_c2 = ('Yes' == 'No')# you are not now, and at any time in the future, going to be continued
            for dex_deg_i in range(0, degeneracy_i):
                for dex_deg_f in range(0, degeneracy_f):
                    ##fileout.write(dex_deg_i, dex_deg_f)
                    ##fileout.write(HISTORICAL_RECORD[index_record_i + pointer_i][2][dex_deg_i], HISTORICAL_RECORD[index_record_f + pointer_f][2][dex_deg_f])
                    if HISTORICAL_RECORD[index_record_i + pointer_i][2][dex_deg_i] == HISTORICAL_RECORD[index_record_f + pointer_f][2][dex_deg_f]:
                        number_continuation_events = number_continuation_events + 1
                        pointer_if_continuation = [dex_deg_i, dex_deg_f]
                        condition_if_c2 = ('Yes' == 'Yes')# you are not now, and at any time in the future, going to be continued
            if number_continuation_events > 1:
                fileout.write('Danger; Alert')
            if not(((pointer_i == 7) and not(pointer_f == 7)) or (not(pointer_i == 7) and (pointer_f == 7))):
                condition_if_c3 = (1 == 1)
            else:
                condition_if_c3 = (0 == 1)
            #fileout.write('CONTINUE CONDITIONS: >>', condition_if_c1, condition_if_c2, condition_if_c3)
            if ((condition_if_c1 or condition_if_c2) and condition_if_c3):
                if ((HISTORICAL_RECORD[index_record_i + pointer_i][1] != 'None') and (HISTORICAL_RECORD[index_record_f + pointer_f][1] != 'None')):
                    do_transition_if_continuation[pointer_i-1,pointer_f-1] = 1
                    fileout.write('Continuation: ( %d > %d): (%d > %d); (%d,%d) > (%d,%d)'.format(HISTORICAL_RECORD[index_record_i + pointer_i][0],HISTORICAL_RECORD[index_record_f + pointer_f][0],pointer_i,pointer_f, HISTORICAL_RECORD[index_record_i + pointer_i][1], HISTORICAL_RECORD[index_record_i + pointer_i][2][pointer_if_continuation[0]], HISTORICAL_RECORD[index_record_f + pointer_f][1], HISTORICAL_RECORD[index_record_f + pointer_f][2][pointer_if_continuation[1]]))
                    # fileout.write('Continuation: (',HISTORICAL_RECORD[index_record_i + pointer_i][0],'>',HISTORICAL_RECORD[index_record_f + pointer_f][0],'): (',pointer_i,'>',pointer_f,'); (', HISTORICAL_RECORD[index_record_i + pointer_i][1], ',', HISTORICAL_RECORD[index_record_i + pointer_i][2][pointer_if_continuation[0]], ') > (', HISTORICAL_RECORD[index_record_f + pointer_f][1], ',', HISTORICAL_RECORD[index_record_f + pointer_f][2][pointer_if_continuation[1]], ')')
                    fileout.write(str([pointer_i, pointer_f]))
            else:
                do_transition_if_continuation[pointer_i-1,pointer_f-1] = 0
            
            number_annihalation_events = 0  # default value
            pointer_if_annihalation = [[0, 7]]*degeneracy_i # default value
            # ANNIHALATION CONDITIONS begin here
            # condition_xy_a1: is the name of the final person we are looking at right now equivalent to 'None'
            # condition_xy_a2: is the name of the initial person we are looking at right now equivalent to the name of the final person 
            # condition_xy_a3: is the (i) initial person's party affiliation(s) equivalent to final person's party affiliation(s)??
            condition_if_a1 = (pointer_f == NUMBER_STABLE_STATES)#  this means that i am now looking at a final state in the void
            condition_if_a2 = ('Yes' == 'Yes');
            for jarbon in range(1, NUMBER_STABLE_STATES+1):
                if (HISTORICAL_RECORD[index_record_i + pointer_i][1] == HISTORICAL_RECORD[index_record_f + jarbon][1]): # is the name of the initial party equivalent to the name of any of the final parties?? 
                    condition_if_a2 = ('Yes' == 'No'); # you are not now, nor at any time in the future, going to be anhialated
            condition_if_a3 = ['Yes' == 'Yes']*degeneracy_i
            for deg_dex_i in range(0, degeneracy_i):
                for jarbon in range(1, NUMBER_STABLE_STATES + 1):
                    degeneracy_jarbon = len(HISTORICAL_RECORD[index_record_f + jarbon][3])
                    for deg_dex_f in range(0, degeneracy_jarbon):
                        if 1 == 0:
                            fileout.write('jarbon', jarbon, deg_dexy_i, deg_dexy_f)
                            fileout.write(HISTORICAL_RECORD[index_record_f + pointer_f])
                            fileout.write(HISTORICAL_RECORD[index_record_f + pointer_f][2][deg_dexy_f])
                            fileout.write(HISTORICAL_RECORD[index_record_i])
                            fileout.write(HISTORICAL_RECORD[index_record_i + jarbon])
                            fileout.write(HISTORICAL_RECORD[index_record_i + jarbon][2][deg_dexy_i])
                        if (HISTORICAL_RECORD[index_record_i + pointer_i][2][deg_dex_i] == HISTORICAL_RECORD[index_record_f + jarbon][2][deg_dex_f]): # is the name of the final party that we are considering now equivalent to the name of any of the initial parties??
                            condition_if_a3[deg_dex_i] = ('Yes' == 'No') # you are not now, nor at any time in the future, going to be anhilated
                # now check it
                if condition_if_a3[deg_dex_i]:
                    number_annihalation_events = number_annihalation_events + 1
                    if number_annihalation_events > 1:
                        fileout.write('how are you going to solve this???')
                        return
                    pointer_if_annihalation[deg_dex_i] = [deg_dex_i, 0]
            for deg_dex_i in range(0, degeneracy_i):
                #fileout.write('ANHIL CONDITIONS:', condition_if_a1, condition_if_a2, condition_if_a3[deg_dex_i], 'only (1 & 2 & 3) will trigger annihalation')
                if (condition_if_a1 and (condition_if_a2 and condition_if_a3[deg_dex_i])):
                    do_transition_if_annihalation[pointer_i-1,pointer_f-1] = 1
                    # fileout the info regarding this event
                    fileout.write('Annihalation: (',HISTORICAL_RECORD[index_record_i + pointer_i][0],'>',HISTORICAL_RECORD[index_record_f + pointer_f][0],'): (',pointer_i,'>',pointer_f,'); (', HISTORICAL_RECORD[index_record_i + pointer_i][1], ',', HISTORICAL_RECORD[index_record_i + pointer_i][2][deg_dex_i], ') > (', HISTORICAL_RECORD[index_record_f + pointer_f][1], ',', HISTORICAL_RECORD[index_record_f + pointer_f][2], ')')
                    fileout.write(pointer_i, pointer_f)
                else:
                    do_transition_if_annihalation[pointer_i-1,pointer_f-1] = 0 

    return percent_vote_i, percent_vote_f, do_transition_if_generation, do_transition_if_continuation, do_transition_if_annihalation

def buildOneSubTransitionMatrices(percent_vote_i, percent_vote_f, do_transition_if_generation, do_transition_if_continuation, do_transition_if_annihalation):
    import numpy as np


    DO_DISPLAY_DATA = 0
    NUMBER_STABLE_STATES = 7

    diag_percent_vote_i = np.zeros(shape=(NUMBER_STABLE_STATES,NUMBER_STABLE_STATES))
    diag_percent_vote_f = np.zeros(shape=(NUMBER_STABLE_STATES,NUMBER_STABLE_STATES))
    diag_reciprocal_sum_prob = np.zeros(shape=(NUMBER_STABLE_STATES,NUMBER_STABLE_STATES))

    do_transition_if = do_transition_if_generation + do_transition_if_continuation + do_transition_if_annihalation

    if DO_DISPLAY_DATA:
        fileout.write(' ')
        fileout.write('DO_DISPLAY_DATA BEGINS')
        fileout.write('INiTIAL percent_vote_i = ')
        fileout.write(np.matrix.round(percent_vote_i*1000)/1000)
        
    sum_prob = np.dot(do_transition_if, np.transpose(percent_vote_f))
    for dex in range(0, 7):
        if dex == 6:
            percent_vote_i[dex] = sum_prob[6];
        else:
            percent_vote_i[dex] = (1 - sum_prob[6])*percent_vote_i[dex];
    
    percent_vote_f[0,6] = 1
    sum_prob = np.dot(do_transition_if, np.transpose(percent_vote_f))

    for electoral_tally_pointer in range(1, NUMBER_STABLE_STATES+1):
        diag_percent_vote_i[electoral_tally_pointer - 1, electoral_tally_pointer - 1] = percent_vote_i[electoral_tally_pointer - 1, 0]
        diag_percent_vote_f[electoral_tally_pointer - 1, electoral_tally_pointer - 1] = percent_vote_f[0, electoral_tally_pointer - 1]
        if sum_prob[electoral_tally_pointer - 1, 0] == 0:
            diag_reciprocal_sum_prob[electoral_tally_pointer - 1, electoral_tally_pointer - 1] = 0
        else:
            diag_reciprocal_sum_prob[electoral_tally_pointer - 1, electoral_tally_pointer - 1] = 1/sum_prob[electoral_tally_pointer - 1, 0]
    product_i = np.dot(diag_percent_vote_i, do_transition_if)
    product_f = np.dot(do_transition_if, diag_percent_vote_f)
    product_reciprocal_sum = np.dot(diag_reciprocal_sum_prob, do_transition_if)
    transition_matrix = (product_i*product_f)*product_reciprocal_sum
    sum_transition_matrix = sum(sum(transition_matrix))

    if DO_DISPLAY_DATA:
        fileout.write('FINAL percent_vote_i = ')
        fileout.write(np.matrix.round(percent_vote_i*1000)/1000)
        fileout.write('percent_vote_f = ')
        fileout.write(np.matrix.round(percent_vote_f*1000)/1000)
        fileout.write('do_transition_if = ')
        fileout.write(np.matrix.round(do_transition_if*1000)/1000)
        
        fileout.write('sum_prob')
        fileout.write(np.matrix.round(sum_prob*1000)/1000)
        fileout.write('diag_percent_vote_i = ')
        fileout.write(np.matrix.round(diag_percent_vote_i*1000)/1000)
        fileout.write('diag_percent_vote_f = ')
        fileout.write(np.matrix.round(diag_percent_vote_f*1000)/1000)
        fileout.write('diag_reciprocal_sum_prob = ')
        fileout.write(np.matrix.round(diag_reciprocal_sum_prob*1000)/1000)
        fileout.write('product_i =')
        fileout.write(np.matrix.round(product_i*1000)/1000)
        fileout.write('product_f =')
        fileout.write(np.matrix.round(product_f*1000)/1000)
        fileout.write('product_reciprocal_sum =')
        fileout.write(np.matrix.round(product_reciprocal_sum*1000)/1000)
        fileout.write('transition_matrix=')
        fileout.write(np.matrix.round(transition_matrix*1000)/1000)
        fileout.write('sum sum transition_matrix =')
        fileout.write(sum_transition_matrix)

    return transition_matrix



def buildThreeSubTransitionMatrices(percent_vote_i, percent_vote_f, do_transition_if_generation, do_transition_if_continuation, do_transition_if_annihalation):
    import numpy as np

    NUMBER_STABLE_STATES = 7

    diag_percent_vote_i = np.zeros(shape=(NUMBER_STABLE_STATES,NUMBER_STABLE_STATES))
    diag_percent_vote_f = np.zeros(shape=(NUMBER_STABLE_STATES,NUMBER_STABLE_STATES))
    diag_reciprocal_sum_prob_generation = np.zeros(shape=(NUMBER_STABLE_STATES,NUMBER_STABLE_STATES))
    diag_reciprocal_sum_prob_continuation = np.zeros(shape=(NUMBER_STABLE_STATES,NUMBER_STABLE_STATES))
    diag_reciprocal_sum_prob_annihalation = np.zeros(shape=(NUMBER_STABLE_STATES,NUMBER_STABLE_STATES))

    #fileout.write('do_transition_if generation = ')
    #fileout.write(np.matrix.round(do_transition_if_generation*1000)/1000)
    sum_prob_generation = np.dot(do_transition_if_generation, np.transpose(percent_vote_f))
    #fileout.write('[sum_i{prob(1>i)}, sum_i{prob(2>i)}, ..., sum_i{prob(7>i)}')
    #fileout.write(sum_prob_generation)
    for electoral_tally_pointer in range(1, NUMBER_STABLE_STATES+1):
        diag_percent_vote_i[electoral_tally_pointer - 1, electoral_tally_pointer - 1] = percent_vote_i[electoral_tally_pointer - 1, 0]
        diag_percent_vote_f[electoral_tally_pointer - 1, electoral_tally_pointer - 1] = percent_vote_f[0, electoral_tally_pointer - 1]
        if sum_prob_generation[electoral_tally_pointer - 1, 0] == 0:
            diag_reciprocal_sum_prob_generation[electoral_tally_pointer - 1, electoral_tally_pointer - 1] = 0
        else:
            diag_reciprocal_sum_prob_generation[electoral_tally_pointer - 1, electoral_tally_pointer - 1] = 1/sum_prob_generation[electoral_tally_pointer - 1, 0]
    product_i_generation = np.dot(diag_percent_vote_i, do_transition_if_generation)
    product_f_generation = np.dot(do_transition_if_generation, diag_percent_vote_f)
    product_reciprocal_sum_generation = np.dot(diag_reciprocal_sum_prob_generation, do_transition_if_generation)
    if 0 == 1:
        fileout.write('generation factor 1 generation =')
        fileout.write(np.matrix.round(product_i_generation*1000)/1000)
        fileout.write('generation factor 2 generation =')
        fileout.write(np.matrix.round(product_f_generation*1000)/1000)
        fileout.write('generation  factor 3 generation =')
        fileout.write(np.matrix.round(product_reciprocal_sum_generation*1000)/1000)
    #fileout.write('transition_matrix_generation =')
    transition_matrix_generation = do_transition_if_generation*product_f_generation
    #fileout.write(np.matrix.round(transition_matrix_generation*1000)/1000)
    sum_transition_matrix_generation = sum(sum(transition_matrix_generation))
    #fileout.write('sum sum transition_matrix_generation =')
    #fileout.write(sum_transition_matrix_generation)
       
    #fileout.write('do_transition_if continuation = ')
    #fileout.write(np.matrix.round(do_transition_if_continuation*1000)/1000)
    sum_prob_continuation = np.dot(do_transition_if_continuation, np.transpose(percent_vote_f))
    #fileout.write('[sum_i{prob(1>i)}, sum_i{prob(2>i)}, ..., sum_i{prob(7>i)}')
    #fileout.write(sum_prob_continuation)
    for electoral_tally_pointer in range(1, NUMBER_STABLE_STATES):
        diag_percent_vote_i[electoral_tally_pointer - 1, electoral_tally_pointer - 1] = percent_vote_i[electoral_tally_pointer - 1, 0]
        diag_percent_vote_f[electoral_tally_pointer - 1, electoral_tally_pointer - 1] = percent_vote_f[0, electoral_tally_pointer - 1]
        if sum_prob_continuation[electoral_tally_pointer - 1, 0] == 0:
            diag_reciprocal_sum_prob_continuation[electoral_tally_pointer - 1, electoral_tally_pointer - 1] = 0
        else:
            diag_reciprocal_sum_prob_continuation[electoral_tally_pointer - 1, electoral_tally_pointer - 1] = 1/sum_prob_continuation[electoral_tally_pointer - 1, 0]
    product_i_continuation = np.dot(diag_percent_vote_i, do_transition_if_continuation)
    product_f_continuation = np.dot(do_transition_if_continuation, diag_percent_vote_f)
    product_reciprocal_sum_continuation = np.dot(diag_reciprocal_sum_prob_continuation, do_transition_if_continuation)
    if 0 == 1:
        fileout.write('continuation factor 1 =')
        fileout.write(np.matrix.round(product_i_continuation*1000)/1000)
        fileout.write('continuation factor 2 =')
        fileout.write(np.matrix.round(product_f_continuation*1000)/1000)
        fileout.write('continuation factor 3 =')
        fileout.write(np.matrix.round(product_reciprocal_sum_continuation*1000)/1000)
    transition_matrix_continuation = (product_i_continuation*product_f_continuation)*product_reciprocal_sum_continuation
    #fileout.write(' transition_matrix_continuation =')
    #fileout.write(np.matrix.round(transition_matrix_continuation*1000)/1000)
    sum_transition_matrix_continuation = sum(sum(transition_matrix_continuation))
    #fileout.write('sum sum transition_matrix_continuation =')
    #fileout.write(sum_transition_matrix_continuation)
    
    #fileout.write('do_transition_if annihalation = ')
    #fileout.write(np.matrix.round(do_transition_if_annihalation*1000)/1000)
    sum_prob_annihalation = np.dot(do_transition_if_annihalation, np.transpose(percent_vote_f))
    #fileout.write('[sum_i{prob(1>i)}, sum_i{prob(2>i)}, ..., sum_i{prob(7>i)}')
    #fileout.write(sum_prob_annihalation)
    for electoral_tally_pointer in range(1, NUMBER_STABLE_STATES):
        diag_percent_vote_i[electoral_tally_pointer - 1, electoral_tally_pointer - 1] = percent_vote_i[electoral_tally_pointer - 1, 0]
        diag_percent_vote_f[electoral_tally_pointer - 1, electoral_tally_pointer - 1] = percent_vote_f[0, electoral_tally_pointer - 1]
        if sum_prob_annihalation[electoral_tally_pointer - 1, 0] == 0:
            diag_reciprocal_sum_prob_annihalation[electoral_tally_pointer - 1, electoral_tally_pointer - 1] = 0
        else:
            diag_reciprocal_sum_prob_annihalation[electoral_tally_pointer - 1, electoral_tally_pointer - 1] = 1/sum_prob_annihalation[electoral_tally_pointer - 1, 0]
    product_i_annihalation = np.dot(diag_percent_vote_i, do_transition_if_annihalation)
    product_f_annihalation = np.dot(do_transition_if_annihalation, diag_percent_vote_f)
    product_reciprocal_sum_annihalation = np.dot(diag_reciprocal_sum_prob_annihalation, do_transition_if_annihalation)
    if 0 == 1:
        fileout.write('annihalation factor 1 =')
        fileout.write(np.matrix.round(product_i_annihalation*1000)/1000)
        fileout.write('annihalation factor 2 =')
        fileout.write(np.matrix.round(product_f_annihalation*1000)/1000)
        fileout.write('annihalation factor 3 =')
        fileout.write(np.matrix.round(product_reciprocal_sum_annihalation*1000)/1000)
        fileout.write('======')
        fileout.write(do_transition_if_annihalation)
        fileout.write('======')
        fileout.write(product_i_annihalation)
        fileout.write('======')
    transition_matrix_annihalation = do_transition_if_annihalation*product_i_annihalation
    sum_transition_matrix_annihalation = sum(sum(transition_matrix_annihalation))

    if abs(np.sum(percent_vote_f) - 1) < 0.000001:
        fileout.write('transition_matrix_generation = ')
        fileout.write(transition_matrix_generation)
        factor = (1 - sum_transition_matrix_generation)/np.sum(percent_vote_f)
        fileout.write('factor = ')
        fileout.write(factor)
        transition_matrix_continuation = factor*transition_matrix_continuation
        transition_matrix_generation = (1 - factor)*transition_matrix_generation
    else:
        fileout.write('BIG PROBLEM HERE')
        return

    #fileout.write('===   transition_matrix_continuation = ')
    #fileout.write(transition_matrix_continuation)
    return transition_matrix_generation, transition_matrix_continuation, transition_matrix_annihalation


def makePredictions(zz):
    # NOW I WANT TO MAKE MY PREDICTIONS (MORE OR LESS)
    # for more information on Cholesky factorizations see:
    # (1) http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.cholesky.html
    # (2) http://www.iecn.u-nancy.fr/~pincon/nsp/nsp_manual/manualli133.html
    NUM_SAMPLES = 53000;
    #covariance_matrix = np.cov(e_1, e_2, e_3) # this calculates the covariance matrix
    #fileout.write(0)
    #fileout.write(0)
    #fileout.write(e)
    covariance_matrix = np.cov([e_1, e_2, e_3, e_4, e_5, e_6, e_7]) # this calculates the covariance matrix
    covariance_matrix = np.cov([e_1, e_2, e_3, e_4, e_5, e_6]) # this calculates the covariance matrix
    covariance_matrix = np.cov([e_1, e_2, e_3, e_4, e_5]) # this calculates the covariance matrix
    fileout.write('covariance_matrix=')
    fileout.write(covariance_matrix)
    L = np.linalg.cholesky(covariance_matrix)
    fileout.write('L=')
    fileout.write(L)
    #z = np.tile(mean_array, (NUM_SAMPLES,1)) + np.dot(np.random.randn(NUM_SAMPLES,3), L)
    z1 = np.tile(mean_array, (NUM_SAMPLES,1))
    z2 = np.dot(np.random.randn(NUM_SAMPLES,5), np.transpose(L))
    #fileout.write(z1)
    #fileout.write(z2)
    z = z1 + z2
    #now perform a check
    mean_ = np.mean(z, axis=0)
    std_ = np.std(z, axis=0)
    fileout.write('Compare the following two mean arrays')
    fileout.write(mean_array)
    fileout.write(mean_)
    fileout.write('Compare the following two std arrays')
    fileout.write(std_array, )
    fileout.write(std_)
    #####################################################
    ############### SECTION TWO #######################
    #####################################################



    DISPLAY = 1
    if DISPLAY == 1:
        fileout.write(mean_array)
        fileout.write(std_array)
        fileout.write(covariance_matrix)
        fileout.write(L)
        fileout.write('z1 = ')
        fileout.write(mean_)
        fileout.write(std_)
    
    
    #A = covariance_matrix*mean_array
    #B = covariance_matrix*(mean_array.T)
    #fileout.write(A)
    #fileout.write(B)
    
def make_historical_data(a):

    #####################################################
    ###############SECTION ONE: DECALRE DATA #######################
    #####################################################
    # Key on Parties
    #  1 = Federalist (1789)
    #  2 = Democratic-Republican (1796)
    #  3 = Clintonian (1808; 6___ votes; one-hit wonder)
    #  4 = N-R (1820)
    #  5 = Clay (1824)
    #  6 = Democratic (1828)
    #  7 = Floyd (1832)
    #  8 = Wirt
    #  9 = Whig (1836)?
    # 10 = Republican (1856)
    # 11 = American (1856)
    # 12 = Constutional-Union (1860)
    # 13 = Douglas (1860)
    # 14 = Liberal Republican (1872)
    # 15 = Weaver (1892)
    # 16 = Populist (1896)
    # 17 = Taft (1912)
    # 18 = Progressive (1924)
    # 19 = Dixicrat (1948; one-hit wonder)
    # REVIEW ALL ABOVE FOR PARENTHESIS
    # 20 = Jones (1956; 1 faithless elector; one-hit wonder)
    # 21 = Byrd (1960; 15 unpledged electors; one-hit wonder)
    # 22 = American Independent [SPELLING?] (1968; 46 electors; one-hit wonder)
    # 23 = Libertarian Party (1972; 1 ___ elector; one-hit wonder)
    # 24 = Reagan (1976; 1 ___ elector; not one-hit wonder)
    # 25 = Bentsen (1988;  1 _____ elector; Notes: Bentsen, who is dukakis VP choice, actually gets one vote for a Bentson(P)/Dukakis (VP) ticket)
    # 99 = Not-Affiliated
    
    # 1872, 8, 4, 353, '', voting age adults in census
    
    # HISTORICAL RECORD (12th Ammendment ratified June 15, 1804)
    # Thomas Jefferson (1792, 1796, 1800, 1804)
    # 1800 THIS IS ALSO DECIDED IN THE HOUSE, http://history1800s.about.com/od/leaders/a/electionof1800.htm
    HISTORICAL_RECORD = [
        [1804, 8, 3, 176, '', 0],
        [1804, 'Thomas Jefferson', ['Democratic-Republican'], [162], 'win', 0, 'no record of popular vote'],
        [1804, 'Charles C. Pinckney', ['Federalist'], [14], 'lose', 0, 'no record of popular vote'],
        [1804, 'None', ['No'], [0], 'lose', 0, 'no record of popular vote'],
        [1804, 'None', ['No'], [0], 'lose', 0, 'no record of popular vote'],
        [1804, 'None', ['No'], [0], 'lose', 0, 'no record of popular vote'],
        [1804, 'None', ['No'], [0], 'lose', 0, 'no record of popular vote'],
        [1804, 'None', ['No'], [0], 'lose', 0, 'no record of popular vote'],
        [1808, 8, 4, 175, '', 0],
        [1808, 'James Madison', ['Democratic-Republican'], [122], 'win', 0, 'no record of popular vote'],
        [1808, 'Charles C. Pinckney', ['Federalist'], [47], 'lose', 0, 'no record of popular vote'],
        [1808, 'George Clinton', ['Democratic-Republican'], [6], 'lose', 0, 'no record of popular vote'],
        [1808, 'None', ['No'], [0], 'lose', 'no record of popular vote'],
        [1808, 'None', ['No'], [0], 'lose', 'no record of popular vote'],
        [1808, 'None', ['No'], [0], 'lose', 'no record of popular vote'],
        [1808, 'None', ['No'], [0], 'lose', 'no record of popular vote'],
        [1812, 8, 3, 217, 3, 0],
        [1812, 'James Madison', ['Democratic-Republican'], [128], 'win', 0, 'no record of popular vote'],
        [1812, 'De Witt Clinton', ['Federalist'], [89], 'lose', 0, 'no record of popular vote'],
        [1812, 'None', ['No'], [0], 'lose', 0, 'no record of popular vote'],
        [1812, 'None', ['No'], [0], 'lose', 0, 'no record of popular vote'],
        [1812, 'None', ['No'], [0], 'lose', 0, 'no record of popular vote'],
        [1812, 'None', ['No'], [0], 'lose', 0, 'no record of popular vote'],
        [1812, 'None', ['No'], [0], 'lose', 0, 'no record of popular vote'],
        [1816, 8, 3, 217, 3, 0],
        [1816, 'James Monroe', ['Democratic-Republican'], [183], 'win', 0, 'no record of popular vote'],
        [1816, 'Rufus King', ['Federalist'], [34], 'lose', 0, 'no record of popular vote'],
        [1816, 'None', ['No'], [0], 'lose', 0, 'no record of popular vote'],
        [1816, 'None', ['No'], [0], 'lose', 0, 'no record of popular vote'],
        [1816, 'None', ['No'], [0], 'lose', 0, 'no record of popular vote'],
        [1816, 'None', ['No'], [0], 'lose', 0, 'no record of popular vote'],
        [1816, 'None', ['No'], [0], 'lose', 0, 'no record of popular vote'],
        [1820, 8, 3, 235-3, 3, 0], # three electors died and there votes were not counted at the quorum of electors
        [1820, 'James Monroe', ['Democratic-Republican'], [231], 'win', 0, 'no record of popular vote'],
        [1820, 'John Quincy Adams', ['Democratic-Republican'], [1], 'lose', 0, 'no record of popular vote'], # unfaithful leclector gives 1 to qunicy
        [1820, 'None', ['No'], [0], 'lose', 'no record of popular vote'],
        [1820, 'None', ['No'], [0], 'lose', 'no record of popular vote'],
        [1820, 'None', ['No'], [0], 'lose', 'no record of popular vote'],
        [1820, 'None', ['No'], [0], 'lose', 'no record of popular vote'],
        [1820, 'None', ['No'], [0], 'lose', 'no record of popular vote'],
        [1824, 8, 5, 261, 2, 0],
        [1824, 'Andrew Jackson', ['Democratic-Republican'], [99], 'lose', 151271, 'NOTES',  'http://en.wikipedia.org/wiki/United_States_presidential_election,_1824, http://history1800s.about.com/od/leaders/a/electionof1824.htm'],
        [1824, 'John Quincy Adams', ['Democratic-Republican'], [84], 'win', 113122, 'NOTES',  'http://en.wikipedia.org/wiki/United_States_presidential_election,_1824, http://history1800s.about.com/od/leaders/a/electionof1824.htm'],
        [1824, 'William H. Crawford', ['Democratic-Republican', 'Crawfords_Jacksonian-Democratic-Republicans'], [41], 'lose', 40856, 'NOTES',  'http://en.wikipedia.org/wiki/United_States_presidential_election,_1824, http://history1800s.about.com/od/leaders/a/electionof1824.htm'],
        [1824, 'Henry Clay', ['Democratic-Republican'], [37], 'lose', 47531, 'NOTES', 'http://en.wikipedia.org/wiki/United_States_presidential_election,_1824, http://history1800s.about.com/od/leaders/a/electionof1824.htm'],
        [1824, 'None', ['No'], [0], 'lose',  999999999],
        [1824, 'None', ['No'], [0], 'lose',  999999999],
        [1824, 'None', ['No'], [0], 'lose',  999999999],
        [1828, 8, 3, 261, '', 0],
        [1828, 'Andrew Jackson', ['Democratic', 'Crawfords_Jacksonian-Democratic-Republicans'], [178*(84)/(41 + 84), 178*(41)/(41 + 84)], 'win', 642553], #this is fucking crazy 100% of crawford D-R and andre jackson on the democrats merge. sum of two vote totals ... 'NOTES:The Democratic Party merged its strength from the existing supporters of Jackson and their coalition with some of the supporters of William H. Crawford (the "Old Republicans") and Vice-President John C. Calhoun',  'http://en.wikipedia.org/wiki/United_States_presidential_election,_1828'],
        [1828, 'John Quincy Adams', ['National-Republican'], [83], 'lose', 500897],
        [1828, 'None', ['No'], [0], 'lose',  999999999],
        [1828, 'None', ['No'], [0], 'lose',  999999999],
        [1828, 'None', ['No'], [0], 'lose',  999999999],
        [1828, 'None', ['No'], [0], 'lose',  999999999],
        [1828, 'None', ['No'], [0], 'lose',  999999999],
        [1832, 8, 5, 286, '', 0],
        [1832, 'Andrew Jackson', ['Democratic'], [219], 'win', 701780],
        [1832, 'Henry Clay', ['National-Republican'], [49], 'lose', 484205],
        [1832, 'John Floyd', ['Nullifier'], [11], 'lose', 999999999], 
        [1832, 'William Wirt', ['Anti-Masonic'], [7], 'lose', 999999999],
        [1832, 'None', ['No'], [0], 'lose',  999999999],
        [1832, 'None', ['No'], [0], 'lose',  999999999],
        [1832, 'None', ['No'], [0], 'lose',  999999999],
        [1836, 8, 6, 294, 3, 0],
        [1836, 'Martin Van Buren', ['Democratic'], [170], 'win', 764176],
        [1836, 'William Henry Harrison', ['Whig'], [73], 'lose', 550816],
        [1836, 'Hugh L. White', ['Whig'], [26], 'lose', 999999999],
        [1836, 'Daniel Webster', ['Whig'], [14], 'lose', 999999999],
        [1836, 'William P. Mangum', ['Whig'], [11], 'lose', 999999999],
        [1836, 'None', ['No'], [0], 'lose',  999999999],
        [1836, 'None', ['No'], [0], 'lose',  999999999],
        [1840, 8, 3, 294, 3, 0],
        [1840, 'William Henry Harrison', ['Whig'], [234], 'win', 1275390],
        [1840, 'Martin Van Buren', ['Democratic'], [60], 'lose', 1128854],
        [1840, 'None', ['No'], [0], 'lose',  999999999],
        [1840, 'None', ['No'], [0], 'lose',  999999999],
        [1840, 'None', ['No'], [0], 'lose',  999999999],
        [1840, 'None', ['No'], [0], 'lose',  999999999],
        [1840, 'None', ['No'], [0], 'lose',  999999999],
        [1844, 8, 3, 275, 3, 0],
        [1844, 'James K. Polk', ['Democratic'], [170], 'win', 1339494],
        [1844, 'Henry Clay', ['Whig'], [105], 'lose', 1300004],
        [1844, 'None', ['No'], [0], 'lose',  999999999],
        [1844, 'None', ['No'], [0], 'lose',  999999999],
        [1844, 'None', ['No'], [0], 'lose',  999999999],
        [1844, 'None', ['No'], [0], 'lose',  999999999],
        [1844, 'None', ['No'], [0], 'lose',  999999999],
        [1848, 8, 3, 290, 3, 0],
        [1848, 'Zachary Taylor', ['Whig'], [163], 'win', 1361393],
        [1848, 'Lewis Cass', ['Democratic'], [127], 'lose', 1223460],
        [1848, 'None', ['No'], [0], 'lose',  999999999],
        [1848, 'None', ['No'], [0], 'lose',  999999999],
        [1848, 'None', ['No'], [0], 'lose',  999999999],
        [1848, 'None', ['No'], [0], 'lose',  999999999],
        [1848, 'None', ['No'], [0], 'lose',  999999999],
        [1852, 8, 3, 296, 3, 0],
        [1852, 'Franklin Pierce', ['Democratic'], [254], 'win', 1607510],
        [1852, 'Winfield Scott', ['Whig'], [42], 'lose', 1386942],
        [1852, 'None', ['No'], [0], 'lose',  999999999],
        [1852, 'None', ['No'], [0], 'lose',  999999999],
        [1852, 'None', ['No'], [0], 'lose',  999999999],
        [1852, 'None', ['No'], [0], 'lose',  999999999],
        [1852, 'None', ['No'], [0], 'lose',  999999999],
        [1856, 8, 4, 296, 4, 0],
        [1856, 'James Buchanan', ['Democratic'], [174], 'win', 1836072],
        [1856, 'John C. Freemont', ['Republican'], [114], 'lose', 1342345],
        [1856, 'Millard Fillmore', 'American', [8], 'lose', 999999999],
        [1856, 'None', ['No'], [0], 'lose',  999999999],
        [1856, 'None', ['No'], [0], 'lose',  999999999],
        [1856, 'None', ['No'], [0], 'lose',  999999999],
        [1856, 'None', ['No'], [0], 'lose',  999999999],
        [1860, 8, 5, 303, '', 0],
        [1860, 'Abraham Lincoln', ['Republican'], [180], 'win', 1865908],
        [1860, 'John C. Breckinridge', ['Democratic'], [72], 'lose', 848019],
        [1860, 'John Bell', 'Constitutional Union', [39], 'lose', 999999999],
        [1860, 'Stephen A. Douglas', ['Democratic'], [12], 'lose', 999999999],
        [1860, 'None', ['No'], [0], 'lose',  999999999],
        [1860, 'None', ['No'], [0], 'lose',  999999999],
        [1860, 'None', ['No'], [0], 'lose',  999999999],
        [1864, 8, 3, 233, 3, 0],
        [1864, 'Abraham Lincoln', ['Republican'], [212], 'win', 2218388],
        [1864, 'George B. McClellan', ['Democratic'], [21], 'lose', 1812807],
        [1864, 'None', ['No'], [0], 'lose',  999999999],
        [1864, 'None', ['No'], [0], 'lose',  999999999],
        [1864, 'None', ['No'], [0], 'lose',  999999999],
        [1864, 'None', ['No'], [0], 'lose',  999999999],
        [1864, 'None', ['No'], [0], 'lose',  999999999],
        [1868, 8, 3, 294, '', 0],
        [1868, 'Ulysses S. Grant', ['Republican'], [214], 'win', 3013650],
        [1868, 'Horatio Seymour', ['Democratic'], [80], 'lose', 2708744],
        [1868, 'None', ['No'], [0], 'lose',  999999999],
        [1868, 'None', ['No'], [0], 'lose',  999999999],
        [1868, 'None', ['No'], [0], 'lose',  999999999],
        [1868, 'None', ['No'], [0], 'lose',  999999999],
        [1868, 'None', ['No'], [0], 'lose',  999999999],
        [1872, 8, 7, 352-3, '', 0], # horace greeley died before electors were counted. three votes not counted
        [1872, 'Ulysses S. Grant', ['Republican'], [286], 'win', 3598235],
        [1872, 'Thomas A. Hendricks', ['Democratic'], [42], 'lose', 999999999],
        [1872, 'B. Gratz Brown', 'Liberal Republican', [18], 'lose', 999999999],
        [1872, 'Charles J. Jenkins', ['Democratic'], [2], 'lose', 999999999],
        [1872, 'David Davis', 'Liberal Republican', [1], 'lose', 999999999],
        [1872, 'Horace Greeley', ['Democratic'], [0], 'lose', 0, 'Is this correct popular vote?? IS THIS THE CORRECT PARTY??three votes cast for him where disqualified by House beacause he was dead'],
        [1872, 'None', ['No'], [0], 'lose',  999999999],
        [1876, 8, 3, 369, '', 0],
        [1876, 'Rutherford B. Hayes', ['Republican'], [185], 'win', 4034311],
        [1876, 'Samuel J. Tilden', ['Democratic'], [184], 'lose', 4288546],
        [1876, 'None', ['No'], [0], 'lose',  999999999],
        [1876, 'None', ['No'], [0], 'lose',  999999999],
        [1876, 'None', ['No'], [0], 'lose',  999999999],
        [1876, 'None', ['No'], [0], 'lose',  999999999],
        [1876, 'None', ['No'], [0], 'lose',  999999999],
        [1880, 8, 3, 369, 3, 0],
        [1880, 'James Garfield', ['Republican'], [214], 'win', 4446158],
        [1880, 'Winfield S. Hancock', ['Democratic'], [155], 'lose', 4444260],
        [1880, 'None', ['No'], [0], 'lose',  999999999],
        [1880, 'None', ['No'], [0], 'lose',  999999999],
        [1880, 'None', ['No'], [0], 'lose',  999999999],
        [1880, 'None', ['No'], [0], 'lose',  999999999],
        [1880, 'None', ['No'], [0], 'lose',  999999999],
        [1884, 8, 3, 401, 3, 0],
        [1884, 'Grover Cleaveland', ['Democratic'], [219], 'win', 4874621],
        [1884, 'James G. Blaine', ['Republican'], [182], 'lose', 4848936],
        [1884, 'None', ['No'], [0], 'lose',  999999999],
        [1884, 'None', ['No'], [0], 'lose',  999999999],
        [1884, 'None', ['No'], [0], 'lose',  999999999],
        [1884, 'None', ['No'], [0], 'lose',  999999999],
        [1884, 'None', ['No'], [0], 'lose',  999999999],
        [1888, 8, 3, 401, 3, 0],
        [1888, 'Benjamin Harrison', ['Republican'], [233], 'win', 5443892],
        [1888, 'Grover Cleaveland', ['Democratic'], [168], 'lose', 5534488],
        [1888, 'None', ['No'], [0], 'lose',  999999999],
        [1888, 'None', ['No'], [0], 'lose',  999999999],
        [1888, 'None', ['No'], [0], 'lose',  999999999],
        [1888, 'None', ['No'], [0], 'lose',  999999999],
        [1888, 'None', ['No'], [0], 'lose',  999999999],
        [1892, 8, 4, 444, '', 0],
        [1892, 'Grover Cleaveland', ['Democratic'], [277], 'win', 5551883],
        [1892, 'Benjamin Harrison', ['Republican'], [145], 'lose', 5179244],
        [1892, 'James B. Weaver', ['Populist'], [22], 'lose',  999999999],
        [1892, 'None', ['No'], [0], 'lose',  999999999],
        [1892, 'None', ['No'], [0], 'lose',  999999999],
        [1892, 'None', ['No'], [0], 'lose',  999999999],
        [1892, 'None', ['No'], [0], 'lose',  999999999],
        [1896, 8, 3, 447, 4, 0],
        [1896, 'William McKinley', ['Republican'], [271], 'win',  7108480],
        [1896, 'William J. Bryan', ['Democratic', 'Populist'], [149, 27], 'lose',  6511495, 'file:///C:/Users/qBear/Desktop/U.%20S.%20Electoral%20College%20%20Historical%20Election%20Results%201789-1996.htm'],## THIS NEEDS TO BE LOKED AT CAREFULLY
        [1896, 'None', ['No'], [0], 'lose',  999999999],
        [1896, 'None', ['No'], [0], 'lose',  999999999],
        [1896, 'None', ['No'], [0], 'lose',  999999999],
        [1896, 'None', ['No'], [0], 'lose',  999999999],
        [1896, 'None', ['No'], [0], 'lose',  999999999],
        [1900, 8, 3, 447, 3, 0],
        [1900, 'William McKinley', ['Republican'], [292], 'win', 7228864, 'http://en.wikipedia.org/wiki/United_States_presidential_election,_1900'],
        [1900, 'William J. Bryan', ['Democratic', 'Populist'], [100, 55], 'lose',  6370932, 'http://en.wikipedia.org/wiki/United_States_presidential_election,_1900'], ## THIS NEEDS TO BE LOKED AT CAREFULLY
        [1900, 'None', ['No'], [0], 'lose',  999999999],
        [1900, 'None', ['No'], [0], 'lose',  999999999],
        [1900, 'None', ['No'], [0], 'lose',  999999999],
        [1900, 'None', ['No'], [0], 'lose',  999999999],
        [1900, 'None', ['No'], [0], 'lose',  999999999],
        [1904, 8, 3, 476, 3, 0],
        [1904, 'Theodore Roosevelt', ['Republican'], [336], 'win', 7626593],
        [1904, 'Alton B. Parker', ['Democratic'], [140], 'lose',  5082898],
        [1904, 'None', ['No'], [0], 'lose',  999999999],
        [1904, 'None', ['No'], [0], 'lose',  999999999],
        [1904, 'None', ['No'], [0], 'lose',  999999999],
        [1904, 'None', ['No'], [0], 'lose',  999999999],
        [1904, 'None', ['No'], [0], 'lose',  999999999],
        [1908, 8, 3, 483, 3, 0],
        [1908, 'William H. Taft', ['Republican'], [321], 'win', 7676258],
        [1908, 'William J. Bryan', ['Democratic'], [162], 'lose', 6406801],
        [1908, 'None', ['No'], [0], 'lose',  999999999],
        [1908, 'None', ['No'], [0], 'lose',  999999999],
        [1908, 'None', ['No'], [0], 'lose',  999999999],
        [1908, 'None', ['No'], [0], 'lose',  999999999],
        [1908, 'None', ['No'], [0], 'lose',  999999999],
        [1912, 8, 4, 531, 4, 0],
        [1912, 'Woodrow Wilson', ['Democratic'], [435], 'win', 6293152],
        [1912, 'Theodore Roosevelt', ['Progressive'], [88], 'lose', 4119207],
        [1912, 'William H. Taft', ['Republican'], [8], 'lose', 7676258], 
        [1912, 'None', ['No'], [0], 'lose',  999999999],
        [1912, 'None', ['No'], [0], 'lose',  999999999],
        [1912, 'None', ['No'], [0], 'lose',  999999999],
        [1912, 'None', ['No'], [0], 'lose',  999999999],
        [1916, 8, 3, 531, 3, 0],
        [1916, 'Woodrow Wilson', ['Democratic'], [277], 'win', 6293152],
        [1916, 'Charles E. Hughes', ['Republican'], [254], 'lose', 4119207],
        [1916, 'None', ['No'], [0], 'lose',  999999999],
        [1916, 'None', ['No'], [0], 'lose',  999999999],
        [1916, 'None', ['No'], [0], 'lose',  999999999],
        [1916, 'None', ['No'], [0], 'lose',  999999999],
        [1916, 'None', ['No'], [0], 'lose',  999999999],
        [1920, 8, 3, 531, 3, 0],
        [1920, 'Warren G. Harding', ['Republican'], [404], 'win', 16153115],
        [1920, 'James M. Cox', ['Democratic'], [127], 'lose', 9133092],
        [1920, 'None', ['No'], [0], 'lose',  999999999],
        [1920, 'None', ['No'], [0], 'lose',  999999999],
        [1920, 'None', ['No'], [0], 'lose',  999999999],
        [1920, 'None', ['No'], [0], 'lose',  999999999],
        [1920, 'None', ['No'], [0], 'lose',  999999999],
        [1924, 8, 4, 531, 4, 0],
        [1924, 'Calvin Coolidge', ['Republican'], [382], 'win', 15719921],
        [1924, 'John W. Davis', ['Democratic'], [136], 'lose', 8386704],
        [1924, 'Robert M. LaFollette', 'Progressive', [13], 'lose', 4822856],
        [1924, 'None', ['No'], [0], 'lose',  999999999],
        [1924, 'None', ['No'], [0], 'lose',  999999999],
        [1924, 'None', ['No'], [0], 'lose',  999999999],
        [1924, 'None', ['No'], [0], 'lose',  999999999],
        [1928, 8, 3, 531, 3, 0],
        [1928, 'Herbert C. Hoover', ['Republican'], [444], 'win', 21437277],
        [1928, 'Alfred E. Smith', ['Democratic'], [87], 'lose', 15007698],
        [1928, 'None', ['No'], [0], 'lose',  999999999],
        [1928, 'None', ['No'], [0], 'lose',  999999999],
        [1928, 'None', ['No'], [0], 'lose',  999999999],
        [1928, 'None', ['No'], [0], 'lose',  999999999],
        [1928, 'None', ['No'], [0], 'lose',  999999999],
        [1932, 8, 3, 531, 3, 0],
        [1932, 'Franklin D. Roosevelt', ['Democratic'], [472], 'win', 22829501],
        [1932, 'Herbert C. Hoover', ['Republican'], [59], 'lose', 15760684],
        [1932, 'None', ['No'], [0], 'lose',  999999999],
        [1932, 'None', ['No'], [0], 'lose',  999999999],
        [1932, 'None', ['No'], [0], 'lose',  999999999],
        [1932, 'None', ['No'], [0], 'lose',  999999999],
        [1932, 'None', ['No'], [0], 'lose',  999999999],
        [1936, 8, 3, 531, 3, 0],
        [1936, 'Franklin D. Roosevelt', ['Democratic'], [523], 'win', 27757333],
        [1936, 'Alfred M. Landon', ['Republican'], [8], 'lose', 16684231],
        [1936, 'None', ['No'], [0], 'lose',  999999999],
        [1936, 'None', ['No'], [0], 'lose',  999999999],
        [1936, 'None', ['No'], [0], 'lose',  999999999],
        [1936, 'None', ['No'], [0], 'lose',  999999999],
        [1936, 'None', ['No'], [0], 'lose',  999999999],
        [1940, 8, 3, 531, 3, 0],
        [1940, 'Franklin D. Roosevelt', ['Democratic'], [449], 'win', 27313041],
        [1940, 'Wendell L. Wilkie', ['Republican'], [82], 'lose',  22348480],
        [1940, 'None', ['No'], [0], 'lose',  999999999],
        [1940, 'None', ['No'], [0], 'lose',  999999999],
        [1940, 'None', ['No'], [0], 'lose',  999999999],
        [1940, 'None', ['No'], [0], 'lose',  999999999],
        [1940, 'None', ['No'], [0], 'lose',  999999999],
        [1944, 8, 3, 531, 3, 0],
        [1944, 'Franklin D. Roosevelt', ['Democratic'], [432 ], 'win', 25612610],
        [1944, 'Thomas E. Dewey', ['Republican'], [99], 'lose',  22117617],
        [1944, 'None', ['No'], [0], 'lose',  999999999],
        [1944, 'None', ['No'], [0], 'lose',  999999999],
        [1944, 'None', ['No'], [0], 'lose',  999999999],
        [1944, 'None', ['No'], [0], 'lose',  999999999],
        [1944, 'None', ['No'], [0], 'lose',  999999999],
        [1948, 8, 4, 531, 4, 0],
        [1948, 'Harry S. Truman', ['Democratic'], [303 ], 'win', 24179345],
        [1948, 'Thomas E. Dewey', ['Republican'], [189], 'lose', 21991291], 
        [1948, 'J. Strom Thurmond', ['Dixiecrat'], [39], 'lose', 999999999],
        [1948, 'None', ['No'], [0], 'lose',  999999999],
        [1948, 'None', ['No'], [0], 'lose',  999999999],
        [1948, 'None', ['No'], [0], 'lose',  999999999],
        [1948, 'None', ['No'], [0], 'lose',  999999999],
        [1952, 8, 3, 531, 3, 0],
        [1952, 'Dwight D. Eisenhower', ['Republican'], [442], 'win', 33936234],
        [1952, 'Adlai Stevenson', ['Democratic'], [89], 'lose', 27314992],
        [1952, 'None', ['No'], [0], 'lose',  999999999],
        [1952, 'None', ['No'], [0], 'lose',  999999999],
        [1952, 'None', ['No'], [0], 'lose',  999999999],
        [1952, 'None', ['No'], [0], 'lose',  999999999],
        [1952, 'None', ['No'], [0], 'lose',  999999999],
        [1956, 8, 4, 531, 3, 0],
        [1956, 'Dwight D. Eisenhower', ['Republican'], [457], 'win', 33936234],
        [1956, 'Adlai Stevenson', ['Democratic'], [73], 'lose', 27314992],
        [1956, 'Walter B. Jones', ['Democratic'], [1], 'lose', 999999999, 'http://en.wikipedia.org/wiki/Walter_Burgwyn_Jones#United_States_presidential_election_of_1956)   In the 1956 Presidential election, faithless elector W. F. Turner cast his vote for Jones, who was a circuit court judge in Turners home town, for President of the United States and Herman E. Talmadge for Vice President, instead of voting for Adlai Stevenson, and Estes Kefauver'],
        [1956, 'None', ['No'], [0], 'lose',  999999999],
        [1956, 'None', ['No'], [0], 'lose',  999999999],
        [1956, 'None', ['No'], [0], 'lose',  999999999],
        [1956, 'None', ['No'], [0], 'lose',  999999999],
        [1960, 8, 4, 537, '', 0],
        [1960, 'John F. Kennedy', ['Democratic'], [303], 'win', 34226731],
        [1960, 'Richard M. Nixon', ['Republican'], [219], 'lose', 34108157],
        [1960, 'Harry F. Byrd', ['Democratic'], [15], 'lose', 'Notes: another strange one.  Although Byrd [sic 3-rd highest getter] was never a candidate in a presidential election, he nevertheless received 116,248 votes in the 1956 election. In the 1960 election, he received 15 electoral votes from unpledged electors: all eight from Mississippi, six of Alabamas 11 (the rest going to John F. Kennedy), and 1 from Oklahoma (the rest going to Richard Nixon).'],
        [1960, 'None', ['No'], [0], 'lose',  999999999],
        [1960, 'None', ['No'], [0], 'lose',  999999999],
        [1960, 'None', ['No'], [0], 'lose',  999999999],
        [1960, 'None', ['No'], [0], 'lose',  999999999],
        [1964, 8, 3, 538, 3, 0],
        [1964, 'Lyndon B. Johnson', ['Democratic'], [486], 'win', 43129566],
        [1964, 'Barry M. Goldwater', ['Republican'], [52], 'lose',  27178188],
        [1964, 'None', ['No'], [0], 'lose',  999999999],
        [1964, 'None', ['No'], [0], 'lose',  999999999],
        [1964, 'None', ['No'], [0], 'lose',  999999999],
        [1964, 'None', ['No'], [0], 'lose',  999999999],
        [1964, 'None', ['No'], [0], 'lose',  999999999],
        [1968, 8, 4, 538, 4, 0],
        [1968, 'Richard M. Nixon', ['Republican'], [301], 'win', 31785480],
        [1968, 'Hubert H. Humphrey', ['Democratic'], [191], 'lose', 31275166],
        [1968, 'George C. Wallace', ['American Independent'], [46], 'lose', 9906473],
        [1968, 'None', ['No'], [0], 'lose',  999999999],
        [1968, 'None', ['No'], [0], 'lose',  999999999],
        [1968, 'None', ['No'], [0], 'lose',  999999999],
        [1968, 'None', ['No'], [0], 'lose',  999999999],
        [1972, 8, 4, 538, 4, 0],
        [1972, 'Richard M. Nixon', ['Republican'], [520], 'win', 47169911],
        [1972, 'George S. McGovern', ['Democratic'], [17], 'lose', 29170383],
        [1972, 'John Hospers', ['Libertarian'], [1], 'lose', 999999999],
        [1972, 'None', ['No'], [0], 'lose',  999999999],
        [1972, 'None', ['No'], [0], 'lose',  999999999],
        [1972, 'None', ['No'], [0], 'lose',  999999999],
        [1972, 'None', ['No'], [0], 'lose',  999999999],
        [1976, 8, 4, 538, 4, 0],
        [1976, 'Jimmy Carter', ['Democratic'], [297], 'win', 40830763],
        [1976, 'Gerald R. Ford', ['Republican'], [240], 'lose', 39147793],
        [1976, 'Ronald Regan', ['Republican'], [1], 'lose', 999999999],
        [1976, 'None', ['No'], [0], 'lose',  999999999],
        [1976, 'None', ['No'], [0], 'lose',  999999999],
        [1976, 'None', ['No'], [0], 'lose',  999999999],
        [1976, 'None', ['No'], [0], 'lose',  999999999],
        [1980, 8, 3, 538, 3, 0],
        [1980, 'Ronald Regan', ['Republican'], [489], 'win', 43904153],
        [1980, 'Jimmy Carter', ['Democratic'], [49], 'lose', 35483883],
        [1980, 'None', ['No'], [0], 'lose',  999999999],
        [1980, 'None', ['No'], [0], 'lose',  999999999],
        [1980, 'None', ['No'], [0], 'lose',  999999999],
        [1980, 'None', ['No'], [0], 'lose',  999999999],
        [1980, 'None', ['No'], [0], 'lose',  999999999],
        [1984, 8, 3, 538, '', 0],
        [1984, 'Ronald Regan', ['Republican'], [525], 'win', 54455075],
        [1984, 'Walter F. Mondale', ['Democratic'], [13], 'lose', 37577185],
        [1984, 'None', ['No'], [0], 'lose',  999999999],
        [1984, 'None', ['No'], [0], 'lose',  999999999],
        [1984, 'None', ['No'], [0], 'lose',  999999999],
        [1984, 'None', ['No'], [0], 'lose',  999999999],
        [1984, 'None', ['No'], [0], 'lose',  999999999],
        [1988, 8, 3, 538, 3, 0],
        [1988, 'George Bush', ['Republican'], [426], 'win', 48886097],
        [1988, 'Michael S. Dukakis', ['Democratic'], [111], 'lose', 41809074],
        [1988, 'Lloyd Bentsen', ['I DO NOT KNOW'], [1], 'lose',  999999999],
        [1988, 'None', ['No'], [0], 'lose',  999999999],
        [1988, 'None', ['No'], [0], 'lose',  999999999],
        [1988, 'None', ['No'], [0], 'lose',  999999999],
        [1988, 'None', ['No'], [0], 'lose',  999999999],
        [1992, 8, 3, 538, 3, 0],
        [1992, 'William J. Clinton', ['Democratic'], [370], 'win', 44908254],
        [1992, 'George Bush', ['Republican'], [168], 'lose', 39102343],
        [1992, 'None', ['No'], [0], 'lose',  999999999],
        [1992, 'None', ['No'], [0], 'lose',  999999999],
        [1992, 'None', ['No'], [0], 'lose',  999999999],
        [1992, 'None', ['No'], [0], 'lose',  999999999],
        [1992, 'None', ['No'], [0], 'lose',  999999999],
        [1996, 8, 3, 538, 3, 0],
        [1996, 'William J. Clinton', ['Democratic'], [379], 'win', 45590703],
        [1996, 'Bob Dole', ['Republican'], [159], 'lose', 37816307],
        [1996, 'None', ['No'], [0], 'lose',  999999999],
        [1996, 'None', ['No'], [0], 'lose',  999999999],
        [1996, 'None', ['No'], [0], 'lose',  999999999],
        [1996, 'None', ['No'], [0], 'lose',  999999999],
        [1996, 'None', ['No'], [0], 'lose',  999999999],
        [2000, 8, 3, 538, 3, 0],  
        [2000, 'George Bush', ['Republican'], [525], 'win', 54455075],
        [2000, 'Al Gore', ['Democratic'], [13], 'lose', 37577185],
        [2000, 'None', ['No'], [0], 'lose',  999999999],
        [2000, 'None', ['No'], [0], 'lose',  999999999],
        [2000, 'None', ['No'], [0], 'lose',  999999999],
        [2000, 'None', ['No'], [0], 'lose',  999999999],
        [2000, 'None', ['No'], [0], 'lose',  999999999],
        [2004, 8, 3, 538, 3, 0],  
        [2004, 'George Bush', ['Republican'], [525], 'win', 54455075],
        [2004, 'John Kerry', ['Democratic'], [13], 'lose', 37577185],
        [2004, 'None', ['No'], [0], 'lose',  999999999],
        [2004, 'None', ['No'], [0], 'lose',  999999999],
        [2004, 'None', ['No'], [0], 'lose',  999999999],
        [2004, 'None', ['No'], [0], 'lose',  999999999],
        [2004, 'None', ['No'], [0], 'lose',  999999999],
        [2008, 8, 3, 538, 3, 0],  
        [2008, 'Barack Obama', ['Democratic'], [525], 'win', 54455075],
        [2008, 'John McCain', ['Republican'], [13], 'lose', 37577185],
        [2008, 'None', ['No'], [0], 'lose',  999999999],
        [2008, 'None', ['No'], [0], 'lose',  999999999],
        [2008, 'None', ['No'], [0], 'lose',  999999999],
        [2008, 'None', ['No'], [0], 'lose',  999999999],
        [2008, 'None', ['No'], [0], 'lose',  999999999],
        [2012, 8, 3, 538, 3, 0],  
        [2012, 'Barack Obama', ['Democratic'], [525], 'win', 54455075],
        [2012, 'Mitt Romney', ['Republican'], [13], 'lose', 37577185],
        [2012, 'None', ['No'], [0], 'lose',  999999999],
        [2012, 'None', ['No'], [0], 'lose',  999999999],
        [2012, 'None', ['No'], [0], 'lose',  999999999],
        [2012, 'None', ['No'], [0], 'lose',  999999999],
        [2012, 'None', ['No'], [0], 'lose',  999999999],
        [2012, 'None', ['No'], [0], 'lose',  999999999]]            
    return HISTORICAL_RECORD
# check from 2000 down inclusive
    # [1872, 'David Davis', ____, 1, ______],
    # look carefully at 1836 elelection
    # 999999999
    # 1832 Floyd Party is Nullifier Party http://en.wikipedia.org/wiki/Nullifier_Party

def misc(a):
    input()
    raw_input('Press <ENTER> to continue')
    index_record = -1;
    for index_election in range(0, 52):
        index_record = index_record + 1;
        #fileout.write(HISTORICAL_RECORD[index_record][0])
        number_electoral_college_vote_getters = HISTORICAL_RECORD[index_record][1] - 1
        electoral_college_max = HISTORICAL_RECORD[index_record][3]
        #fileout.write(electoral_college_max)
        for index_candidates in range(0, 7):
            if index_candidates == 0:
                index_record = index_record + 1;
                #fileout.write((HISTORICAL_RECORD[index_record][3]))
                e_temp = [(HISTORICAL_RECORD[index_record][3])/electoral_college_max]
            elif index_candidates < (number_electoral_college_vote_getters - 1):
                index_record = index_record + 1;
                #fileout.write((HISTORICAL_RECORD[index_record][3]))
                e_temp = e_temp + [(HISTORICAL_RECORD[index_record][3])/electoral_college_max]
            elif index_candidates == (number_electoral_college_vote_getters - 1):
                index_record = index_record + 1;
                #fileout.write((HISTORICAL_RECORD[index_record][3]))
                e_temp = e_temp + [(HISTORICAL_RECORD[index_record][3])/electoral_college_max]
            else:
                #fileout.write((HISTORICAL_RECORD[index_record][3]))
                e_temp = e_temp + [(HISTORICAL_RECORD[index_record][3])/electoral_college_max]
        if index_election == 0:
            e = [e_temp]
        else:
            e = e + [e_temp]        
    #fileout.write(e)
    for index_election in range(0, 52):
        if index_election == 0:
            e_1 = [e[index_election][0]]
            e_2 = [e[index_election][1]]
            e_3 = [e[index_election][2]]
            e_4 = [e[index_election][3]]
            e_5 = [e[index_election][4]]
            e_6 = [e[index_election][5]]
            e_7 = [e[index_election][6]]
        else:
            e_1 = e_1 + [e[index_election][0]]
            e_2 = e_2 + [e[index_election][1]]
            e_3 = e_3 + [e[index_election][2]]
            e_4 = e_4 + [e[index_election][3]]
            e_5 = e_5 + [e[index_election][4]]
            e_6 = e_6 + [e[index_election][5]]
            e_7 = e_7 + [e[index_election][6]]
    #fileout.write(e_1)
    #fileout.write(e_7)
                   #e_temp = eval("%s%d%s" % ('HISTORICAL_RECORD[', index_record, '][3]/electoral_college_max'))#
                    #fileout.write(e_temp)
                    #fileout.write("%s%d%s%d%s" % ('e_', index_candidates, ' = [', e_temp, ']')) #
                    #fileout.write(e_1)
                #E__1 = HISTORICAL_RECORD[index_record][3]
                #e__1 = ][3]

    E_1 = np.array([332,365,286,271,379,370,426,525,489,297,520,301,486,303,457,442,303,432,449,523,472,444,382,404,277,435,321,336,292,271,277,233,219,214,185,286,214,212,180,174,254,163,170,234,170,219,178,84,231,183,128,122,162])
    #####################################################
    ###############SECTION TWO: COMPILE DATA #######################
    #####################################################
    # E_1 is the highest number of electoral votes obtained by the field of candidates
    # E_2 is the second-highest number of electoral votes obtained by the field of candidates
    # E_3 is the third-highest number of electoral votes obtained by the field of candidates
    # E_4 is the fourth-highest number of electoral votes obtained by the field of candidates
    # E_5 is the fifth-highest number of electoral votes obtained by the field of candidates
    # E_9 is the lowest number of electoral votes obtained by the field of candidates
    # E_max is the total number of electorals ever intended to be admitted to the the electoral-college vote
    E_1 = np.array([332,365,286,271,379,370,426,525,489,297,520,301,486,303,457,442,303,432,449,523,472,444,382,404,277,435,321,336,292,271,277,233,219,214,185,286,214,212,180,174,254,163,170,234,170,219,178,84,231,183,128,122,162])
    E_2 = np.array([206,173,251,266,159,168,111,13,49,240,17,191,52,219,73,89,189,99,82,8,59,87,136,127,254,88,162,140,155,176,145,168,182,155,184,42,80,21,72,114,42,127,105,60,73,49,83,99,1,34,89,47,14])
    E_max = np.array([538,538,538,538,538,538,538,538,538,538,538,538,538,537,531,531,531,531,531,531,531,531,531,531,531,531,483,476,447,447,444,401,401,369,369,352,294,233,303,296,296,290,275,294,294,286,261,261,235,217,217,175,176])
    # normalize the data
    #e_1 = E_1/E_max
    #e_2 = E_2/E_max
    # get data
    mean_array = np.array([np.mean(e_1), np.mean(e_2), np.mean(e_3), np.mean(e_4), np.mean(e_5), np.mean(e_6), np.mean(e_7)])
    std_array = np.array([np.std(e_1), np.std(e_2), np.std(e_3), np.std(e_4), np.std(e_5), np.std(e_6), np.std(e_7)])
    fileout.write('Relative stability (a/b)= ')
    fileout.write(mean_array[0:5]/std_array[0:5])
    fileout.write('Relative stability (a/b)*exp((a/b)) = ')
    fileout.write((mean_array[0:5]/std_array[0:5])*np.exp((mean_array[0:5]/std_array[0:5])))
    mean_array = np.array([np.mean(e_1), np.mean(e_2), np.mean(e_3), np.mean(e_4), np.mean(e_5)])
    std_array = np.array([np.std(e_1), np.std(e_2), np.std(e_3), np.std(e_4), np.std(e_5)])




