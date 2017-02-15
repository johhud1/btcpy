import time
import operator
import numpy as np, argparse
import matplotlib.pyplot as plt
from scipy.stats.stats import zscore

from obj_func import obj_func
from sample_entropy import sample_entropy
import scipy as sp
from scipy.cluster.vq import vq, kmeans, whiten
from numpy.linalg.linalg import norm
from scipy.optimize._differentialevolution import differential_evolution

# TODO: implementing with 'r' (ask - bid volume factor)  missing since we don't have any ask/volume data in our okcoin.csv

# TODO: note below describe implementation difference from matlab impl. Investigate impact?
# NOTE: implementation differs from BTCPredictor matlab impl, in that the kmeans clustered vectors
# do not contain a last element whose value is the price jump. Clustering is done on time-series vectors
# one element longer than described (181, 361, 721), and price diff is calculated post clustering.

parser = argparse.ArgumentParser(description='Run algo-trading sim')
parser.add_argument('file', metavar='F', type=str, help='BTC exchange price data. Containing prices in second column')
parser.add_argument('--delimiter', metavar='-D', default=',', type=str,
                    help='Delimiter char to use when parsing price data file')

args = parser.parse_args()

print "Parsing price data from file: " + args.file

coin_data = np.genfromtxt(args.file, delimiter=args.delimiter)
prices = np.array(coin_data[:, 1])
print str(len(prices)) + " price points read from data file "

print 'Price start time: ' + str(coin_data[0, 0])
print 'Price end time: ' + str(coin_data[len(coin_data)-1, 0])
print 'Price window (hours): ' + str((coin_data[len(coin_data)-1,0] - coin_data[0,0]) / (60*60))

p1 = np.array(prices[0:len(prices) / 3])
p2 = np.array(prices[len(p1):2 * len(p1)])
p3 = np.array(prices[2 * len(p1): len(prices)])

# TODO: parameterize the rest of the algorithm, so that these lengths are not hardcoded into bunch of logic
timeseries_lengths = [180, 360, 720]

end = len(p1) - 721
S = [np.zeros((end, 181)), np.zeros((end, 361)), np.zeros((end, 721))]
# lets just skip to the last full window for longest time-series, dropping S_1 and S_2 series
# in this window is relatively small.. shouldn't affect much
for i in range(0, end):
    S[0][i] = p1[i:i + 181]
    S[1][i] = p1[i:i + 361]
    S[2][i] = p1[i:i + 721]

print 'done building time-series subsets.'

print 'Clustering..'
clustering_start = time.time()
print 'Started: ' + str(clustering_start)

clusters = 100


def clusterSeries(shortTS, mediumTS, longTS, clusters):
    # kmeans cluster the price vectors in price segment 1
    print 'shape(short TS data): ' + str(np.shape(shortTS))
    print 'shape(medium TS data): ' + str(np.shape(mediumTS))
    print 'shape(long TS data): ' + str(np.shape(longTS))
    kmean180, dist = kmeans(whiten(shortTS), clusters, iter=4)
    kmean360, dist = kmeans(whiten(mediumTS), clusters, iter=4)
    kmean720, dist = kmeans(whiten(longTS), clusters, iter=4)
    print 'shape(short TS data clusters): ' + str(np.shape(kmean180))
    print 'shape(medium TS data clusters): ' + str(np.shape(kmean360))
    print 'shape(long TS data clusters): ' + str(np.shape(kmean720))
    return zscore(kmean180), zscore(kmean360), zscore(kmean720)
# TODO: should append price-diff and not zscore it (like how matlab impl does it)??

timeseries_clustered = clusterSeries(S[0], S[1], S[2], clusters)

clustering_end = time.time()
print 'End    : ' + str(clustering_end)
print 'Elapsed: ' + str(clustering_end - clustering_start)
print 'Done clustering..'

# for i in range(0, len(timeseries_clustered)):
#    assert len(timeseries_clustered[i]) == clusters, \
#            'time-series cluster array length not equal to expected number of clusters. Expected: %r -- Actual: %r' \
#            % (clusters, len(timeseries_clustered[i]))


print 'Sample Entropy'
sample_entropy_start = time.time()
print 'Started : ' + str(sample_entropy_start)
top_ent_clusters = []
for i in range(0, len(timeseries_clustered)):
    top_ent_clusters.append({})
    for k in range(0, len(timeseries_clustered[i])):
        c_entropy = sample_entropy(timeseries_clustered[i][k], 2)
        # take sample entropy of m = 2
        top_ent_clusters[i].update({c_entropy[1]: timeseries_clustered[i][k]})

    # sort map (key = entropy, value = timeseries) by key and take top 20 values
    top_20_entropy_ts = sorted(top_ent_clusters[i].items(), key=operator.itemgetter(0), reverse=True)[0:20]
    top_ent_clusters[i] = [x[1] for x in top_20_entropy_ts]

# entropys should now contain array (length 3) of array (length 20) highest entropy time-series (each of length of that particular
top_ent_clusters = np.array(top_ent_clusters)
sample_entropy_end = time.time()
print 'End     : ' + str(sample_entropy_end)
print 'Elapsed : ' + str(sample_entropy_end - sample_entropy_start)
print 'Done with Sample Entropy...'


def bsn_distance(v1, v2):
    # TODO: best choice of c?
    #c = -1 / 4
    c = -0.15
    return\
        np.exp(
            np.multiply(c,
                        np.power(norm(np.subtract(v1, v2),
                                      2),
                                 2)))


def bayesian(price_slice, price_clusters):
    numerator = 0
    denominator = 0
    end = len(price_slice)
    # iterate over 20 characteristic price data time-series
    for i in range(0, len(price_clusters)):
        distance = bsn_distance(price_slice, price_clusters[i][0:len(price_slice)])
        expected_price_d = np.subtract(price_clusters[i][end], price_clusters[i][end - 1])
        numerator += np.multiply(distance, expected_price_d)
        denominator += distance
    result = np.divide(numerator, denominator)
    return result

print 'Bayesion regression'
bayesian_reg_start = time.time()
print 'Started: ' + str(bayesian_reg_start)

# TODO: add w0
regressorX = np.zeros((len(p2), 3))
regressorY = np.zeros((len(p2), 1))
for i in range(0, len(p2) - 721):
    ps = [p2[i:180], p2[i:360], p2[i:720]]
    for k in range(0, 3):
        regressorX[i][k] = bayesian(zscore(ps[k]), top_ent_clusters[k])

bayesian_reg_end = time.time()
print 'End    : ' + str(bayesian_reg_end)
print 'Elapsed: ' + str(bayesian_reg_end - bayesian_reg_start)
print 'Done with Bayesian Regression...'

print 'Differential Evolution fit'
de_start = time.time()
print 'Started: ' + str(de_start)

# TODO: add one more bound when adding r
bounds = [[0, 1], [0, 1], [0, 1], [0, 1]]
# TODO: verify optional params on differential_evolution
expectation_weights = differential_evolution(obj_func, bounds, args=(regressorX, regressorY), popsize=50, maxiter=15000)

de_end = time.time()
print 'End    : ' + str(de_end)
print 'Elapsed: ' + str(de_end - de_start)
print 'Expectation Weights (thetas): ' + str(expectation_weights)
print 'Done with Differential Evolution...'

# TODO: trade testing
def trade(clusters, weights, prices):
    t = 0.1
    position = 0
    error = 0
    abs_movement_total = 0
    sales = 0
    winning_sales = 0
    purchase_price = 0
    profit = 0
    for i in range(720, len(prices)-1):
       prices180 = zscore(prices[i-180:i])
       prices360 = zscore(prices[i-360:i])
       prices720 = zscore(prices[i-720:i])
       dp1 = bayesian(prices180, clusters[0])
       dp2 = bayesian(prices360, clusters[1])
       dp3 = bayesian(prices720, clusters[2])
       dp = weights[0] + weights[1] * dp1 + weights[2] * dp2 + weights[3] * dp3
       actual_dp = prices[i+1] - prices[i]
       if(i % 1000 == 0):
           print 'Actual dp: ' + str(actual_dp) + ' predicted dp: ' + str(dp)
       error = error + abs(actual_dp - dp)
       abs_movement_total = abs_movement_total + abs(actual_dp)
       if((dp > t) & (position == 0)):
           position = 1
           purchase_price = prices[i]
       if((dp < -t) & (position == 1)):
           position = 0
           sales = sales + 1
           sale_profit = prices[i] - purchase_price
           profit = profit + sale_profit
           if (sale_profit > 0):
               winning_sales = winning_sales + 1
    print 'Trading finished'
    print 'Results:'
    print 'Error: ' + str(error) + ' Absolute price movement: ' + str(abs_movement_total)
    print 'Profit:  ' + str(profit)
    print 'Winning Sales: ' + str(winning_sales) + ' Sales: ' + str(sales) + ' Ratio: ' + str(winning_sales / sales)

#TODO: return stuff
    return

print 'Trade Simulation'
trading_start = time.time()
print 'Started: ' + str(trading_start)

trade(top_ent_clusters, expectation_weights.x, p3)

trading_end = time.time()
print 'End    : ' + str(trading_end)
print 'Elapsed: ' + str(trading_end - trading_start)
print 'Done with Trade Simulation...'

print 'Finished'
# TODO: graph some things???
