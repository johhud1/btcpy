import time
import operator
import random
import sys
from collections import defaultdict
import numpy as np, argparse
import matplotlib.pyplot as plt
from scipy.stats.stats import zscore

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import metrics

from obj_func import obj_func
from sample_entropy import sample_entropy
import scipy as sp
from scipy.cluster.vq import vq, kmeans, whiten
from numpy.linalg.linalg import norm
from scipy.optimize._differentialevolution import differential_evolution
from multiprocessing import Pool
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
    S[0][i] = zscore(p1[i:i + 181])
    S[0][i][180] = np.subtract(p1[i+180], p1[i+181])
    S[1][i] = zscore(p1[i:i + 361])
    S[1][i][360] = np.subtract(p1[i+360], p1[i+361])
    S[2][i] = zscore(p1[i:i + 721])
    S[2][i][720] = np.subtract(p1[i+720], p1[i+721])

print 'done building time-series subsets.'

print 'Clustering..'
clustering_start = time.time()
print 'Started: ' + str(clustering_start)

clusters = 300
# TODO: should append price-diff and not zscore it (like how matlab impl does it)??
# TODO: is initializing a new clusterer necessary for each sample set??
clusterer180 = MiniBatchKMeans(n_clusters=clusters, random_state=10, n_init=4)
clusterer360 = MiniBatchKMeans(n_clusters=clusters, random_state=10, n_init=4)
clusterer720 = MiniBatchKMeans(n_clusters=clusters, random_state=10, n_init=4)
kmeans = []
kmeans.append(clusterer180.fit(S[0]))
kmeans.append(clusterer360.fit(S[1]))
kmeans.append(clusterer720.fit(S[2]))
print 'np.shape of MiniBatchKmeans fit: ' + str(np.shape(kmeans))

clustering_end = time.time()
print 'End    : ' + str(clustering_end)
print 'Elapsed: ' + str(clustering_end - clustering_start)
print 'Done clustering..'

# for i in range(0, len(timeseries_clustered)):
#    assert len(timeseries_clustered[i]) == clusters, \
#            'time-series cluster array length not equal to expected number of clusters. Expected: %r -- Actual: %r' \
#            % (clusters, len(timeseries_clustered[i]))

def sub_sample(S):
    indices = sorted(random.sample(range(len(S[0])), min(len(S[0]), 20000)))
    print 'Down sampling large number of samples ' + str(len(S[0])) + ' to ' + str(len(indices))
    s_sampled = []
    for k in range (0, len(S)):
        s_sampled.append([S[k][i] for i in sorted(indices)])
    return s_sampled

print 'Cluster Selection'
sample_entropy_start = time.time()
print 'Started : ' + str(sample_entropy_start)
top_clusters = []
labels = []
S_sub_sampled = sub_sample(S)
labels.append(kmeans[0].predict(S_sub_sampled[0]))
labels.append(kmeans[1].predict(S_sub_sampled[1]))
labels.append(kmeans[2].predict(S_sub_sampled[2]))
samples = []
samples.append(silhouette_samples(S_sub_sampled[0], labels[0]))
samples.append(silhouette_samples(S_sub_sampled[1], labels[1]))
samples.append(silhouette_samples(S_sub_sampled[2], labels[2]))
label_ss = []

for i in range(0, len(S)):
    print 'Avg silhouette score for sample set ' + str(i) + ': ' + str(np.mean(samples[i]))

for k in range(0, len(samples)):
    label_ss.append(defaultdict(list))
    for label, sample in zip (labels[k], samples[k]):
        label_ss[k][label].append(sample)

label_ss_avg = [None] * len(samples)
for i in range(0, len(samples)):
    label_ss_avg[i] = [None]*clusters
    for label, label_samples in label_ss[i].iteritems():
        label_ss_avg[i][label] = np.mean(label_samples)

for i in range(0, len(kmeans)):
    top_clusters.append({})
    for k in range(0, len(kmeans[i].cluster_centers_)):
        #testing out using cluster silhouette to select most 'useful' 20 clusters

        # take sample entropy of m = 2
        #c_entropy = sample_entropy(kmeans[i].cluster_centers_[k], 2)
        #top_clusters[i].update({c_entropy[1]: kmeans[i].cluster_centers_[k]})
        top_clusters[i].update({label_ss_avg[i][k]: kmeans[i].cluster_centers_[k]})

    # sort map (key = entropy, value = timeseries) by key and take top 20 values
    top_20_cluster_ts = sorted(top_clusters[i].items(), key=operator.itemgetter(0), reverse=True)[0:130]
    top_clusters[i] = [x[1] for x in top_20_cluster_ts]
    print 'Avg silhouette score of select clusters for sample set ' + str(i) + ': ' + str(np.mean([x[0] for x in top_20_cluster_ts]))

# entropys should now contain array (length 3) of array (length 20) highest entropy time-series (each of length of that particular
top_clusters = np.array(top_clusters)
sample_entropy_end = time.time()
print 'End     : ' + str(sample_entropy_end)
print 'Elapsed : ' + str(sample_entropy_end - sample_entropy_start)
print 'Done with Cluster Selection...'

#for k in range(0, len(top_clusters)):
#    for i in range(0, len(top_clusters[k])):
#        plt.plot(top_clusters[k][i])
#        plt.show()

def bsn_distance(v1, v2):
    # TODO: best choice of c?
    #c = -1 / 4
    c = -0.03
    #return\
    r =    np.exp(
            np.multiply(c,
                        np.power(norm(np.subtract(v1, v2),
                                      2),
                                 2)))
   # plt.plot(range(0, len(v1)), v1, range(0, len(v2)), v2)
   # plt.suptitle("Bayesian distance between vectors: dist = " + str(r))
   # plt.show()
    return r


def bayesian(price_slice, price_clusters):
    numerator = 0
    denominator = 0
    end = len(price_slice)
    # iterate over 20 characteristic price data time-series
    for i in range(0, len(price_clusters)):
        distance = bsn_distance(price_slice, price_clusters[i][0:len(price_slice)])
        #expected_price_d = np.subtract(price_clusters[i][end], price_clusters[i][end - 1])
        expected_price_d = price_clusters[i][end]
        numerator += np.multiply(distance, expected_price_d)
        denominator += distance
    result = np.divide(numerator, denominator)
    return result

print 'Bayesion regression'
bayesian_reg_start = time.time()
print 'Started: ' + str(bayesian_reg_start)

def bayesian_slice(i, price_data, cluster_data, slice_size, store, store_index):
    print 'Started bayesian regression for element: ' + str(i) + '/' + str(len(price_data))
    sys.out.flush()
    store[store_index][i] = bayesian(zscore(price_data[i-slice_size:i]), cluster_data[store_index])
    print 'Completed bayesian regression for element: ' + str(i) + '/' + str(len(price_data))
    sys.out.flush()

# TODO: add w0
regressorX = np.zeros((len(p2), 3))
regressorY = np.zeros((len(p2), 1))
p = Pool(4)

#regressorX[0] = p.map(lambda x: bayesian(zscore(p2[x-180:x]), S[0]), range(720, len(p2)))
#regressorX[1] = p.map(lambda x: bayesian(zscore(p2[x-360:x]), S[0]), range(720, len(p2)))
#regressorX[2] = p.map(lambda x: bayesian(zscore(p2[x-720:x]), S[0]), range(720, len(p2)))

#regressorX = np.transpose(regressorX)

for i in range(720, len(p2)):
    #bayesian predicted dp
    regressorX[i][0] = bayesian(zscore(p2[i-180:i]), S[0])
    regressorX[i][1] = bayesian(zscore(p2[i-360:i]), S[1])
    regressorX[i][2] = bayesian(zscore(p2[i-720:i]), S[2])
    print 'Completed bayesian regression for element: ' + str(i) + '/' + str(len(p2))
    #actual dp
    regressorY[i] = p2[i-1] - p2[i]
  #  print 'Predicted price d: ' + str(regressorX[i]) + ' actual: ' + str(regressorY[i]) + ' diff: ' + str(np.sum(regressorX[i]) - regressorY[i])

p.close()
p.join()
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
expectation_weights = differential_evolution(obj_func, bounds, args=(regressorX, regressorY), popsize=50)

de_end = time.time()
print 'End    : ' + str(de_end)
print 'Elapsed: ' + str(de_end - de_start)
print 'Expectation Weights (thetas): ' + str(expectation_weights)
print 'Done with Differential Evolution...'

# TODO: trade testing
def trade(clusters, weights, prices):
    t = 0.01
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
          # print 'dp1: ' + str(dp1) + ' dp2: ' + str(dp2) + ' dp3: ' + str(dp3)
           print 'Error ' + str(100.0 * ((actual_dp - dp) / actual_dp)) + '% Actual dp: ' + str(actual_dp) + ' predicted dp: ' + str(dp)
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

trade(top_clusters, expectation_weights.x, p3)

trading_end = time.time()
print 'End    : ' + str(trading_end)
print 'Elapsed: ' + str(trading_end - trading_start)
print 'Done with Trade Simulation...'

print 'Finished'
# TODO: graph some things???
