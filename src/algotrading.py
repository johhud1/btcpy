import time
import operator
from sample_entropy import sample_entropy
import numpy as np, argparse
from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Run algo-trading sim')
parser.add_argument('file', metavar='F', type=str, help='BTC exchange price data. Containing prices in second column')
parser.add_argument('--delimiter', metavar='-D', default=',', type=str, help='Delimiter char to use when parsing price data file')

args = parser.parse_args()

print "Parsing price data from file: " + args.file

coin_data = np.genfromtxt(args.file, delimiter=args.delimiter)
prices = np.array(coin_data[:,1])
print str(len(prices)) + " price points read from data file "


p1 = np.array(prices[0:len(prices)/3])
p2 = np.array(prices[len(p1):2*len(p1)])

#TODO: parameterize the rest of the algorithm, so that these lengths are not hardcoded into bunch of logic
timeseries_lengths = [180, 360, 720]

end = len(p1)-721
S = [np.zeros((end, 181)), np.zeros((end, 361)), np.zeros((end, 721))]
#lets just skip to the last full window for longest time-series, dropping S_1 and S_2 series
#in this window is relatively small.. shouldn't affect much
for i in range(0, end):
    S[0][i] = p1[i:i+181]
    S[1][i] = p1[i:i+361]
    S[2][i] = p1[i:i+721]

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
    return kmean180, kmean360, kmean720

timeseries_clustered = clusterSeries(S[0], S[1], S[2], clusters)

clustering_end = time.time()
print 'End    : ' + str(clustering_end)
print 'Elapsed: ' + str(clustering_end - clustering_start)
print 'Done clustering..'

#for i in range(0, len(timeseries_clustered)):
#    assert len(timeseries_clustered[i]) == clusters, \
#            'time-series cluster array length not equal to expected number of clusters. Expected: %r -- Actual: %r' \
#            % (clusters, len(timeseries_clustered[i]))


print 'Sample Entropy'
sample_entropy_start = time.time()
print 'Started : ' + str(sample_entropy_start)
entropys = []
for i in range(0, len(timeseries_clustered)):
    entropys.append({})
    for k in range(0, len(timeseries_clustered[i])):
        c_entropy = sample_entropy(timeseries_clustered[i][k], 2)
        #take sample entropy of m = 2
        entropys[i].update({c_entropy[1]: timeseries_clustered[i][k]})

    #sort map (key = entropy, value = timeseries) by key and take top 20 values
    top_20_entropy_ts = sorted(entropys[i].items(), key=operator.itemgetter(0), reverse=True)[0:20]
    entropys[i] = [x[1] for x in top_20_entropy_ts]

#entropys should now contain array (length 3) of array (length 20) highest entropy time-series (each of length of that particular
entropys = np.array(entropys)
sample_entropy_end = time.time()
print 'End     : ' + str(sample_entropy_end)
print 'Elapsed : ' + str(sample_entropy_end - sample_entropy_start)
print 'Done with Sample Entropy...'

#TS (180, 360, 720)

#TODO: bayesian inference
#TODO: differential evolution optimization
#TODO: trade testing

#TODO: graph some things???
