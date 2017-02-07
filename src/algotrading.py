import csv, time
import sample_entropy
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


def clusterSeries(shortTS, mediumTS, longTS):
    # kmeans cluster the price vectors in price segment 1
    clusters = 100
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


clusterSeries(S[0][:, 0:180], S[1][:, 0:360], S[2][:, 0:720])

clustering_end = time.time()
print 'End    : ' + str(clustering_end)
print 'Elapsed: ' + str(clustering_end - clustering_start)
print 'Done clustering..'

