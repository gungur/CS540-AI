import scipy
import numpy
import matplotlib
import csv


def load_data(filepath):
    with open(filepath) as csvfile:
        reader = csv.DictReader(csvfile)
        dictlist = list(reader)
    return dictlist


def calc_features(row):
    valuelist = list(dict.values(row))
    valuelist.pop(0)
    valuelist.pop(0)
    for i in range(len(valuelist)):
        valuelist[i] = float(valuelist[i])
    numpy_array = numpy.array(valuelist, dtype=numpy.float64)
    return numpy_array


def hac(features):
    n = len(features)
    cluster_array = numpy.zeros([(n - 1), 4])
    dist_array = scipy.spatial.distance.cdist(features, features, 'euclidean')
    dist_array = numpy.array(dist_array)
    min_dist = 0
    first_cluster_index = 0
    second_cluster_index = 0
    for i in range(n - 2):
        min_dist = dist_array[0, 1]
        for j in range((i + 1), (n - 1)):
            if min_dist > dist_array[i, j]:
                min_dist = dist_array[i, j]
                first_cluster_index = i
                second_cluster_index = j


def fig_hac(Z, names):
    raise NotImplementedError


def normalize_features(features):
    raise NotImplementedError
