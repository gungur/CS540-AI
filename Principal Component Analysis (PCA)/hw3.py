import scipy.linalg
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    x = np.load(filename)
    np.mean(x, axis=0)
    dataset = x - np.mean(x, axis=0)
    return dataset


def get_covariance(dataset):
    x = dataset
    np.transpose(x)
    np.dot(x, np.transpose(x))
    y = np.dot(np.transpose(x), x)
    covariance = y / (len(x) - 1)
    return covariance


def get_eig(S, m):
    Lamda, U = eigh(S, subset_by_index=[len(S) - m, len(S) - 1])
    Lamda = np.diag(Lamda)
    Lamda = np.flip(Lamda)
    U = np.fliplr(U)
    return Lamda, U


def get_eig_prop(S, prop):
    variance = prop * np.trace(S)
    Lamda, U = eigh(S, subset_by_value=(variance, np.inf))
    Lamda = np.diag(Lamda)
    Lamda = np.flip(Lamda)
    U = np.fliplr(U)
    return Lamda, U


def project_image(image, U):
    alpha = np.dot(np.transpose(U), image)
    pca = np.dot(U, alpha)
    return pca


def display_image(orig, proj):
    orig_reshape = np.reshape(orig, (64, 64))
    orig_reshape = np.transpose(orig_reshape)
    proj_reshape = np.reshape(proj, (64, 64))
    proj_reshape = np.transpose(proj_reshape)
    fig, (ax1, ax2) = plt.subplots(figsize=(9, 3), ncols=2)

    ax1.set_title('Original')
    ax2.set_title('Projection')
    plt.imshow(orig_reshape, aspect='equal')
    plt.imshow(proj_reshape, aspect='equal')

    ax1_mappable = ax1.imshow(orig_reshape, cmap='viridis', interpolation='none')
    ax2_mappable = ax2.imshow(proj_reshape, cmap='viridis', interpolation='none')

    fig.colorbar(ax1_mappable, ax=ax1, location='right')
    fig.colorbar(ax2_mappable, ax=ax2, location='right')

    return fig, ax1, ax2
