import numpy as np
import matplotlib.pyplot as plt


def show_data(x_train, y_train):
    """
    Plots features pairwise, of the following set:
    - average allelic fraction
    - hematocrit
    - platelet
    - white blood cell count
    - hemoglobin
    - age

    :param x_train: x training values (2d numpy array)
    :param y_train: y traiing values (1d numpy array)
    """

    # Isolate features
    genes = x_train[:, 0]
    for i in range(1, 45):
        genes += x_train[:, i]
    genes = genes / 45
    hematocrit = x_train[:, 45]
    platelet = x_train[:, 46]
    wbc = x_train[:, 47]
    hemoglobin = x_train[:, 48]
    age = x_train[:, 49]

    # Mask by +1, -1
    yplus = np.ma.masked_where(y_train <= 0, y_train)
    yminus = np.ma.masked_where(y_train > 0, y_train)
    genes = (genes[~np.array(yplus.mask)], genes[~np.array(yminus.mask)])
    hematocrit = (hematocrit[~np.array(yplus.mask)], hematocrit[~np.array(yminus.mask)])
    platelet = (platelet[~np.array(yplus.mask)], platelet[~np.array(yminus.mask)])
    wbc = (wbc[~np.array(yplus.mask)], wbc[~np.array(yminus.mask)])
    hemoglobin = (hemoglobin[~np.array(yplus.mask)], hemoglobin[~np.array(yminus.mask)])
    age = (age[~np.array(yplus.mask)], age[~np.array(yminus.mask)])

    features = [genes, hematocrit, platelet, wbc, hemoglobin, age]
    labels = ["Genes", "Hematocrit", "Platelet", "White Blood Cell", "Hemoglobin", "Age"]

    # Grid
    num_rows = 2
    num_cols = 2
    fig_index = 0

    # Make all plots
    for i in range(6):
        for j in range(i+1, 6):
            if fig_index % 4 == 0:
                plt.figure(figsize=(10, 10))
                plt.subplots_adjust(hspace=.8, wspace=.8)
            plt.subplot(num_rows, num_cols, (fig_index % 4) + 1, facecolor='white')
            plt.scatter(features[i][0], features[j][0], marker='+', c='r', label='+1 labels for training set')
            plt.scatter(features[i][1], features[j][1], marker=r'$-$', c='b', label='-1 labels for training set')
            plt.xlabel(labels[i])
            plt.ylabel(labels[j])
            fig_index += 1
            if fig_index % 4 == 0 and fig_index != 0:
                plt.show()

    plt.show()
