{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyO828ggNbghLBCgKt65FIz5",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/sumantsharma16/Deep-learning/blob/sumantsharma16-machinelearning/breast%20cancer%20prediction%20using%20deep%20learning.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Breast cancer classification with a simple Neural Network(NN)"
      ],
      "metadata": {
        "id": "XRY29fjeHrG0"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Import dependencies"
      ],
      "metadata": {
        "id": "XwNts_qfz1eY"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd # wrapper for both numpy and pandas\n",
        "import numpy as np # for mathematical operations\n",
        "import matplotlib.pyplot as plt # for visulization of data\n",
        "import sklearn as sns\n",
        "import sklearn.datasets # for import inbuilt datasets\n",
        "from sklearn.model_selection import train_test_split # for splitting the data into training and testing "
      ],
      "metadata": {
        "id": "GxyoAMgoz-zK"
      },
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Data Collection and processing\n",
        "\n",
        "Data is loaded from prebuilt dataset, and processing over come any anomaly in the dataset"
      ],
      "metadata": {
        "id": "nzmlN7zp0qv3"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# loading the data from sklearn\n",
        "breast_cancer_dataset = sklearn.datasets.load_breast_cancer()"
      ],
      "metadata": {
        "id": "QOtiMxoV1DPy"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(breast_cancer_dataset)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8Gb0T7oM1Oiy",
        "outputId": "661c4741-1a0c-430f-cea2-269544f725c0"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "{'data': array([[1.799e+01, 1.038e+01, 1.228e+02, ..., 2.654e-01, 4.601e-01,\n",
            "        1.189e-01],\n",
            "       [2.057e+01, 1.777e+01, 1.329e+02, ..., 1.860e-01, 2.750e-01,\n",
            "        8.902e-02],\n",
            "       [1.969e+01, 2.125e+01, 1.300e+02, ..., 2.430e-01, 3.613e-01,\n",
            "        8.758e-02],\n",
            "       ...,\n",
            "       [1.660e+01, 2.808e+01, 1.083e+02, ..., 1.418e-01, 2.218e-01,\n",
            "        7.820e-02],\n",
            "       [2.060e+01, 2.933e+01, 1.401e+02, ..., 2.650e-01, 4.087e-01,\n",
            "        1.240e-01],\n",
            "       [7.760e+00, 2.454e+01, 4.792e+01, ..., 0.000e+00, 2.871e-01,\n",
            "        7.039e-02]]), 'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,\n",
            "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,\n",
            "       0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0,\n",
            "       1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0,\n",
            "       1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1,\n",
            "       1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0,\n",
            "       0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,\n",
            "       1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1,\n",
            "       1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0,\n",
            "       0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0,\n",
            "       1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1,\n",
            "       1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "       0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,\n",
            "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,\n",
            "       1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0,\n",
            "       0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0,\n",
            "       0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0,\n",
            "       1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,\n",
            "       1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0,\n",
            "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1,\n",
            "       1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,\n",
            "       1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,\n",
            "       1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,\n",
            "       1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,\n",
            "       1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
            "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]), 'frame': None, 'target_names': array(['malignant', 'benign'], dtype='<U9'), 'DESCR': '.. _breast_cancer_dataset:\\n\\nBreast cancer wisconsin (diagnostic) dataset\\n--------------------------------------------\\n\\n**Data Set Characteristics:**\\n\\n    :Number of Instances: 569\\n\\n    :Number of Attributes: 30 numeric, predictive attributes and the class\\n\\n    :Attribute Information:\\n        - radius (mean of distances from center to points on the perimeter)\\n        - texture (standard deviation of gray-scale values)\\n        - perimeter\\n        - area\\n        - smoothness (local variation in radius lengths)\\n        - compactness (perimeter^2 / area - 1.0)\\n        - concavity (severity of concave portions of the contour)\\n        - concave points (number of concave portions of the contour)\\n        - symmetry\\n        - fractal dimension (\"coastline approximation\" - 1)\\n\\n        The mean, standard error, and \"worst\" or largest (mean of the three\\n        worst/largest values) of these features were computed for each image,\\n        resulting in 30 features.  For instance, field 0 is Mean Radius, field\\n        10 is Radius SE, field 20 is Worst Radius.\\n\\n        - class:\\n                - WDBC-Malignant\\n                - WDBC-Benign\\n\\n    :Summary Statistics:\\n\\n    ===================================== ====== ======\\n                                           Min    Max\\n    ===================================== ====== ======\\n    radius (mean):                        6.981  28.11\\n    texture (mean):                       9.71   39.28\\n    perimeter (mean):                     43.79  188.5\\n    area (mean):                          143.5  2501.0\\n    smoothness (mean):                    0.053  0.163\\n    compactness (mean):                   0.019  0.345\\n    concavity (mean):                     0.0    0.427\\n    concave points (mean):                0.0    0.201\\n    symmetry (mean):                      0.106  0.304\\n    fractal dimension (mean):             0.05   0.097\\n    radius (standard error):              0.112  2.873\\n    texture (standard error):             0.36   4.885\\n    perimeter (standard error):           0.757  21.98\\n    area (standard error):                6.802  542.2\\n    smoothness (standard error):          0.002  0.031\\n    compactness (standard error):         0.002  0.135\\n    concavity (standard error):           0.0    0.396\\n    concave points (standard error):      0.0    0.053\\n    symmetry (standard error):            0.008  0.079\\n    fractal dimension (standard error):   0.001  0.03\\n    radius (worst):                       7.93   36.04\\n    texture (worst):                      12.02  49.54\\n    perimeter (worst):                    50.41  251.2\\n    area (worst):                         185.2  4254.0\\n    smoothness (worst):                   0.071  0.223\\n    compactness (worst):                  0.027  1.058\\n    concavity (worst):                    0.0    1.252\\n    concave points (worst):               0.0    0.291\\n    symmetry (worst):                     0.156  0.664\\n    fractal dimension (worst):            0.055  0.208\\n    ===================================== ====== ======\\n\\n    :Missing Attribute Values: None\\n\\n    :Class Distribution: 212 - Malignant, 357 - Benign\\n\\n    :Creator:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian\\n\\n    :Donor: Nick Street\\n\\n    :Date: November, 1995\\n\\nThis is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets.\\nhttps://goo.gl/U2Uwz2\\n\\nFeatures are computed from a digitized image of a fine needle\\naspirate (FNA) of a breast mass.  They describe\\ncharacteristics of the cell nuclei present in the image.\\n\\nSeparating plane described above was obtained using\\nMultisurface Method-Tree (MSM-T) [K. P. Bennett, \"Decision Tree\\nConstruction Via Linear Programming.\" Proceedings of the 4th\\nMidwest Artificial Intelligence and Cognitive Science Society,\\npp. 97-101, 1992], a classification method which uses linear\\nprogramming to construct a decision tree.  Relevant features\\nwere selected using an exhaustive search in the space of 1-4\\nfeatures and 1-3 separating planes.\\n\\nThe actual linear program used to obtain the separating plane\\nin the 3-dimensional space is that described in:\\n[K. P. Bennett and O. L. Mangasarian: \"Robust Linear\\nProgramming Discrimination of Two Linearly Inseparable Sets\",\\nOptimization Methods and Software 1, 1992, 23-34].\\n\\nThis database is also available through the UW CS ftp server:\\n\\nftp ftp.cs.wisc.edu\\ncd math-prog/cpo-dataset/machine-learn/WDBC/\\n\\n.. topic:: References\\n\\n   - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction \\n     for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on \\n     Electronic Imaging: Science and Technology, volume 1905, pages 861-870,\\n     San Jose, CA, 1993.\\n   - O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and \\n     prognosis via linear programming. Operations Research, 43(4), pages 570-577, \\n     July-August 1995.\\n   - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques\\n     to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) \\n     163-171.', 'feature_names': array(['mean radius', 'mean texture', 'mean perimeter', 'mean area',\n",
            "       'mean smoothness', 'mean compactness', 'mean concavity',\n",
            "       'mean concave points', 'mean symmetry', 'mean fractal dimension',\n",
            "       'radius error', 'texture error', 'perimeter error', 'area error',\n",
            "       'smoothness error', 'compactness error', 'concavity error',\n",
            "       'concave points error', 'symmetry error',\n",
            "       'fractal dimension error', 'worst radius', 'worst texture',\n",
            "       'worst perimeter', 'worst area', 'worst smoothness',\n",
            "       'worst compactness', 'worst concavity', 'worst concave points',\n",
            "       'worst symmetry', 'worst fractal dimension'], dtype='<U23'), 'filename': 'breast_cancer.csv', 'data_module': 'sklearn.datasets.data'}\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# loading the data to a data frame\n",
        "data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)"
      ],
      "metadata": {
        "id": "lViTHtBl1R1h"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# print first five rows of data frame\n",
        "data_frame.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 334
        },
        "id": "02GuLDEp1ha1",
        "outputId": "b9d4f915-cb7e-46b2-8055-2a9ba21fa81a"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   mean radius  mean texture  mean perimeter  mean area  mean smoothness  \\\n",
              "0        17.99         10.38          122.80     1001.0          0.11840   \n",
              "1        20.57         17.77          132.90     1326.0          0.08474   \n",
              "2        19.69         21.25          130.00     1203.0          0.10960   \n",
              "3        11.42         20.38           77.58      386.1          0.14250   \n",
              "4        20.29         14.34          135.10     1297.0          0.10030   \n",
              "\n",
              "   mean compactness  mean concavity  mean concave points  mean symmetry  \\\n",
              "0           0.27760          0.3001              0.14710         0.2419   \n",
              "1           0.07864          0.0869              0.07017         0.1812   \n",
              "2           0.15990          0.1974              0.12790         0.2069   \n",
              "3           0.28390          0.2414              0.10520         0.2597   \n",
              "4           0.13280          0.1980              0.10430         0.1809   \n",
              "\n",
              "   mean fractal dimension  ...  worst radius  worst texture  worst perimeter  \\\n",
              "0                 0.07871  ...         25.38          17.33           184.60   \n",
              "1                 0.05667  ...         24.99          23.41           158.80   \n",
              "2                 0.05999  ...         23.57          25.53           152.50   \n",
              "3                 0.09744  ...         14.91          26.50            98.87   \n",
              "4                 0.05883  ...         22.54          16.67           152.20   \n",
              "\n",
              "   worst area  worst smoothness  worst compactness  worst concavity  \\\n",
              "0      2019.0            0.1622             0.6656           0.7119   \n",
              "1      1956.0            0.1238             0.1866           0.2416   \n",
              "2      1709.0            0.1444             0.4245           0.4504   \n",
              "3       567.7            0.2098             0.8663           0.6869   \n",
              "4      1575.0            0.1374             0.2050           0.4000   \n",
              "\n",
              "   worst concave points  worst symmetry  worst fractal dimension  \n",
              "0                0.2654          0.4601                  0.11890  \n",
              "1                0.1860          0.2750                  0.08902  \n",
              "2                0.2430          0.3613                  0.08758  \n",
              "3                0.2575          0.6638                  0.17300  \n",
              "4                0.1625          0.2364                  0.07678  \n",
              "\n",
              "[5 rows x 30 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-382cd625-d9e0-446a-82ba-b2b0bc1721c8\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>mean radius</th>\n",
              "      <th>mean texture</th>\n",
              "      <th>mean perimeter</th>\n",
              "      <th>mean area</th>\n",
              "      <th>mean smoothness</th>\n",
              "      <th>mean compactness</th>\n",
              "      <th>mean concavity</th>\n",
              "      <th>mean concave points</th>\n",
              "      <th>mean symmetry</th>\n",
              "      <th>mean fractal dimension</th>\n",
              "      <th>...</th>\n",
              "      <th>worst radius</th>\n",
              "      <th>worst texture</th>\n",
              "      <th>worst perimeter</th>\n",
              "      <th>worst area</th>\n",
              "      <th>worst smoothness</th>\n",
              "      <th>worst compactness</th>\n",
              "      <th>worst concavity</th>\n",
              "      <th>worst concave points</th>\n",
              "      <th>worst symmetry</th>\n",
              "      <th>worst fractal dimension</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>17.99</td>\n",
              "      <td>10.38</td>\n",
              "      <td>122.80</td>\n",
              "      <td>1001.0</td>\n",
              "      <td>0.11840</td>\n",
              "      <td>0.27760</td>\n",
              "      <td>0.3001</td>\n",
              "      <td>0.14710</td>\n",
              "      <td>0.2419</td>\n",
              "      <td>0.07871</td>\n",
              "      <td>...</td>\n",
              "      <td>25.38</td>\n",
              "      <td>17.33</td>\n",
              "      <td>184.60</td>\n",
              "      <td>2019.0</td>\n",
              "      <td>0.1622</td>\n",
              "      <td>0.6656</td>\n",
              "      <td>0.7119</td>\n",
              "      <td>0.2654</td>\n",
              "      <td>0.4601</td>\n",
              "      <td>0.11890</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>20.57</td>\n",
              "      <td>17.77</td>\n",
              "      <td>132.90</td>\n",
              "      <td>1326.0</td>\n",
              "      <td>0.08474</td>\n",
              "      <td>0.07864</td>\n",
              "      <td>0.0869</td>\n",
              "      <td>0.07017</td>\n",
              "      <td>0.1812</td>\n",
              "      <td>0.05667</td>\n",
              "      <td>...</td>\n",
              "      <td>24.99</td>\n",
              "      <td>23.41</td>\n",
              "      <td>158.80</td>\n",
              "      <td>1956.0</td>\n",
              "      <td>0.1238</td>\n",
              "      <td>0.1866</td>\n",
              "      <td>0.2416</td>\n",
              "      <td>0.1860</td>\n",
              "      <td>0.2750</td>\n",
              "      <td>0.08902</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>19.69</td>\n",
              "      <td>21.25</td>\n",
              "      <td>130.00</td>\n",
              "      <td>1203.0</td>\n",
              "      <td>0.10960</td>\n",
              "      <td>0.15990</td>\n",
              "      <td>0.1974</td>\n",
              "      <td>0.12790</td>\n",
              "      <td>0.2069</td>\n",
              "      <td>0.05999</td>\n",
              "      <td>...</td>\n",
              "      <td>23.57</td>\n",
              "      <td>25.53</td>\n",
              "      <td>152.50</td>\n",
              "      <td>1709.0</td>\n",
              "      <td>0.1444</td>\n",
              "      <td>0.4245</td>\n",
              "      <td>0.4504</td>\n",
              "      <td>0.2430</td>\n",
              "      <td>0.3613</td>\n",
              "      <td>0.08758</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>11.42</td>\n",
              "      <td>20.38</td>\n",
              "      <td>77.58</td>\n",
              "      <td>386.1</td>\n",
              "      <td>0.14250</td>\n",
              "      <td>0.28390</td>\n",
              "      <td>0.2414</td>\n",
              "      <td>0.10520</td>\n",
              "      <td>0.2597</td>\n",
              "      <td>0.09744</td>\n",
              "      <td>...</td>\n",
              "      <td>14.91</td>\n",
              "      <td>26.50</td>\n",
              "      <td>98.87</td>\n",
              "      <td>567.7</td>\n",
              "      <td>0.2098</td>\n",
              "      <td>0.8663</td>\n",
              "      <td>0.6869</td>\n",
              "      <td>0.2575</td>\n",
              "      <td>0.6638</td>\n",
              "      <td>0.17300</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>20.29</td>\n",
              "      <td>14.34</td>\n",
              "      <td>135.10</td>\n",
              "      <td>1297.0</td>\n",
              "      <td>0.10030</td>\n",
              "      <td>0.13280</td>\n",
              "      <td>0.1980</td>\n",
              "      <td>0.10430</td>\n",
              "      <td>0.1809</td>\n",
              "      <td>0.05883</td>\n",
              "      <td>...</td>\n",
              "      <td>22.54</td>\n",
              "      <td>16.67</td>\n",
              "      <td>152.20</td>\n",
              "      <td>1575.0</td>\n",
              "      <td>0.1374</td>\n",
              "      <td>0.2050</td>\n",
              "      <td>0.4000</td>\n",
              "      <td>0.1625</td>\n",
              "      <td>0.2364</td>\n",
              "      <td>0.07678</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 30 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-382cd625-d9e0-446a-82ba-b2b0bc1721c8')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-382cd625-d9e0-446a-82ba-b2b0bc1721c8 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-382cd625-d9e0-446a-82ba-b2b0bc1721c8');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Adding target columns to the data frame\n",
        "data_frame['label'] = breast_cancer_dataset.target"
      ],
      "metadata": {
        "id": "95QxfBfI1oVl"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# print last five rows of dataframe\n",
        "data_frame.tail()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 334
        },
        "id": "NG6C3HP-105U",
        "outputId": "f937aeae-676f-4fe9-854d-a7b256c7398c"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "     mean radius  mean texture  mean perimeter  mean area  mean smoothness  \\\n",
              "564        21.56         22.39          142.00     1479.0          0.11100   \n",
              "565        20.13         28.25          131.20     1261.0          0.09780   \n",
              "566        16.60         28.08          108.30      858.1          0.08455   \n",
              "567        20.60         29.33          140.10     1265.0          0.11780   \n",
              "568         7.76         24.54           47.92      181.0          0.05263   \n",
              "\n",
              "     mean compactness  mean concavity  mean concave points  mean symmetry  \\\n",
              "564           0.11590         0.24390              0.13890         0.1726   \n",
              "565           0.10340         0.14400              0.09791         0.1752   \n",
              "566           0.10230         0.09251              0.05302         0.1590   \n",
              "567           0.27700         0.35140              0.15200         0.2397   \n",
              "568           0.04362         0.00000              0.00000         0.1587   \n",
              "\n",
              "     mean fractal dimension  ...  worst texture  worst perimeter  worst area  \\\n",
              "564                 0.05623  ...          26.40           166.10      2027.0   \n",
              "565                 0.05533  ...          38.25           155.00      1731.0   \n",
              "566                 0.05648  ...          34.12           126.70      1124.0   \n",
              "567                 0.07016  ...          39.42           184.60      1821.0   \n",
              "568                 0.05884  ...          30.37            59.16       268.6   \n",
              "\n",
              "     worst smoothness  worst compactness  worst concavity  \\\n",
              "564           0.14100            0.21130           0.4107   \n",
              "565           0.11660            0.19220           0.3215   \n",
              "566           0.11390            0.30940           0.3403   \n",
              "567           0.16500            0.86810           0.9387   \n",
              "568           0.08996            0.06444           0.0000   \n",
              "\n",
              "     worst concave points  worst symmetry  worst fractal dimension  label  \n",
              "564                0.2216          0.2060                  0.07115      0  \n",
              "565                0.1628          0.2572                  0.06637      0  \n",
              "566                0.1418          0.2218                  0.07820      0  \n",
              "567                0.2650          0.4087                  0.12400      0  \n",
              "568                0.0000          0.2871                  0.07039      1  \n",
              "\n",
              "[5 rows x 31 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-0fa80e62-4a88-492a-b18a-cb94693ca2d4\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>mean radius</th>\n",
              "      <th>mean texture</th>\n",
              "      <th>mean perimeter</th>\n",
              "      <th>mean area</th>\n",
              "      <th>mean smoothness</th>\n",
              "      <th>mean compactness</th>\n",
              "      <th>mean concavity</th>\n",
              "      <th>mean concave points</th>\n",
              "      <th>mean symmetry</th>\n",
              "      <th>mean fractal dimension</th>\n",
              "      <th>...</th>\n",
              "      <th>worst texture</th>\n",
              "      <th>worst perimeter</th>\n",
              "      <th>worst area</th>\n",
              "      <th>worst smoothness</th>\n",
              "      <th>worst compactness</th>\n",
              "      <th>worst concavity</th>\n",
              "      <th>worst concave points</th>\n",
              "      <th>worst symmetry</th>\n",
              "      <th>worst fractal dimension</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>564</th>\n",
              "      <td>21.56</td>\n",
              "      <td>22.39</td>\n",
              "      <td>142.00</td>\n",
              "      <td>1479.0</td>\n",
              "      <td>0.11100</td>\n",
              "      <td>0.11590</td>\n",
              "      <td>0.24390</td>\n",
              "      <td>0.13890</td>\n",
              "      <td>0.1726</td>\n",
              "      <td>0.05623</td>\n",
              "      <td>...</td>\n",
              "      <td>26.40</td>\n",
              "      <td>166.10</td>\n",
              "      <td>2027.0</td>\n",
              "      <td>0.14100</td>\n",
              "      <td>0.21130</td>\n",
              "      <td>0.4107</td>\n",
              "      <td>0.2216</td>\n",
              "      <td>0.2060</td>\n",
              "      <td>0.07115</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>565</th>\n",
              "      <td>20.13</td>\n",
              "      <td>28.25</td>\n",
              "      <td>131.20</td>\n",
              "      <td>1261.0</td>\n",
              "      <td>0.09780</td>\n",
              "      <td>0.10340</td>\n",
              "      <td>0.14400</td>\n",
              "      <td>0.09791</td>\n",
              "      <td>0.1752</td>\n",
              "      <td>0.05533</td>\n",
              "      <td>...</td>\n",
              "      <td>38.25</td>\n",
              "      <td>155.00</td>\n",
              "      <td>1731.0</td>\n",
              "      <td>0.11660</td>\n",
              "      <td>0.19220</td>\n",
              "      <td>0.3215</td>\n",
              "      <td>0.1628</td>\n",
              "      <td>0.2572</td>\n",
              "      <td>0.06637</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>566</th>\n",
              "      <td>16.60</td>\n",
              "      <td>28.08</td>\n",
              "      <td>108.30</td>\n",
              "      <td>858.1</td>\n",
              "      <td>0.08455</td>\n",
              "      <td>0.10230</td>\n",
              "      <td>0.09251</td>\n",
              "      <td>0.05302</td>\n",
              "      <td>0.1590</td>\n",
              "      <td>0.05648</td>\n",
              "      <td>...</td>\n",
              "      <td>34.12</td>\n",
              "      <td>126.70</td>\n",
              "      <td>1124.0</td>\n",
              "      <td>0.11390</td>\n",
              "      <td>0.30940</td>\n",
              "      <td>0.3403</td>\n",
              "      <td>0.1418</td>\n",
              "      <td>0.2218</td>\n",
              "      <td>0.07820</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>567</th>\n",
              "      <td>20.60</td>\n",
              "      <td>29.33</td>\n",
              "      <td>140.10</td>\n",
              "      <td>1265.0</td>\n",
              "      <td>0.11780</td>\n",
              "      <td>0.27700</td>\n",
              "      <td>0.35140</td>\n",
              "      <td>0.15200</td>\n",
              "      <td>0.2397</td>\n",
              "      <td>0.07016</td>\n",
              "      <td>...</td>\n",
              "      <td>39.42</td>\n",
              "      <td>184.60</td>\n",
              "      <td>1821.0</td>\n",
              "      <td>0.16500</td>\n",
              "      <td>0.86810</td>\n",
              "      <td>0.9387</td>\n",
              "      <td>0.2650</td>\n",
              "      <td>0.4087</td>\n",
              "      <td>0.12400</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>568</th>\n",
              "      <td>7.76</td>\n",
              "      <td>24.54</td>\n",
              "      <td>47.92</td>\n",
              "      <td>181.0</td>\n",
              "      <td>0.05263</td>\n",
              "      <td>0.04362</td>\n",
              "      <td>0.00000</td>\n",
              "      <td>0.00000</td>\n",
              "      <td>0.1587</td>\n",
              "      <td>0.05884</td>\n",
              "      <td>...</td>\n",
              "      <td>30.37</td>\n",
              "      <td>59.16</td>\n",
              "      <td>268.6</td>\n",
              "      <td>0.08996</td>\n",
              "      <td>0.06444</td>\n",
              "      <td>0.0000</td>\n",
              "      <td>0.0000</td>\n",
              "      <td>0.2871</td>\n",
              "      <td>0.07039</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 31 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-0fa80e62-4a88-492a-b18a-cb94693ca2d4')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-0fa80e62-4a88-492a-b18a-cb94693ca2d4 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-0fa80e62-4a88-492a-b18a-cb94693ca2d4');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# getting some information about the data\n",
        "data_frame.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Gkm5l1Uw18j-",
        "outputId": "0d334d33-1fc3-401c-fbfd-e930c343236a"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 569 entries, 0 to 568\n",
            "Data columns (total 31 columns):\n",
            " #   Column                   Non-Null Count  Dtype  \n",
            "---  ------                   --------------  -----  \n",
            " 0   mean radius              569 non-null    float64\n",
            " 1   mean texture             569 non-null    float64\n",
            " 2   mean perimeter           569 non-null    float64\n",
            " 3   mean area                569 non-null    float64\n",
            " 4   mean smoothness          569 non-null    float64\n",
            " 5   mean compactness         569 non-null    float64\n",
            " 6   mean concavity           569 non-null    float64\n",
            " 7   mean concave points      569 non-null    float64\n",
            " 8   mean symmetry            569 non-null    float64\n",
            " 9   mean fractal dimension   569 non-null    float64\n",
            " 10  radius error             569 non-null    float64\n",
            " 11  texture error            569 non-null    float64\n",
            " 12  perimeter error          569 non-null    float64\n",
            " 13  area error               569 non-null    float64\n",
            " 14  smoothness error         569 non-null    float64\n",
            " 15  compactness error        569 non-null    float64\n",
            " 16  concavity error          569 non-null    float64\n",
            " 17  concave points error     569 non-null    float64\n",
            " 18  symmetry error           569 non-null    float64\n",
            " 19  fractal dimension error  569 non-null    float64\n",
            " 20  worst radius             569 non-null    float64\n",
            " 21  worst texture            569 non-null    float64\n",
            " 22  worst perimeter          569 non-null    float64\n",
            " 23  worst area               569 non-null    float64\n",
            " 24  worst smoothness         569 non-null    float64\n",
            " 25  worst compactness        569 non-null    float64\n",
            " 26  worst concavity          569 non-null    float64\n",
            " 27  worst concave points     569 non-null    float64\n",
            " 28  worst symmetry           569 non-null    float64\n",
            " 29  worst fractal dimension  569 non-null    float64\n",
            " 30  label                    569 non-null    int64  \n",
            "dtypes: float64(30), int64(1)\n",
            "memory usage: 137.9 KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# checking the missing values in thedata frame\n",
        "data_frame.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zzVngovu2FJl",
        "outputId": "27c55177-1059-4573-fe5e-0f3d617919c3"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "mean radius                0\n",
              "mean texture               0\n",
              "mean perimeter             0\n",
              "mean area                  0\n",
              "mean smoothness            0\n",
              "mean compactness           0\n",
              "mean concavity             0\n",
              "mean concave points        0\n",
              "mean symmetry              0\n",
              "mean fractal dimension     0\n",
              "radius error               0\n",
              "texture error              0\n",
              "perimeter error            0\n",
              "area error                 0\n",
              "smoothness error           0\n",
              "compactness error          0\n",
              "concavity error            0\n",
              "concave points error       0\n",
              "symmetry error             0\n",
              "fractal dimension error    0\n",
              "worst radius               0\n",
              "worst texture              0\n",
              "worst perimeter            0\n",
              "worst area                 0\n",
              "worst smoothness           0\n",
              "worst compactness          0\n",
              "worst concavity            0\n",
              "worst concave points       0\n",
              "worst symmetry             0\n",
              "worst fractal dimension    0\n",
              "label                      0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# statistical measure about the data\n",
        "data_frame.describe()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 428
        },
        "id": "BnfiJXUy2QLl",
        "outputId": "66175560-2307-42d5-da13-a2e77737810b"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "       mean radius  mean texture  mean perimeter    mean area  \\\n",
              "count   569.000000    569.000000      569.000000   569.000000   \n",
              "mean     14.127292     19.289649       91.969033   654.889104   \n",
              "std       3.524049      4.301036       24.298981   351.914129   \n",
              "min       6.981000      9.710000       43.790000   143.500000   \n",
              "25%      11.700000     16.170000       75.170000   420.300000   \n",
              "50%      13.370000     18.840000       86.240000   551.100000   \n",
              "75%      15.780000     21.800000      104.100000   782.700000   \n",
              "max      28.110000     39.280000      188.500000  2501.000000   \n",
              "\n",
              "       mean smoothness  mean compactness  mean concavity  mean concave points  \\\n",
              "count       569.000000        569.000000      569.000000           569.000000   \n",
              "mean          0.096360          0.104341        0.088799             0.048919   \n",
              "std           0.014064          0.052813        0.079720             0.038803   \n",
              "min           0.052630          0.019380        0.000000             0.000000   \n",
              "25%           0.086370          0.064920        0.029560             0.020310   \n",
              "50%           0.095870          0.092630        0.061540             0.033500   \n",
              "75%           0.105300          0.130400        0.130700             0.074000   \n",
              "max           0.163400          0.345400        0.426800             0.201200   \n",
              "\n",
              "       mean symmetry  mean fractal dimension  ...  worst texture  \\\n",
              "count     569.000000              569.000000  ...     569.000000   \n",
              "mean        0.181162                0.062798  ...      25.677223   \n",
              "std         0.027414                0.007060  ...       6.146258   \n",
              "min         0.106000                0.049960  ...      12.020000   \n",
              "25%         0.161900                0.057700  ...      21.080000   \n",
              "50%         0.179200                0.061540  ...      25.410000   \n",
              "75%         0.195700                0.066120  ...      29.720000   \n",
              "max         0.304000                0.097440  ...      49.540000   \n",
              "\n",
              "       worst perimeter   worst area  worst smoothness  worst compactness  \\\n",
              "count       569.000000   569.000000        569.000000         569.000000   \n",
              "mean        107.261213   880.583128          0.132369           0.254265   \n",
              "std          33.602542   569.356993          0.022832           0.157336   \n",
              "min          50.410000   185.200000          0.071170           0.027290   \n",
              "25%          84.110000   515.300000          0.116600           0.147200   \n",
              "50%          97.660000   686.500000          0.131300           0.211900   \n",
              "75%         125.400000  1084.000000          0.146000           0.339100   \n",
              "max         251.200000  4254.000000          0.222600           1.058000   \n",
              "\n",
              "       worst concavity  worst concave points  worst symmetry  \\\n",
              "count       569.000000            569.000000      569.000000   \n",
              "mean          0.272188              0.114606        0.290076   \n",
              "std           0.208624              0.065732        0.061867   \n",
              "min           0.000000              0.000000        0.156500   \n",
              "25%           0.114500              0.064930        0.250400   \n",
              "50%           0.226700              0.099930        0.282200   \n",
              "75%           0.382900              0.161400        0.317900   \n",
              "max           1.252000              0.291000        0.663800   \n",
              "\n",
              "       worst fractal dimension       label  \n",
              "count               569.000000  569.000000  \n",
              "mean                  0.083946    0.627417  \n",
              "std                   0.018061    0.483918  \n",
              "min                   0.055040    0.000000  \n",
              "25%                   0.071460    0.000000  \n",
              "50%                   0.080040    1.000000  \n",
              "75%                   0.092080    1.000000  \n",
              "max                   0.207500    1.000000  \n",
              "\n",
              "[8 rows x 31 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-32087afb-e1f4-4f61-b49e-4794c9bbde7c\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>mean radius</th>\n",
              "      <th>mean texture</th>\n",
              "      <th>mean perimeter</th>\n",
              "      <th>mean area</th>\n",
              "      <th>mean smoothness</th>\n",
              "      <th>mean compactness</th>\n",
              "      <th>mean concavity</th>\n",
              "      <th>mean concave points</th>\n",
              "      <th>mean symmetry</th>\n",
              "      <th>mean fractal dimension</th>\n",
              "      <th>...</th>\n",
              "      <th>worst texture</th>\n",
              "      <th>worst perimeter</th>\n",
              "      <th>worst area</th>\n",
              "      <th>worst smoothness</th>\n",
              "      <th>worst compactness</th>\n",
              "      <th>worst concavity</th>\n",
              "      <th>worst concave points</th>\n",
              "      <th>worst symmetry</th>\n",
              "      <th>worst fractal dimension</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>569.000000</td>\n",
              "      <td>569.000000</td>\n",
              "      <td>569.000000</td>\n",
              "      <td>569.000000</td>\n",
              "      <td>569.000000</td>\n",
              "      <td>569.000000</td>\n",
              "      <td>569.000000</td>\n",
              "      <td>569.000000</td>\n",
              "      <td>569.000000</td>\n",
              "      <td>569.000000</td>\n",
              "      <td>...</td>\n",
              "      <td>569.000000</td>\n",
              "      <td>569.000000</td>\n",
              "      <td>569.000000</td>\n",
              "      <td>569.000000</td>\n",
              "      <td>569.000000</td>\n",
              "      <td>569.000000</td>\n",
              "      <td>569.000000</td>\n",
              "      <td>569.000000</td>\n",
              "      <td>569.000000</td>\n",
              "      <td>569.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>14.127292</td>\n",
              "      <td>19.289649</td>\n",
              "      <td>91.969033</td>\n",
              "      <td>654.889104</td>\n",
              "      <td>0.096360</td>\n",
              "      <td>0.104341</td>\n",
              "      <td>0.088799</td>\n",
              "      <td>0.048919</td>\n",
              "      <td>0.181162</td>\n",
              "      <td>0.062798</td>\n",
              "      <td>...</td>\n",
              "      <td>25.677223</td>\n",
              "      <td>107.261213</td>\n",
              "      <td>880.583128</td>\n",
              "      <td>0.132369</td>\n",
              "      <td>0.254265</td>\n",
              "      <td>0.272188</td>\n",
              "      <td>0.114606</td>\n",
              "      <td>0.290076</td>\n",
              "      <td>0.083946</td>\n",
              "      <td>0.627417</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>3.524049</td>\n",
              "      <td>4.301036</td>\n",
              "      <td>24.298981</td>\n",
              "      <td>351.914129</td>\n",
              "      <td>0.014064</td>\n",
              "      <td>0.052813</td>\n",
              "      <td>0.079720</td>\n",
              "      <td>0.038803</td>\n",
              "      <td>0.027414</td>\n",
              "      <td>0.007060</td>\n",
              "      <td>...</td>\n",
              "      <td>6.146258</td>\n",
              "      <td>33.602542</td>\n",
              "      <td>569.356993</td>\n",
              "      <td>0.022832</td>\n",
              "      <td>0.157336</td>\n",
              "      <td>0.208624</td>\n",
              "      <td>0.065732</td>\n",
              "      <td>0.061867</td>\n",
              "      <td>0.018061</td>\n",
              "      <td>0.483918</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>6.981000</td>\n",
              "      <td>9.710000</td>\n",
              "      <td>43.790000</td>\n",
              "      <td>143.500000</td>\n",
              "      <td>0.052630</td>\n",
              "      <td>0.019380</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.106000</td>\n",
              "      <td>0.049960</td>\n",
              "      <td>...</td>\n",
              "      <td>12.020000</td>\n",
              "      <td>50.410000</td>\n",
              "      <td>185.200000</td>\n",
              "      <td>0.071170</td>\n",
              "      <td>0.027290</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.156500</td>\n",
              "      <td>0.055040</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>11.700000</td>\n",
              "      <td>16.170000</td>\n",
              "      <td>75.170000</td>\n",
              "      <td>420.300000</td>\n",
              "      <td>0.086370</td>\n",
              "      <td>0.064920</td>\n",
              "      <td>0.029560</td>\n",
              "      <td>0.020310</td>\n",
              "      <td>0.161900</td>\n",
              "      <td>0.057700</td>\n",
              "      <td>...</td>\n",
              "      <td>21.080000</td>\n",
              "      <td>84.110000</td>\n",
              "      <td>515.300000</td>\n",
              "      <td>0.116600</td>\n",
              "      <td>0.147200</td>\n",
              "      <td>0.114500</td>\n",
              "      <td>0.064930</td>\n",
              "      <td>0.250400</td>\n",
              "      <td>0.071460</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>13.370000</td>\n",
              "      <td>18.840000</td>\n",
              "      <td>86.240000</td>\n",
              "      <td>551.100000</td>\n",
              "      <td>0.095870</td>\n",
              "      <td>0.092630</td>\n",
              "      <td>0.061540</td>\n",
              "      <td>0.033500</td>\n",
              "      <td>0.179200</td>\n",
              "      <td>0.061540</td>\n",
              "      <td>...</td>\n",
              "      <td>25.410000</td>\n",
              "      <td>97.660000</td>\n",
              "      <td>686.500000</td>\n",
              "      <td>0.131300</td>\n",
              "      <td>0.211900</td>\n",
              "      <td>0.226700</td>\n",
              "      <td>0.099930</td>\n",
              "      <td>0.282200</td>\n",
              "      <td>0.080040</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>15.780000</td>\n",
              "      <td>21.800000</td>\n",
              "      <td>104.100000</td>\n",
              "      <td>782.700000</td>\n",
              "      <td>0.105300</td>\n",
              "      <td>0.130400</td>\n",
              "      <td>0.130700</td>\n",
              "      <td>0.074000</td>\n",
              "      <td>0.195700</td>\n",
              "      <td>0.066120</td>\n",
              "      <td>...</td>\n",
              "      <td>29.720000</td>\n",
              "      <td>125.400000</td>\n",
              "      <td>1084.000000</td>\n",
              "      <td>0.146000</td>\n",
              "      <td>0.339100</td>\n",
              "      <td>0.382900</td>\n",
              "      <td>0.161400</td>\n",
              "      <td>0.317900</td>\n",
              "      <td>0.092080</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>28.110000</td>\n",
              "      <td>39.280000</td>\n",
              "      <td>188.500000</td>\n",
              "      <td>2501.000000</td>\n",
              "      <td>0.163400</td>\n",
              "      <td>0.345400</td>\n",
              "      <td>0.426800</td>\n",
              "      <td>0.201200</td>\n",
              "      <td>0.304000</td>\n",
              "      <td>0.097440</td>\n",
              "      <td>...</td>\n",
              "      <td>49.540000</td>\n",
              "      <td>251.200000</td>\n",
              "      <td>4254.000000</td>\n",
              "      <td>0.222600</td>\n",
              "      <td>1.058000</td>\n",
              "      <td>1.252000</td>\n",
              "      <td>0.291000</td>\n",
              "      <td>0.663800</td>\n",
              "      <td>0.207500</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>8 rows × 31 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-32087afb-e1f4-4f61-b49e-4794c9bbde7c')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-32087afb-e1f4-4f61-b49e-4794c9bbde7c button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-32087afb-e1f4-4f61-b49e-4794c9bbde7c');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# checking the distribution of target data varible\n",
        "data_frame['label'].value_counts()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "m_evIAPY2g-s",
        "outputId": "2e0f2e86-dac4-4524-c585-dd9bbaba5a49"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1    357\n",
              "0    212\n",
              "Name: label, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 11
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "1 -> Benign\n",
        "\n",
        "2 -> Malignant"
      ],
      "metadata": {
        "id": "VZ6E7ko53A8d"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "data_frame.groupby('label').mean()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 271
        },
        "id": "8mhCTclz3Lbd",
        "outputId": "457e94db-4fa9-40dd-c502-e6604c335204"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "       mean radius  mean texture  mean perimeter   mean area  mean smoothness  \\\n",
              "label                                                                           \n",
              "0        17.462830     21.604906      115.365377  978.376415         0.102898   \n",
              "1        12.146524     17.914762       78.075406  462.790196         0.092478   \n",
              "\n",
              "       mean compactness  mean concavity  mean concave points  mean symmetry  \\\n",
              "label                                                                         \n",
              "0              0.145188        0.160775             0.087990       0.192909   \n",
              "1              0.080085        0.046058             0.025717       0.174186   \n",
              "\n",
              "       mean fractal dimension  ...  worst radius  worst texture  \\\n",
              "label                          ...                                \n",
              "0                    0.062680  ...     21.134811      29.318208   \n",
              "1                    0.062867  ...     13.379801      23.515070   \n",
              "\n",
              "       worst perimeter   worst area  worst smoothness  worst compactness  \\\n",
              "label                                                                      \n",
              "0           141.370330  1422.286321          0.144845           0.374824   \n",
              "1            87.005938   558.899440          0.124959           0.182673   \n",
              "\n",
              "       worst concavity  worst concave points  worst symmetry  \\\n",
              "label                                                          \n",
              "0             0.450606              0.182237        0.323468   \n",
              "1             0.166238              0.074444        0.270246   \n",
              "\n",
              "       worst fractal dimension  \n",
              "label                           \n",
              "0                     0.091530  \n",
              "1                     0.079442  \n",
              "\n",
              "[2 rows x 30 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-86f92e57-5daa-4c26-98ff-16062a3c0cad\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>mean radius</th>\n",
              "      <th>mean texture</th>\n",
              "      <th>mean perimeter</th>\n",
              "      <th>mean area</th>\n",
              "      <th>mean smoothness</th>\n",
              "      <th>mean compactness</th>\n",
              "      <th>mean concavity</th>\n",
              "      <th>mean concave points</th>\n",
              "      <th>mean symmetry</th>\n",
              "      <th>mean fractal dimension</th>\n",
              "      <th>...</th>\n",
              "      <th>worst radius</th>\n",
              "      <th>worst texture</th>\n",
              "      <th>worst perimeter</th>\n",
              "      <th>worst area</th>\n",
              "      <th>worst smoothness</th>\n",
              "      <th>worst compactness</th>\n",
              "      <th>worst concavity</th>\n",
              "      <th>worst concave points</th>\n",
              "      <th>worst symmetry</th>\n",
              "      <th>worst fractal dimension</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>label</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>17.462830</td>\n",
              "      <td>21.604906</td>\n",
              "      <td>115.365377</td>\n",
              "      <td>978.376415</td>\n",
              "      <td>0.102898</td>\n",
              "      <td>0.145188</td>\n",
              "      <td>0.160775</td>\n",
              "      <td>0.087990</td>\n",
              "      <td>0.192909</td>\n",
              "      <td>0.062680</td>\n",
              "      <td>...</td>\n",
              "      <td>21.134811</td>\n",
              "      <td>29.318208</td>\n",
              "      <td>141.370330</td>\n",
              "      <td>1422.286321</td>\n",
              "      <td>0.144845</td>\n",
              "      <td>0.374824</td>\n",
              "      <td>0.450606</td>\n",
              "      <td>0.182237</td>\n",
              "      <td>0.323468</td>\n",
              "      <td>0.091530</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>12.146524</td>\n",
              "      <td>17.914762</td>\n",
              "      <td>78.075406</td>\n",
              "      <td>462.790196</td>\n",
              "      <td>0.092478</td>\n",
              "      <td>0.080085</td>\n",
              "      <td>0.046058</td>\n",
              "      <td>0.025717</td>\n",
              "      <td>0.174186</td>\n",
              "      <td>0.062867</td>\n",
              "      <td>...</td>\n",
              "      <td>13.379801</td>\n",
              "      <td>23.515070</td>\n",
              "      <td>87.005938</td>\n",
              "      <td>558.899440</td>\n",
              "      <td>0.124959</td>\n",
              "      <td>0.182673</td>\n",
              "      <td>0.166238</td>\n",
              "      <td>0.074444</td>\n",
              "      <td>0.270246</td>\n",
              "      <td>0.079442</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>2 rows × 30 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-86f92e57-5daa-4c26-98ff-16062a3c0cad')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-86f92e57-5daa-4c26-98ff-16062a3c0cad button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-86f92e57-5daa-4c26-98ff-16062a3c0cad');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Seprating the features and target\n"
      ],
      "metadata": {
        "id": "4JY70zSN3JKr"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "x = data_frame.drop(columns='label', axis=1)\n",
        "y = data_frame['label']"
      ],
      "metadata": {
        "id": "4Xlk8wRm3ab4"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(x)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "w1ggz4f43kSS",
        "outputId": "91821b84-6e74-475e-d45b-e798e5ade9fe"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "     mean radius  mean texture  mean perimeter  mean area  mean smoothness  \\\n",
            "0          17.99         10.38          122.80     1001.0          0.11840   \n",
            "1          20.57         17.77          132.90     1326.0          0.08474   \n",
            "2          19.69         21.25          130.00     1203.0          0.10960   \n",
            "3          11.42         20.38           77.58      386.1          0.14250   \n",
            "4          20.29         14.34          135.10     1297.0          0.10030   \n",
            "..           ...           ...             ...        ...              ...   \n",
            "564        21.56         22.39          142.00     1479.0          0.11100   \n",
            "565        20.13         28.25          131.20     1261.0          0.09780   \n",
            "566        16.60         28.08          108.30      858.1          0.08455   \n",
            "567        20.60         29.33          140.10     1265.0          0.11780   \n",
            "568         7.76         24.54           47.92      181.0          0.05263   \n",
            "\n",
            "     mean compactness  mean concavity  mean concave points  mean symmetry  \\\n",
            "0             0.27760         0.30010              0.14710         0.2419   \n",
            "1             0.07864         0.08690              0.07017         0.1812   \n",
            "2             0.15990         0.19740              0.12790         0.2069   \n",
            "3             0.28390         0.24140              0.10520         0.2597   \n",
            "4             0.13280         0.19800              0.10430         0.1809   \n",
            "..                ...             ...                  ...            ...   \n",
            "564           0.11590         0.24390              0.13890         0.1726   \n",
            "565           0.10340         0.14400              0.09791         0.1752   \n",
            "566           0.10230         0.09251              0.05302         0.1590   \n",
            "567           0.27700         0.35140              0.15200         0.2397   \n",
            "568           0.04362         0.00000              0.00000         0.1587   \n",
            "\n",
            "     mean fractal dimension  ...  worst radius  worst texture  \\\n",
            "0                   0.07871  ...        25.380          17.33   \n",
            "1                   0.05667  ...        24.990          23.41   \n",
            "2                   0.05999  ...        23.570          25.53   \n",
            "3                   0.09744  ...        14.910          26.50   \n",
            "4                   0.05883  ...        22.540          16.67   \n",
            "..                      ...  ...           ...            ...   \n",
            "564                 0.05623  ...        25.450          26.40   \n",
            "565                 0.05533  ...        23.690          38.25   \n",
            "566                 0.05648  ...        18.980          34.12   \n",
            "567                 0.07016  ...        25.740          39.42   \n",
            "568                 0.05884  ...         9.456          30.37   \n",
            "\n",
            "     worst perimeter  worst area  worst smoothness  worst compactness  \\\n",
            "0             184.60      2019.0           0.16220            0.66560   \n",
            "1             158.80      1956.0           0.12380            0.18660   \n",
            "2             152.50      1709.0           0.14440            0.42450   \n",
            "3              98.87       567.7           0.20980            0.86630   \n",
            "4             152.20      1575.0           0.13740            0.20500   \n",
            "..               ...         ...               ...                ...   \n",
            "564           166.10      2027.0           0.14100            0.21130   \n",
            "565           155.00      1731.0           0.11660            0.19220   \n",
            "566           126.70      1124.0           0.11390            0.30940   \n",
            "567           184.60      1821.0           0.16500            0.86810   \n",
            "568            59.16       268.6           0.08996            0.06444   \n",
            "\n",
            "     worst concavity  worst concave points  worst symmetry  \\\n",
            "0             0.7119                0.2654          0.4601   \n",
            "1             0.2416                0.1860          0.2750   \n",
            "2             0.4504                0.2430          0.3613   \n",
            "3             0.6869                0.2575          0.6638   \n",
            "4             0.4000                0.1625          0.2364   \n",
            "..               ...                   ...             ...   \n",
            "564           0.4107                0.2216          0.2060   \n",
            "565           0.3215                0.1628          0.2572   \n",
            "566           0.3403                0.1418          0.2218   \n",
            "567           0.9387                0.2650          0.4087   \n",
            "568           0.0000                0.0000          0.2871   \n",
            "\n",
            "     worst fractal dimension  \n",
            "0                    0.11890  \n",
            "1                    0.08902  \n",
            "2                    0.08758  \n",
            "3                    0.17300  \n",
            "4                    0.07678  \n",
            "..                       ...  \n",
            "564                  0.07115  \n",
            "565                  0.06637  \n",
            "566                  0.07820  \n",
            "567                  0.12400  \n",
            "568                  0.07039  \n",
            "\n",
            "[569 rows x 30 columns]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(y)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xIfMkAne3lJ1",
        "outputId": "49507a6f-0e97-4a3d-adf7-f9ad06f59839"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0      0\n",
            "1      0\n",
            "2      0\n",
            "3      0\n",
            "4      0\n",
            "      ..\n",
            "564    0\n",
            "565    0\n",
            "566    0\n",
            "567    0\n",
            "568    1\n",
            "Name: label, Length: 569, dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "splitting the data into training data and testing data"
      ],
      "metadata": {
        "id": "rUYZGxeD3lx7"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.2, random_state=2)"
      ],
      "metadata": {
        "id": "_Bq3e85N3u39"
      },
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(x.shape, x_test.shape, x_train.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RELNkCdb39pG",
        "outputId": "fff0e053-b4f8-45bd-bf93-b06e56ab100a"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(569, 30) (114, 30) (455, 30)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "stadardized the data"
      ],
      "metadata": {
        "id": "Y3xHFYV-4E50"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.preprocessing import StandardScaler"
      ],
      "metadata": {
        "id": "SdiHTw5m4Lky"
      },
      "execution_count": 20,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "scaler = StandardScaler()"
      ],
      "metadata": {
        "id": "N_DLymr94RvC"
      },
      "execution_count": 23,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x_train_std = scaler.fit_transform(x_train)\n",
        "x_test_std = scaler.fit_transform(x_test)"
      ],
      "metadata": {
        "id": "6KDtFZrX4WAt"
      },
      "execution_count": 24,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(x_train_std)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "b457VHOS4lIH",
        "outputId": "85d3d7b9-1961-4143-ea25-d61a815f59b8"
      },
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[-0.01330339  1.7757658  -0.01491962 ... -0.13236958 -1.08014517\n",
            "  -0.03527943]\n",
            " [-0.8448276  -0.6284278  -0.87702746 ... -1.11552632 -0.85773964\n",
            "  -0.72098905]\n",
            " [ 1.44755936  0.71180168  1.47428816 ...  0.87583964  0.4967602\n",
            "   0.46321706]\n",
            " ...\n",
            " [-0.46608541 -1.49375484 -0.53234924 ... -1.32388956 -1.02997851\n",
            "  -0.75145272]\n",
            " [-0.50025764 -1.62161319 -0.527814   ... -0.0987626   0.35796577\n",
            "  -0.43906159]\n",
            " [ 0.96060511  1.21181916  1.00427242 ...  0.8956983  -1.23064515\n",
            "   0.50697397]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(x_test_std)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "n9XJNMne4seS",
        "outputId": "3b9cb711-9176-4ebb-e72b-2aadf6dba4eb"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[-0.08700339 -1.47192915 -0.10537391 ... -0.26629174 -1.09776353\n",
            "  -0.65597459]\n",
            " [ 0.19989092  0.03577342  0.1706179  ...  0.44844054  0.06066588\n",
            "   0.02108157]\n",
            " [-1.28858427 -0.21847659 -1.30667757 ... -1.41981535  0.19788632\n",
            "  -0.31050377]\n",
            " ...\n",
            " [ 0.67523542  0.61546345  0.70329853 ...  1.36221218  1.000987\n",
            "   0.62759948]\n",
            " [ 0.20832899  1.5866985   0.10942329 ... -1.35965118 -1.95719681\n",
            "  -1.62740299]\n",
            " [ 0.78774299  0.03068842  0.84293725 ...  2.03773974  0.27299646\n",
            "   0.34822356]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Building the neural network**"
      ],
      "metadata": {
        "id": "6RcqCRTp4uP0"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# import tensorflow and keras\n",
        "import tensorflow as tf\n",
        "tf.random.set_seed(3)\n",
        "from tensorflow import keras"
      ],
      "metadata": {
        "id": "AAm4mbYc44-J"
      },
      "execution_count": 30,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# setting up the layers of Neural network\n",
        "\n",
        "model = keras.Sequential([\n",
        "                        keras.layers.Flatten(input_shape=(30,)),\n",
        "                        keras.layers.Dense(20, activation = 'relu'),\n",
        "                        keras.layers.Dense(2, activation='sigmoid')\n",
        "                      \n",
        "])"
      ],
      "metadata": {
        "id": "xdgBscgN5FoL"
      },
      "execution_count": 32,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# compiling the neural network\n",
        "\n",
        "model.compile(optimizer='adam',\n",
        "              loss='sparse_categorical_crossentropy',\n",
        "              metrics=['accuracy'])"
      ],
      "metadata": {
        "id": "Kz6J-FE-6auR"
      },
      "execution_count": 33,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# training the neural network\n",
        "history = model.fit(x_train_std, y_train, validation_split=0.2, epochs=10)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "D37RDWuV614a",
        "outputId": "bb317407-626c-482a-bf8a-f13d5a9310a6"
      },
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/10\n",
            "12/12 [==============================] - 1s 27ms/step - loss: 0.9896 - accuracy: 0.3929 - val_loss: 0.7633 - val_accuracy: 0.5275\n",
            "Epoch 2/10\n",
            "12/12 [==============================] - 0s 6ms/step - loss: 0.7008 - accuracy: 0.6374 - val_loss: 0.5512 - val_accuracy: 0.7912\n",
            "Epoch 3/10\n",
            "12/12 [==============================] - 0s 6ms/step - loss: 0.5224 - accuracy: 0.8049 - val_loss: 0.4186 - val_accuracy: 0.8571\n",
            "Epoch 4/10\n",
            "12/12 [==============================] - 0s 6ms/step - loss: 0.4031 - accuracy: 0.8846 - val_loss: 0.3344 - val_accuracy: 0.9011\n",
            "Epoch 5/10\n",
            "12/12 [==============================] - 0s 7ms/step - loss: 0.3290 - accuracy: 0.9066 - val_loss: 0.2815 - val_accuracy: 0.9011\n",
            "Epoch 6/10\n",
            "12/12 [==============================] - 0s 8ms/step - loss: 0.2809 - accuracy: 0.9203 - val_loss: 0.2427 - val_accuracy: 0.9011\n",
            "Epoch 7/10\n",
            "12/12 [==============================] - 0s 7ms/step - loss: 0.2408 - accuracy: 0.9286 - val_loss: 0.2158 - val_accuracy: 0.9231\n",
            "Epoch 8/10\n",
            "12/12 [==============================] - 0s 6ms/step - loss: 0.2137 - accuracy: 0.9313 - val_loss: 0.1938 - val_accuracy: 0.9231\n",
            "Epoch 9/10\n",
            "12/12 [==============================] - 0s 7ms/step - loss: 0.1909 - accuracy: 0.9505 - val_loss: 0.1769 - val_accuracy: 0.9341\n",
            "Epoch 10/10\n",
            "12/12 [==============================] - 0s 6ms/step - loss: 0.1724 - accuracy: 0.9533 - val_loss: 0.1622 - val_accuracy: 0.9231\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Visulizing accuracy and loss\n"
      ],
      "metadata": {
        "id": "HtS7PPHb7BiU"
      },
      "execution_count": 35,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "plt.plot(history.history['accuracy'])\n",
        "plt.plot(history.history['val_accuracy'])\n",
        "\n",
        "plt.title('model accuracy')\n",
        "plt.ylabel('accuray')\n",
        "plt.xlabel('epoch')\n",
        "\n",
        "plt.legend(['trainig data', 'validation data'], loc = 'lower right')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 312
        },
        "id": "hxMLRLkn7Jjl",
        "outputId": "8e2f7081-0fc4-4a5b-bf95-0eea35658925"
      },
      "execution_count": 37,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.legend.Legend at 0x7f116685df70>"
            ]
          },
          "metadata": {},
          "execution_count": 37
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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dWVlZTzxxBOdWvfCCy+krKzssMvcf//9vPPOO53aPsBZZ51FTk7OYZf5wx/+QHV1daf3ESiWCDzyDlTz7637uTIzlfAwOx02JtgOlwgaGxsPu+6yZcvo27fvYZd58MEHOffcczsbnk+6SyKwUUMeL+bkIQJXZtplIWPa+vnrG9hYeNCv2xw9OJ6fXTymw/n33nsvX375JRMmTOC8887joosu4qc//SmJiYls3ryZL774gksuuYS8vDxqa2u54447mD9/PgDp6enk5ORQWVnJBRdcwOmnn87HH39MSkoKr732Gr1792bevHnMnDmTK664gmXLlvG9732P2NhYpk2bxvbt23njjTdaxVNTU8P111/P2rVrGTlyJDU1NYfm3XLLLWRnZ1NTU8MVV1zBz3/+cx577DEKCws5++yzSU5OZsWKFe0u1xVYIgAam5p5KSefM0/sT0rf3sEOxxgDPPTQQ6xfv541a9YA8P7777N69WrWr19/aGjks88+S79+/aipqWHKlClcfvnlJCUltdrO1q1bWbhwIU8//TRXXXUVL7/8Mtdee+2h+bW1tdx00018+OGHZGRkcPXVV7cbz5///GdiYmLYtGkTubm5TJo06dC8X/7yl/Tr14+mpibOOecccnNzuf3223nkkUdYsWIFycnJHS43btw4f/5snWKJAPhwazFFB2t5YNboYIdiTJd0uCP3QJo6dWqr8fGPPfYYS5YsASAvL4+tW7d+LRFkZGQwYcIEACZPnszOnTtbzd+8eTPDhg07tN2rr76aBQsWfG3fH374IbfffjsA48aNa9WAv/jiiyxYsIDGxkb27NnDxo0b223gfV0u0CwRAAtX5pEcF8k5owYEOxRjzGHExsYeev/+++/zzjvv8MknnxATE8NZZ53V7vj5qKivhoKHh4e3uqTjDzt27OC3v/0t2dnZJCYmMm/evHbj8HW5YAj5zuJ9B2t5b/M+Lp+cSkR4yP8cxnQZffr0oaKiosP55eXlJCYmEhMTw+bNm/n00087tZ8RI0awffv2Q2cKixYtane5M844g3/84x8ArF+/ntzcXAAOHjxIbGwsCQkJ7N27lzfffLPd73C45YIt5M8IFq/Op6lZmWOdxMZ0KUlJSUybNo2xY8dywQUXcNFFF7WaP2PGDJ588klGjRrFiBEjOOWUUzq1n969e/PEE08wY8YMYmNj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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.plot(history.history['loss'])\n",
        "plt.plot(history.history['val_loss'])\n",
        "\n",
        "plt.title('model loss')\n",
        "plt.ylabel('loss')\n",
        "plt.xlabel('epoch')\n",
        "\n",
        "plt.legend(['trainig data', 'validation data'], loc = 'lower right')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 312
        },
        "id": "UtQXRXzb7nIX",
        "outputId": "c2caa3b0-c161-4e61-c187-0fe16f10d432"
      },
      "execution_count": 38,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.legend.Legend at 0x7f11645d1df0>"
            ]
          },
          "metadata": {},
          "execution_count": 38
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Accuracy of the model on test data"
      ],
      "metadata": {
        "id": "r9UCGfM17_L2"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "loss, accuracy = model.evaluate(x_test_std, y_test)\n",
        "print(accuracy)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YTrLhal08F0h",
        "outputId": "7c0fea2c-0b15-4218-db20-13447ac452a9"
      },
      "execution_count": 40,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "4/4 [==============================] - 0s 6ms/step - loss: 0.1529 - accuracy: 0.9474\n",
            "0.9473684430122375\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(x_test_std.shape)\n",
        "print(x_test_std[0])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CHDzfvNS8QJt",
        "outputId": "43efbe48-f036-4b1d-b806-cc8cf7a846fe"
      },
      "execution_count": 41,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(114, 30)\n",
            "[-0.08700339 -1.47192915 -0.10537391 -0.21479674  2.05627941 -0.18759821\n",
            "  0.04345969  0.3431473   0.48693221  0.20971492  0.41483725  2.38110688\n",
            "  0.53816721  0.01895993  0.95128447  0.31678369  0.28189043  2.21465008\n",
            " -0.39276605  0.44485916 -0.3863489  -1.69650664 -0.42190004 -0.44557481\n",
            "  0.23041821 -0.75521902 -0.60192371 -0.26629174 -1.09776353 -0.65597459]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred = model.predict(x_test_std)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "uuRaqt4r8bb7",
        "outputId": "49210280-9086-4ab6-8535-273f673fbd3d"
      },
      "execution_count": 43,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "4/4 [==============================] - 0s 4ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(y_pred.shape)\n",
        "print(y_pred[0])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zlbrzxAE8j5e",
        "outputId": "3cd527cb-7518-4a91-d001-d2e845a9484d"
      },
      "execution_count": 44,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(114, 2)\n",
            "[0.37792015 0.54955906]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(x_test_std)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JmQpwQzk8uBd",
        "outputId": "51920e37-ac6c-4790-da28-c91787b084dc"
      },
      "execution_count": 45,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[-0.08700339 -1.47192915 -0.10537391 ... -0.26629174 -1.09776353\n",
            "  -0.65597459]\n",
            " [ 0.19989092  0.03577342  0.1706179  ...  0.44844054  0.06066588\n",
            "   0.02108157]\n",
            " [-1.28858427 -0.21847659 -1.30667757 ... -1.41981535  0.19788632\n",
            "  -0.31050377]\n",
            " ...\n",
            " [ 0.67523542  0.61546345  0.70329853 ...  1.36221218  1.000987\n",
            "   0.62759948]\n",
            " [ 0.20832899  1.5866985   0.10942329 ... -1.35965118 -1.95719681\n",
            "  -1.62740299]\n",
            " [ 0.78774299  0.03068842  0.84293725 ...  2.03773974  0.27299646\n",
            "   0.34822356]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(y_pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6f8BR1RE8w_A",
        "outputId": "7410b4a7-2354-46c1-c952-23794365a452"
      },
      "execution_count": 46,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[0.37792015 0.54955906]\n",
            " [0.5888045  0.62613297]\n",
            " [0.17090304 0.9544517 ]\n",
            " [0.98042953 0.00132382]\n",
            " [0.6302151  0.58440727]\n",
            " [0.9214151  0.02170996]\n",
            " [0.25010642 0.63849795]\n",
            " [0.6369558  0.98989594]\n",
            " [0.39293393 0.9287289 ]\n",
            " [0.5422983  0.9591753 ]\n",
            " [0.39496258 0.7050433 ]\n",
            " [0.25718647 0.86734444]\n",
            " [0.2136843  0.5861833 ]\n",
            " [0.318711   0.8107385 ]\n",
            " [0.23222646 0.950657  ]\n",
            " [0.8054162  0.3362036 ]\n",
            " [0.29853883 0.96435994]\n",
            " [0.26616555 0.85447973]\n",
            " [0.40829638 0.897457  ]\n",
            " [0.94790554 0.03734007]\n",
            " [0.8790277  0.73101586]\n",
            " [0.5523984  0.9705303 ]\n",
            " [0.3149309  0.9282167 ]\n",
            " [0.4462028  0.97738343]\n",
            " [0.5139359  0.9291807 ]\n",
            " [0.79014707 0.07977167]\n",
            " [0.52419657 0.92265606]\n",
            " [0.63296825 0.7662201 ]\n",
            " [0.76458955 0.16561987]\n",
            " [0.76840407 0.08359836]\n",
            " [0.6735921  0.9421296 ]\n",
            " [0.27958238 0.87552214]\n",
            " [0.65406007 0.95952475]\n",
            " [0.9737483  0.00318892]\n",
            " [0.8888764  0.02125787]\n",
            " [0.447226   0.8312976 ]\n",
            " [0.47319183 0.98967004]\n",
            " [0.6219137  0.87979454]\n",
            " [0.18983272 0.9765614 ]\n",
            " [0.19472952 0.9052829 ]\n",
            " [0.9900847  0.00138226]\n",
            " [0.5847063  0.3284881 ]\n",
            " [0.09521236 0.89154124]\n",
            " [0.29016176 0.9166984 ]\n",
            " [0.7301542  0.24877548]\n",
            " [0.36700165 0.9694341 ]\n",
            " [0.24439934 0.98890936]\n",
            " [0.0466972  0.86366224]\n",
            " [0.9105327  0.01696497]\n",
            " [0.75170916 0.1241501 ]\n",
            " [0.6210784  0.9556489 ]\n",
            " [0.68961555 0.33788773]\n",
            " [0.46442533 0.54574233]\n",
            " [0.4192217  0.9601269 ]\n",
            " [0.18355569 0.9687527 ]\n",
            " [0.6983275  0.77500457]\n",
            " [0.23612909 0.87024415]\n",
            " [0.13422865 0.8585206 ]\n",
            " [0.735556   0.01192599]\n",
            " [0.45706508 0.89594615]\n",
            " [0.39045405 0.79389983]\n",
            " [0.8762783  0.13758247]\n",
            " [0.4199237  0.9710142 ]\n",
            " [0.8606452  0.04898447]\n",
            " [0.7806561  0.18650521]\n",
            " [0.6076012  0.42920935]\n",
            " [0.8790344  0.05673262]\n",
            " [0.69332045 0.08645371]\n",
            " [0.66103464 0.78273386]\n",
            " [0.29769972 0.07219449]\n",
            " [0.57189363 0.18920605]\n",
            " [0.7950594  0.08274452]\n",
            " [0.33525252 0.8620286 ]\n",
            " [0.67551625 0.4246088 ]\n",
            " [0.32121322 0.99019444]\n",
            " [0.69854325 0.17030501]\n",
            " [0.12654439 0.91575176]\n",
            " [0.37954244 0.96700597]\n",
            " [0.5074059  0.75466675]\n",
            " [0.6651743  0.48852175]\n",
            " [0.82295185 0.06270169]\n",
            " [0.7309217  0.2826449 ]\n",
            " [0.9511096  0.03987161]\n",
            " [0.31571063 0.5952699 ]\n",
            " [0.26843336 0.83012587]\n",
            " [0.51880527 0.6264354 ]\n",
            " [0.39304584 0.8291491 ]\n",
            " [0.35584214 0.9296143 ]\n",
            " [0.5748968  0.9110775 ]\n",
            " [0.89537853 0.03537892]\n",
            " [0.38183784 0.95659924]\n",
            " [0.53064215 0.90632623]\n",
            " [0.6926491  0.95010096]\n",
            " [0.6995519  0.08117128]\n",
            " [0.73957366 0.26756436]\n",
            " [0.44582063 0.8863692 ]\n",
            " [0.84949327 0.04205887]\n",
            " [0.8940378  0.05706345]\n",
            " [0.3281109  0.91921425]\n",
            " [0.56109065 0.9867837 ]\n",
            " [0.48626855 0.9918608 ]\n",
            " [0.5798698  0.2703857 ]\n",
            " [0.91399044 0.01929011]\n",
            " [0.9496884  0.00676079]\n",
            " [0.4489927  0.9417985 ]\n",
            " [0.21161605 0.96502596]\n",
            " [0.46009886 0.97261894]\n",
            " [0.35361066 0.857855  ]\n",
            " [0.01235864 0.8531862 ]\n",
            " [0.33763397 0.7446587 ]\n",
            " [0.84208554 0.03802197]\n",
            " [0.9034698  0.06399751]\n",
            " [0.66335684 0.58107907]\n",
            " [0.770909   0.1326803 ]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "model.predict() give the prediction probabily of each class for that data point"
      ],
      "metadata": {
        "id": "Y1i5n0wB8zAi"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# argmax function\n",
        "my_list = [0.25, 0.56]\n",
        "\n",
        "index_of_max_value = np.argmax(my_list)\n",
        "print(my_list)\n",
        "print(index_of_max_value)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NA5VuMd49AMR",
        "outputId": "8b76bcfe-a942-4da2-c20d-42742a34f49f"
      },
      "execution_count": 47,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0.25, 0.56]\n",
            "1\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "converting the prediction probability to class label\""
      ],
      "metadata": {
        "id": "9s9PWSjK9O8V"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred_labels = [np.argmax(i) for i in y_pred]\n",
        "print(y_pred_labels)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2tz-ThwV9bSw",
        "outputId": "842bad14-d0e0-4a90-9367-b6c893d14701"
      },
      "execution_count": 50,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Saving the trained model"
      ],
      "metadata": {
        "id": "NN0RhUif_zrL"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pickle"
      ],
      "metadata": {
        "id": "I04RIXMZ_4xq"
      },
      "execution_count": 52,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "filename = 'trained_model.sav'\n",
        "pickle.dump(model, open(filename, 'wb'))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bmjRxIfK_41H",
        "outputId": "97c5ba78-933b-41bc-c5db-15019cb47754"
      },
      "execution_count": 56,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Keras weights file (<HDF5 file \"variables.h5\" (mode r+)>) saving:\n",
            "...layers\n",
            "......dense\n",
            ".........vars\n",
            "............0\n",
            "............1\n",
            "......dense_1\n",
            ".........vars\n",
            "............0\n",
            "............1\n",
            "......flatten\n",
            ".........vars\n",
            "...metrics\n",
            "......mean\n",
            ".........vars\n",
            "............0\n",
            "............1\n",
            "......mean_metric_wrapper\n",
            ".........vars\n",
            "............0\n",
            "............1\n",
            "...optimizer\n",
            "......vars\n",
            ".........0\n",
            ".........1\n",
            ".........2\n",
            ".........3\n",
            ".........4\n",
            ".........5\n",
            ".........6\n",
            ".........7\n",
            ".........8\n",
            "...vars\n",
            "Keras model archive saving:\n",
            "File Name                                             Modified             Size\n",
            "metadata.json                                  2023-03-28 06:57:34           64\n",
            "config.json                                    2023-03-28 06:57:34         1543\n",
            "variables.h5                                   2023-03-28 06:57:34        30472\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Bulding the predictive system**"
      ],
      "metadata": {
        "id": "TCrTgRiJ9r0u"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "input_data = (11.76,21.6,74.72,427.9,0.08637,0.04966,0.01657,0.01115,0.1495,0.05888,0.4062,1.21,2.635,28.47,0.005857,0.009758,0.01168,0.007445,0.02406,0.001769,12.98,25.72,82.98,516.5,0.1085,0.08615,0.05523,0.03715,0.2433,0.06563)\n",
        "\n",
        "# change the input_data to numpy array\n",
        "input_data_as_numpy_array = np.asarray(input_data)\n",
        "\n",
        "# reshape the numpy array as we are predicting for one data point\n",
        "input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)\n",
        "\n",
        "# standardizing the input data\n",
        "input_data_std = scaler.transform(input_data_reshaped)\n",
        "\n",
        "prediction = model.predict(input_data_std)\n",
        "print(prediction)\n",
        "\n",
        "prediction_label = [np.argmax(prediction)]\n",
        "print(prediction_label)\n",
        "\n",
        "if(prediction_label[0]==0):\n",
        "  print('The tumor is Malignant')\n",
        "\n",
        "else:\n",
        "  print('The tumor is Benign')\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GeOQ2gc895OO",
        "outputId": "56d1b646-eef5-4ad8-8ed1-beb5f785a1f6"
      },
      "execution_count": 57,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1/1 [==============================] - 0s 30ms/step\n",
            "[[0.42958477 0.9696815 ]]\n",
            "[1]\n",
            "The tumor is Benign\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.9/dist-packages/sklearn/base.py:439: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names\n",
            "  warnings.warn(\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# loading the save model?\n",
        "loaded_model = pickle.load(open('trained_model.sav', 'rb'))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iFnkFqLu_Ry4",
        "outputId": "3522dfb7-aa24-482a-e3c4-a243335648c4"
      },
      "execution_count": 61,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Keras model archive loading:\n",
            "File Name                                             Modified             Size\n",
            "metadata.json                                  2023-03-28 06:57:34           64\n",
            "config.json                                    2023-03-28 06:57:34         1543\n",
            "variables.h5                                   2023-03-28 06:57:34        30472\n",
            "Keras weights file (<HDF5 file \"variables.h5\" (mode r)>) loading:\n",
            "...layers\n",
            "......dense\n",
            ".........vars\n",
            "............0\n",
            "............1\n",
            "......dense_1\n",
            ".........vars\n",
            "............0\n",
            "............1\n",
            "......flatten\n",
            ".........vars\n",
            "...metrics\n",
            "......mean\n",
            ".........vars\n",
            "............0\n",
            "............1\n",
            "......mean_metric_wrapper\n",
            ".........vars\n",
            "............0\n",
            "............1\n",
            "...optimizer\n",
            "......vars\n",
            ".........0\n",
            ".........1\n",
            ".........2\n",
            ".........3\n",
            ".........4\n",
            ".........5\n",
            ".........6\n",
            ".........7\n",
            ".........8\n",
            "...vars\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "input_data = (11.76,21.6,74.72,427.9,0.08637,0.04966,0.01657,0.01115,0.1495,0.05888,0.4062,1.21,2.635,28.47,0.005857,0.009758,0.01168,0.007445,0.02406,0.001769,12.98,25.72,82.98,516.5,0.1085,0.08615,0.05523,0.03715,0.2433,0.06563)\n",
        "\n",
        "# change the input_data to numpy array\n",
        "input_data_as_numpy_array = np.asarray(input_data)\n",
        "\n",
        "# reshape the numpy array as we are predicting for one data point\n",
        "input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)\n",
        "\n",
        "# standardizing the input data\n",
        "input_data_std = scaler.transform(input_data_reshaped)\n",
        "\n",
        "prediction = loaded_model.predict(input_data_std)\n",
        "print(prediction)\n",
        "\n",
        "prediction_label = [np.argmax(prediction)]\n",
        "print(prediction_label)\n",
        "\n",
        "if(prediction_label[0]==0):\n",
        "  print('The tumor is Malignant')\n",
        "\n",
        "else:\n",
        "  print('The tumor is Benign')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "N05N2BuHBcqM",
        "outputId": "909615c4-47ca-48d8-fd33-20e01fefa7ea"
      },
      "execution_count": 62,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.9/dist-packages/sklearn/base.py:439: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1/1 [==============================] - 0s 138ms/step\n",
            "[[0.42958477 0.9696815 ]]\n",
            "[1]\n",
            "The tumor is Benign\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "vYJY9nkmCFc8"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}