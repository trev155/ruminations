"""
Using the Red Wine Dataset, from https://www.kaggle.com/piyushgoyal443/red-wine-dataset.

The dependent variable is "quality", the last column of the csv.
All other columns are predictors.

See the associated .txt file for more information on the data.
"""

# -- Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -- Read in the data
df = pd.read_csv("wineQualityReds.csv")
df = df.drop(df.columns[[0]], axis=1)
df.columns = ["Fixed Acidity", "Volatile Acidity", "Citric Acid", "Residual Sugar", "Chlorides",
              "Free Sulfur Dioxide", "Total Sulfur Dioxide", "Density", "pH", "Sulphates", "Alcohol",
              "Quality"]


# -- Single variable plots of independent features
# could do either a histogram, or a box plot - since all features are continuous numerical
# I don't do it for all features, just the ones that interest me

def histogram(data, xlab, ylab, title, num_bins):
    """
    Generate a histogram (matplotlib.pyplot)
    :param data: np array, or list-like structure
    :param xlab: x-axis label
    :param ylab: y-axis label
    :param title: title of plot
    """
    plt.close()
    fig, ax = plt.subplots()
    plt.hist(data, bins=num_bins, color="orange", alpha=0.5)
    plt.xlabel(xlab, fontsize=32)
    plt.ylabel(ylab, fontsize=32)
    plt.title(title, fontsize=48)
    ax.tick_params(axis="x", which="major", labelsize=20)
    ax.tick_params(axis="y", which="major", labelsize=20)
    plt.grid(True)
    plt.show()


# Citric Acid Content
histogram(df["Citric Acid"], "Citric Acid Content (g/L)", "Frequency", "Red Wine Data - Citric Acid Content", 16)

# Chloride Content
histogram(df["Chlorides"], "Chloride/Salt Content (g/L)", "Frequency", "Red Wine Data - Chloride Content", 32)

# pH values
histogram(df["pH"], "pH Values", "Frequency", "Red Wine Data - pH Values", 16)

# Alcohol
histogram(df["Alcohol"], "Alcohol Content (Percentage)",  "Frequency", "Red Wine Data - Alcohol Percentages", 16)


# -- Visualizing the dependent variable
# Quality (Dependent Variable, discrete - use a bar graph instead of a histogram)
def bar_graph(data, title, ylab):
    """
    Create a bar graph.
    :param data: should be a dictonary of counts
    :param title: title of plot
    :param ylab: y-axis label
    """
    plt.close()
    num_classes = len(data)
    fig, ax = plt.subplots()
    plt.bar(np.arange(num_classes), data.values(), color="violet")
    plt.xticks(np.arange(num_classes), data.keys(), fontsize=18)
    ax.tick_params(axis="y", which="major", labelsize=20)
    plt.title(title, fontsize=32)
    plt.ylabel(ylab, fontsize=24)
    plt.show()


scores = np.array(df["Quality"])
unique, counts = np.unique(scores, return_counts=True)
score_counts = dict(zip(unique, counts))

bar_graph(score_counts, "Red Wine Data - Quality Ratings", "Counts")


# -- 2-dimensional plots of features vs. dependent variable
# pH vs Quality
