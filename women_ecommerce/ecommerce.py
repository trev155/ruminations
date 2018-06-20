# -- About the Dataset
"""
https://www.kaggle.com/monkey09/women-e-clothing-reviews
Ecommerce reviews of women's clothing items.

Column Descriptions of relevant columns (note that there are several NaNs throughout):
Age - Reviewer’s age.
Title - Review Title.
Review - Text Review body.
Rating - Rating of the product from 1 Worst, to 5 Best.
Recommended IND - whether Customer recommends the product where 1 is recommended, 0 is not recommended.
Positive Feedback Count - Whether other customers found the review helpful.
Division Name - Categorical name of the product high level division.
Department Name - Categorical name of the product department name.
Class Name - Categorical name of the product class name.
"""

# -- Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections
import wordcloud

# -- Import dataset
dataset = pd.read_csv("womens_clothing_ecommerce_reviews.csv")
# retain only the relevant columns
dataset = dataset.iloc[:, 2:]
dataset = dataset.drop(["Division Name", "Department Name"], axis=1)

# -- Simple plot of number of reviews per class name
# get classes and counts
classes = [c for c in dataset.iloc[:, -1].values]
unique, counts = np.unique(classes, return_counts=True)
class_counts = dict(zip(unique, counts))
del class_counts["nan"]
class_counts = collections.OrderedDict(sorted(class_counts.items(), key=lambda x: x[1], reverse=True))

# create bar graph
num_classes = len(class_counts)
fig, ax = plt.subplots()
plt.bar(np.arange(num_classes), class_counts.values(), color="Navy")
plt.xticks(np.arange(num_classes), class_counts.keys(), fontsize=18, rotation=80)
ax.tick_params(axis="y", which="major", labelsize=20)
plt.title("Number of Reviews for each Item Type", fontsize=32)
plt.ylabel("Number of Reviews", fontsize=24)

