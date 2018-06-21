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


def bar_graph_class_names():
    """
    Simple plot of number of reviews per class name
    """
    # get classes and counts
    class_counts = df.groupby(["Class Name"]).size().to_dict()
    class_counts = collections.OrderedDict(sorted(class_counts.items(), key=lambda x: x[1], reverse=True))

    # create bar graph
    num_classes = len(class_counts)
    fig, ax = plt.subplots()
    plt.bar(np.arange(num_classes), class_counts.values(), color="violet")
    plt.xticks(np.arange(num_classes), class_counts.keys(), fontsize=18, rotation=80)
    ax.tick_params(axis="y", which="major", labelsize=20)
    plt.title("Number of Reviews for each Item Type", fontsize=32)
    plt.ylabel("Number of Reviews", fontsize=24)
    plt.show()


def bag_of_words():
    """
    Building a bag-of-words-model for the review texts.
    This function takes a long time to execute due to the size of the data set and the time it takes to process texts.
    """
    # -- Cleaning the texts
    import re
    # import nltk
    # nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    corpus = []
    corpus_size = 5000    # if we were to iterate over all of it, it would take way too long
    for i in range(corpus_size):
        review_text = df["Review Text"][i]
        if type(review_text) != str:
            corpus.append("")
            continue
        review = re.sub('[^a-zA-Z]', " ", review_text)
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)

    # Creating the Bag of Words model
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()
    y = df.iloc[:corpus_size, 4].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Fitting Naive Bayes to the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    classNames = ["no", "yes"]
    ticks = np.arange(len(classNames))
    labels = [["TN", "FP"], ["FN", "TP"]]

    plt.clf()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Wistia)
    plt.title("Product Recommendations", fontsize=36)
    plt.ylabel("True Label", fontsize=32)
    plt.xlabel("Predicted Label", fontsize=32)
    plt.xticks(ticks, classNames, rotation=45, fontsize=24)
    plt.yticks(ticks, classNames, fontsize=24)
    for i in range(len(classNames)):
        for j in range(len(classNames)):
            plt.text(j, i, str(labels[i][j]) + " = " + str(cm[i][j]), fontsize=32)
    plt.show()


if __name__ == "__main__":
    # -- Import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import collections

    # -- Import dataset
    df = pd.read_csv("womens_clothing_ecommerce_reviews.csv")
    # retain only the relevant columns
    df = df.iloc[:, 2:]
    df = df.drop(["Division Name", "Department Name"], axis=1)

    # -- Create bar graph of class names
    bar_graph_class_names()

    # -- Bag of words model and classification for review texts
    # bag_of_words()
