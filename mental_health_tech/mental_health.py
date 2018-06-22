# -- About the Dataset
"""
https://www.kaggle.com/osmi/mental-health-in-tech-survey
Dataset from 2014.

Columns:
- Timestamp
- Age
- Gender
- Country
- state: If you live in the United States, which state or territory do you live in?
- self_employed: Are you self-employed?
- family_history: Do you have a family history of mental illness?
- treatment: Have you sought treatment for a mental health condition?
- work_interfere: If you have a mental health condition, do you feel that it interferes with your work?
- no_employees: How many employees does your company or organization have?
- remote_work: Do you work remotely (outside of an office) at least 50% of the time?
- tech_company: Is your employer primarily a tech company/organization?
- benefits: Does your employer provide mental health benefits?
- care_options: Do you know the options for mental health care your employer provides?
- wellness_program: Has your employer ever discussed mental health as part of an employee wellness program?
- seek_help: Does your employer provide resources to learn more about mental health issues and how to seek help?
- anonymity: Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?
- leave: How easy is it for you to take medical leave for a mental health condition?
- mental_health_consequence: Do you think that discussing a mental health issue with your employer would have negative consequences?
- phys_health_consequence: Do you think that discussing a physical health issue with your employer would have negative consequences?
- coworkers: Would you be willing to discuss a mental health issue with your coworkers?
- supervisor: Would you be willing to discuss a mental health issue with your direct supervisor(s)?
- mental_health_interview: Would you bring up a mental health issue with a potential employer in an interview?
- phys_health_interview: Would you bring up a physical health issue with a potential employer in an interview?
- mental_vs_physical: Do you feel that your employer takes mental health as seriously as physical health?
- obs_consequence: Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?
- comments: Any additional notes or comments
"""


def age_gender_histogram():
    """
    Create a histogram for age and gender.
    """
    ages_m = []
    ages_f = []
    for index, row in df.iterrows():
        gender = row["Gender"]
        if gender == "m":
            ages_m.append(row["Age"])
        if gender == "f":
            ages_f.append(row["Age"])

    # plot histograms
    plt.close()
    fig, ax = plt.subplots()
    plt.hist(ages_m, bins=32, color="orange", alpha=0.5, label="Male")
    plt.hist(ages_f, bins=32, color="violet", alpha=0.5, label="Female")
    plt.xlabel("Age", fontsize=32)
    plt.ylabel("Frequency", fontsize=32)
    plt.title("Histogram of Ages", fontsize=48)
    ax.tick_params(axis="x", which="major", labelsize=20)
    ax.tick_params(axis="y", which="major", labelsize=20)
    plt.grid(True)
    plt.legend(prop={'size': 32})
    plt.show()


def categorical_data_tables():
    """
    Generate categorical data tables.
    """
    from tabulate import tabulate

    # Family History
    family_histories = pd.DataFrame(df.groupby(["family_history"]).size())
    family_histories.columns = ["count"]
    print(tabulate(family_histories, headers="keys", tablefmt="orgtbl"))

    # Treatment
    treatment = pd.DataFrame(df.groupby(["treatment"]).size())
    treatment.columns = ["count"]
    print(tabulate(treatment, headers="keys", tablefmt="orgtbl"))

    # Work Interference
    interference = pd.DataFrame(df.groupby(["work_interfere"]).size())
    interference.columns = ["count"]
    print(tabulate(interference, headers="keys", tablefmt="orgtbl"))

    # Working Remotely
    remotely = pd.DataFrame(df.groupby(["remote_work"]).size())
    remotely.columns = ["count"]
    print(tabulate(remotely, headers="keys", tablefmt="orgtbl"))

    # Mental Health Benefits
    benefits = pd.DataFrame(df.groupby(["benefits"]).size())
    benefits.columns = ["count"]
    print(tabulate(benefits, headers="keys", tablefmt="orgtbl"))

    # Care Options
    care_options = pd.DataFrame(df.groupby(["care_options"]).size())
    care_options.columns = ["count"]
    print(tabulate(care_options, headers="keys", tablefmt="orgtbl"))

    # Seeking Help
    seek_help = pd.DataFrame(df.groupby(["seek_help"]).size())
    seek_help.columns = ["count"]
    print(tabulate(seek_help, headers="keys", tablefmt="orgtbl"))

    # Mental Health Consequence
    mental_consequence = pd.DataFrame(df.groupby(["mental_health_consequence"]).size())
    mental_consequence.columns = ["count"]
    print(tabulate(mental_consequence, headers="keys", tablefmt="orgtbl"))

    # Physical Health Consequence
    physical_consequence = pd.DataFrame(df.groupby(["phys_health_consequence"]).size())
    physical_consequence.columns = ["count"]
    print(tabulate(physical_consequence, headers="keys", tablefmt="orgtbl"))


def wordcloud_comments():
    def preprocess_comment(s):
        """
        Preprocess a comment.
        :return: str, preprocessed comment
        """
        return s.lower().strip()

    def construct_stopwords():
        """
        Add some words to the stopwords list.
        """
        sw = STOPWORDS
        added = ["employer", "employee", "employed", "company", "mental", "health", "issues", "work", "people",
                 "know", "questions"]
        for e in added: sw.add(e)
        return sw

    from wordcloud import WordCloud, STOPWORDS
    comments = np.array([preprocess_comment(comment) for comment in df["comments"].values if type(comment) == str])
    text = " ".join(comments)
    wordcloud = WordCloud(background_color="white", width=700, height=500, collocations=False, max_words=150, stopwords=construct_stopwords())
    wordcloud.generate(text)
    plt.axis("off")
    plt.imshow(wordcloud)


if __name__ == "__main__":
    # -- Import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    # -- Import dataset
    df = pd.read_csv("mental_health_in_tech_survey.csv")

    # -- Cleaning
    # clean age data
    median_age = np.median(df["Age"].values)
    for i in range(len(df["Age"].values)):
        if df["Age"].values[i] < 0 or df["Age"].values[i] > 100:
            df["Age"].values[i] = median_age

    # clean gender data
    def decipher_gender(s):
        """
        Given a string, return either "m" or "f" or "o" (other)
        This handles most of the possible strings a user could enter.
        Some of these I infer myself (eg. malr is a typo of male).

        :param s: some string, likely in the form of "m", "M", "f", "F", "male", "female", or some variant
        :return: string
        """
        s = s.lower().strip()
        if s in ["m", "male", "man", "cis male", "male (cis)", "make", "cis man", "malr", "msle"]:
            return "m"
        elif s in ["f", "female", "woman", "female (cis)", "femake"]:
            return "f"
        else:
            # print(s)
            return "o"

    f = np.vectorize(decipher_gender)
    df["Gender"] = f(df["Gender"])

    # -- First, let's get a feel for our data. Plot a histogram of age / gender.
    age_gender_histogram()

    # -- Look at categorical data columns
    categorical_data_tables()

    # -- Wordclouds for comments
    wordcloud_comments()

    # -- Building a model for seeing what factors affect
    # Consider only these columns
    subDf = df[df["Country"] == "United States"]
    subDf = df[df["tech_company"] == "Yes"]
    # subDf = subDf[["Age", "Gender", "no_employees", "remote_work", "coworkers", "mental_health_consequence", "treatment"]]
    subDf = subDf[["Gender", "no_employees", "coworkers", "treatment", "mental_health_consequence"]]
    X = subDf.iloc[:, :-1].values
    y = subDf.iloc[:, -1].values

    # Encoding categorical data
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder = LabelEncoder()
    X[:, 0] = labelencoder.fit_transform(X[:, 0])
    X[:, 1] = labelencoder.fit_transform(X[:, 1])
    X[:, 2] = labelencoder.fit_transform(X[:, 2])
    X[:, 3] = labelencoder.fit_transform(X[:, 3])
    onehotencoder = OneHotEncoder(categorical_features=[0, 1, 2, 3])
    X_new = onehotencoder.fit_transform(X).toarray()

    # Split into training and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=0)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting Naive Bayes to the Training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
