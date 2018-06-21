# ruminations
Analysis of datasets. Practice in Python and R.

## About
A repository where I do data analysis stuff on various datasets. 
Each subdirectory will contain a different dataset / script that processes that dataset. 
All results will be displayed here for now. 

# Results
## U.S. Incomes by Occupation and Gender
Uses the following dataset:
https://www.kaggle.com/jonavery/incomes-by-career-and-gender

Here is a plot showing weekly median salaries for both men and women across all industries:
<img src="incomes_career_gender/salaries.png" width="800" height="400">

And here is a plot showing the number of men and women in all industires:
<img src="incomes_career_gender/workers.png" width="800" height="400">

What can be said about the data?
Well for sure, men earn more than women in almost all industries.
Men outnumber women in certain industries such as Engineering, Construction, and Maintenance.
Women outnumber men in industries like Office, Healthcare, and Education.

Now, as the dataset does not contain any other contextual data (eg. years of experience, position, etc.), it is difficult to conclude any more than what this above data shows.


## Women's Ecommerce Clothing Reviews
Uses this dataset:
https://www.kaggle.com/monkey09/women-e-clothing-reviews

What are the most commonly reviewed item classes from the dataset?
<img src="women_ecommerce/item_types.png" width="800" height="400">

Perhaps unsurprisingly, `Dresses` is the most reviewed item.

Next, I try my hand at some NLP. I'm following my notes from the Udemy course I'm doing.
First, I take the review texts of the first 1000 reviews (I don't take all of them, since it takes really long
for this method to process all the comments), and I clean them up - stemming, remove punctuation, etc.
Then, I fit a Bag-of-words model, and then fit a classifier to the resulting
document-term matrix. Now I have an array of 1000 cleaned strings, and then we split this array
into a test and training set. A confusion matrix is generated for the test set performance.

<img src="women_ecommerce/confusion_1000.png" width="400" height="300">

Our accuracy is pretty good, but there is a problem.
Although we successfully predict cases where the product is recommended,
we don't predict cases where the product is not recommended.

Let's try again, but with 5000 reviews.

<img src="women_ecommerce/confusion_5000.png" width="400" height="300">

Oof. that's not any better. Perhaps review text is not a good predictor for whether the reviewer recommend the product.

## Mental Health in Tech
Dataset: https://www.kaggle.com/osmi/mental-health-in-tech-survey

