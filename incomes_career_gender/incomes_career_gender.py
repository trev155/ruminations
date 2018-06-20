# About the Dataset
"""
https://www.kaggle.com/jonavery/incomes-by-career-and-gender
Median weekly earnings of full-time wage and salary workers by detailed occupation and sex.

Occupation: Job title as given from BLS. Industry summaries are given in ALL CAPS.
All_workers: Number of workers male and female, in thousands.
All_weekly: Median weekly income including male and female workers, in USD.
M_workers: Number of male workers, in thousands.
M_weekly: Median weekly income for male workers, in USD.
F_workers: Number of female workers, in thousands.
F_weekly: Median weekly income for female workers, in USD.
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import original dataset
dataset = pd.read_csv("incomes_career_gender.csv")

## basic data
print(dataset.iloc[0])

## only need the rows for the top level categories
rows = []
for i in range(1, dataset.shape[0]):
    occupation = dataset.iloc[i, 0]
    if occupation.isupper():
        rows.append(i)
df = pd.DataFrame(dataset.iloc[rows, :].values)
df.columns = ["Occupation Name", "Total Workers", "Median Weekly Salary",
              "Male Workers", "Median Weekly Salary (Male)",
              "Female Workers", "Median Weekly Salary (Female)"]

## convert data values to ints for simplicity
for i in range(1, len(df.columns.values)):
    col_name = df.columns.values[i]
    df[col_name] = df[col_name].astype(int)

# Plot bar graph of men/women weekly salaries
## define reusable vars
ind = np.arange(len(df))
width = 0.4
salaries_m = df.iloc[:, 4].values
salaries_f = df.iloc[:, 6].values
yticklabels = df.iloc[:, 0].values
ylim_bottom = 2*width-1
ylim_top = len(df)

## plot the graph
fig, ax = plt.subplots()
ax.barh(ind, salaries_m, width, color="SkyBlue", label="Male")
ax.barh(ind + width, salaries_f, width, color="IndianRed", label="Female")
ax.set_title("Median Weekly Salaries across Industries (January 2015)", fontsize=32)
ax.tick_params(axis="x", which="major", labelsize=20)
ax.set_xlabel("Salary ($US)", fontsize=24)
ax.set_yticks(ind + width)
ax.set_ylim(ylim_bottom, ylim_top)
ax.set_yticklabels(yticklabels, fontsize=20)
ax.legend(fontsize=32)

plt.show()