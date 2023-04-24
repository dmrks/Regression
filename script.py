# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import codecademylib3

# Read in the data
codecademy = pd.read_csv('codecademy.csv')

# Print the first five rows
print(codecademy.head(5))

# Create a scatter plot of score vs completed
plt.scatter(codecademy.completed,codecademy.score)
plt.xlabel("Completed")
plt.ylabel("Score")
plt.show() # Show the plot
plt.clf() # Clear the plot

# Fit a linear regression to predict score based on prior lessons completed
model = sm.OLS.from_formula("score ~ completed", data = codecademy )
result = model.fit()
print(result.params)

# Intercept interpretation: Intercept = 13.214113 ,O lessons taken result in 13.214113 Points

# Slope interpretation: For every 1 Lesson more completed, the score is 1.306826 better

# Plot the scatter plot with the line on top

plt.scatter(codecademy.completed,codecademy.score)
plt.xlabel("Completed")
plt.ylabel("Score")
plt.plot(codecademy.completed, result.predict(codecademy))
plt.show() # Show the plot
plt.clf() # Clear the plot

# Predict score for learner who has completed 20 prior lessons = 39.350624877322936
twenty_lessons = 1.306825592807168* 20 + 13.21411302117958
print(twenty_lessons)

# Calculate fitted values
fitted_values = result.predict(codecademy)

# Calculate residuals
residuals = codecademy.score - fitted_values
# print(residuals)

# Check normality assumption = normal distributed
plt.hist(residuals)
plt.show() # Show the plot
plt.clf() # Clear the plot

# Check homoscedasticity assumption = homoscedasticity assumption met

plt.scatter(fitted_values,residuals)
plt.show() # Show the plot
plt.clf() # Clear the plot

# Create a boxplot of score vs lesson = Median Score higher on Lesson A
sns.boxplot(codecademy.lesson,codecademy.score)
plt.show() # Show the plot
plt.clf() # Clear the plot

# Fit a linear regression to predict score based on which lesson they took = Intercept	59.22 + lesson[T.Lesson B]	-11.64200000000001

model_two = sm.OLS.from_formula("score ~ lesson", data = codecademy )
results = model_two.fit()
print(results.params)

# Calculate and print the group means and mean difference (for comparison) = Lesson A  59.220 / Lesson B  47.578

print(codecademy.groupby('lesson').mean().score)

# Use `sns.lmplot()` to plot `score` vs. `completed` colored by `lesson`
sns.lmplot(x = 'completed', y = 'score', hue = 'lesson', data = codecademy)
plt.show()
