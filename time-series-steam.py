#Import Libraries and Packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylot as plt
from scipy.stats import pearsonr

##Dataset is composed of a random sample of 100,000 rows (sampled_df.csv) from Steam Data Project. 
#EDA code is found in the file titled "steam-eda.py"

##Import, read and save datasets to dataframes
df = pd.read_csv('sample_df.csv')


##Explore dataset
df.head()
df.info()
df.describe(include='object')


##Correlation Analysis: User Reviews v. Game Reviews
# Group by year and sum the number of reviews
user_reviews_by_year = df.groupby('rec_year')['user_reviews'].sum().reset_index()
game_reviews_by_year = df.groupby('rec_year')['game_reviews'].sum().reset_index()

# Merge both dataframes on 'rec_year'
merged_reviews = pd.merge(user_reviews_by_year, game_reviews_by_year, on='rec_year')

# Calculate Pearson correlation coefficient
correlation, p_value = pearsonr(merged_reviews['user_reviews'], merged_reviews['game_reviews'])

print(f"Pearson Correlation Coefficient: {correlation:.4f}")
print(f"P-value: {p_value:.4f}")

##Create visualization
# Group by year and sum the number of user reviews and game reviews
user_reviews_by_year = df.groupby('rec_year')['user_reviews'].sum().reset_index()
game_reviews_by_year = df.groupby('rec_year')['game_reviews'].sum().reset_index()

# Create the plot
plt.figure(figsize=(10, 6))

# Plot user reviews
plt.plot(user_reviews_by_year['rec_year'], user_reviews_by_year['user_reviews'], marker='o', linestyle='-', color='blue', label='User Reviews')

# Plot game reviews
plt.plot(game_reviews_by_year['rec_year'], game_reviews_by_year['game_reviews'], marker='o', linestyle='-', color='orange', label='Game Reviews')

# Add title and labels
plt.title('Comparison of User Reviews and Game Reviews by Year')
plt.xlabel('Year')
plt.ylabel('Number of Reviews')
plt.legend()  # Add legend to differentiate lines
plt.grid()

# Show plot
plt.show()


## Split graph into two for better clarity.
# Group by year and sum the number of user reviews
user_reviews_by_year = df.groupby('rec_year')['user_reviews'].sum().reset_index()

# Group by year and sum the number of game reviews
game_reviews_by_year = df.groupby('rec_year')['game_reviews'].sum().reset_index()

# Plot the time series in separate subplots
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# Plot user reviews
axes[0].plot(user_reviews_by_year['rec_year'], user_reviews_by_year['user_reviews'], marker='o', linestyle='-', color='blue')
axes[0].set_title('User Reviews by Year')
axes[0].set_ylabel('Number of User Reviews')
axes[0].grid()

# Plot game reviews
axes[1].plot(game_reviews_by_year['rec_year'], game_reviews_by_year['game_reviews'], marker='o', linestyle='-', color='orange')
axes[1].set_title('Game Reviews by Year')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Number of Game Reviews')
axes[1].grid()

plt.tight_layout()
plt.show()


#Recommendations over time
# Group by year and calculate the total recommendations
recommendations_by_year = df.groupby('rec_year')['is_recommended'].sum().reset_index()

plt.figure(figsize=(10, 6))
plt.plot(recommendations_by_year['rec_year'], recommendations_by_year['is_recommended'], marker='o', linestyle='-', label='Recommendations')
plt.title('Recommendations Over Time')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend()
plt.grid()
plt.show()