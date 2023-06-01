#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm




df = pd.read_csv("nbaplayersdraft.csv")
df.head(20)


# ### Part 1
# 
# A) Which NBA team(s) has drafter the most playes who
# 
#     a. went to Duke and were drafter in or before the 2000 draft?
#     b. have a first name that begins with D and were drafter in an even year draft (1990, 1992, 1994, ...)?

# In[21]:


duke_players = df[(df['college'] == 'Duke') & (df['year'] <= 2000)]
duke_players

team_counts = duke_players['team'].value_counts()
team_counts
max_count = team_counts.max()
teams_with_max_count = team_counts[team_counts == max_count]
print("NBA team(s) that drafted the most players from Duke in or before the 2000 draft:")
for team in teams_with_max_count.index:
    print(team)




# In[22]:


filtered_players = df[(df['player'].str.startswith('D')) & (df['year'] %2 == 0)]

team_counts = filtered_players['team'].value_counts()
max_count = team_counts.max()
teams_with_max_count = team_counts[team_counts == max_count]

print("NBA team(s) that drafted the most players with first names starting with 'D' in even years:")
for team in teams_with_max_count.index:
    print(team)





#  B) Describe the relationship between a team's first round pick slot in one year with their first-round pick slot in the subsequent year.

# In[23]:


grouped = df.sort_values('overall_pick').groupby(['team', 'year']).first()


# In[24]:


odd_years = grouped.loc[grouped.index.get_level_values('year') % 2 != 0, 'points_per_game']
even_years = grouped.loc[grouped.index.get_level_values('year') % 2 == 0, 'points_per_game']




# In[25]:


import matplotlib.pyplot as plt


plt.plot(odd_years.index.get_level_values('year'), odd_years.values, 'o', linestyle='None', label='Odd Years')
plt.plot(even_years.index.get_level_values('year'), even_years.values, 'o', linestyle='None', label='Even Years')


plt.xlabel('Year')
plt.ylabel('Points per Game')
plt.title('Odd and Even Years Comparison')
plt.legend()
plt.show()





# In[26]:


#a formula i created in assessing the efficiency of each players
df["efficiency"] = df["points_per_game"] + 0.5 * df["average_total_rebounds"] + 0.75 * df["average_assists"]
df = df.dropna()
df


# In[62]:


df["efficiency_rating"] = (df["efficiency"] - df["efficiency"].min())/ (df["efficiency"].max() - df["efficiency"].min())
df
#normalization


# ### Part 2
# 
# A) Prompt: Analyze draft position value and team success/deficiencies comapared to expectation. 
# 
#  

# a. Create a method for valueing each draft slot in the NBA Draft (picks 1 through 60 in most drafts).

# In[63]:


def calculate_average_values(df):
    avg_all_minutes_played = df.groupby('overall_pick')['average_minutes_played'].mean()
    avg_win_shares = df.groupby('overall_pick')['win_shares'].mean()
    return avg_all_minutes_played, avg_win_shares


avg_all_minutes_played, avg_win_shares = calculate_average_values(df)



# Print average values for each draft position
for draft_position in range(1, 61):
    print(f"Draft Position {draft_position}: Avg Minutes_Played - {avg_all_minutes_played.get(draft_position, 0)}, Avg Win Shares - {avg_win_shares.get(draft_position, 0)}")


# In[64]:


import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

def assign_draft_pick_values(avg_win_shares, observed_values):
    num_picks = len(avg_win_shares)
    draft_values = cp.Variable(num_picks)

    # Objective function: minimize squared error between assigned values and observed values
    objective = cp.Minimize(cp.sum_squares(draft_values - observed_values))

    # Constraints: values should be non-increasing and decrease by at least 5% from pick to pick
    constraints = [
        draft_values[:-1] - draft_values[1:] >= 0,
        draft_values[:-1] - draft_values[1:] >= 0.05 * draft_values[:-1]
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    assigned_values = draft_values.value

    return assigned_values

avg_win_shares = df.groupby('overall_pick')['win_shares'].mean()   
observed_values = df.groupby('overall_pick')['average_minutes_played'].mean()  

assigned_values = assign_draft_pick_values(avg_win_shares, observed_values)
assigned_values

# Plot assigned values
draft_positions = np.arange(1, len(assigned_values) + 1)
plt.plot(draft_positions, assigned_values, marker='o')
plt.xlabel('Draft Position')
plt.ylabel('Assigned Value')
plt.title('Assigned Values for NBA Draft Picks')
plt.grid(True)
plt.show()




# The model proves to be a fairly strong model because the value decreases from pick to pick. I forced the quadratic program to pick numbers that decrease by at least 5% from pick to pick in order to produce a smoother graph. 

# b. Conditional on the expected value of the draft positions, which NBA teams have over or underperformed the most when drafting during this time span. Which College Teams have had the players outperform expectations the most after entering the NBA?

# In[65]:


expected = (1 + 60) / 2
expected
df[df['overall_pick'] >= 30.5]['team'] #underperformed
df[df['overall_pick'] < 30.5]['team'] #overperformed




filtered_df = df[df['overall_pick'] >= 30.5]

# Perform a count of occurrences for each team
team_counts = filtered_df['team'].value_counts()
# Sort the teams based on the count of occurrences
team_counts = team_counts.sort_values(ascending=False)

print("Teams that went above the expected value the most:")
overperforming_teams = team_counts[team_counts > team_counts.mean()]
print(overperforming_teams)

print("\nTeams that went below the expected value the most:")
underperforming_teams = team_counts[team_counts < team_counts.mean()]
print(underperforming_teams)



# In[66]:


average_draft_position = df['overall_pick'].mean()
better_than_average = df[df['overall_pick'] < average_draft_position]
college_frequency = better_than_average['college'].value_counts()
college_frequency = college_frequency.sort_values(ascending=False)

print("College teams with players having better-than-average draft positions:")
print(college_frequency)


# In[67]:


# Group the data by college and calculate the average value over replacement
college_performance = df.groupby('college')['value_over_replacement'].mean().reset_index()

# Sort the colleges based on average value over replacement in descending order
sorted_colleges = college_performance.sort_values('value_over_replacement', ascending=False)
# Generate the ranking graph
plt.figure(figsize=(10, 6))
plt.barh(sorted_colleges['college'], sorted_colleges['value_over_replacement'])
plt.xlabel('Average Value over Replacement')
plt.ylabel('College Teams')
plt.title('College Teams with Players Outperforming Expectations in NBA')
plt.grid(True, axis='x')
plt.show()





# c. Explain and present your findings with tables and visuals. What additional research areas would you focus on if given the opportunity to expand this study?

# Given the oppurtunity, I would expand this study in gaining insights to productivity score by team.
# 

# In[68]:


df.groupby(['team'])['value_over_replacement'].agg(['mean', min, max])



# In[41]:


plt.figure(figsize = (20, 15))
sns.barplot(data = df, x = 'team', y = 'value_over_replacement')


# In[ ]:




