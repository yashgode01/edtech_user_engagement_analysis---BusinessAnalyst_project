#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv(r"C:\Users\Admin\online_learning_platform.csv")


# # 1: Dataset Understanding

# ## Display first 5 records

# In[3]:


df.head()


# ## Find number of rows and columns

# In[4]:


df.shape


# ## Display column names

# In[5]:


df.columns


# ## Check data types

# In[6]:


df.dtypes


# ## Complete dataset information

# In[7]:


df.info()


# ### The dataset contains 420 users and 10 variables. It includes information about user demographics, platform usage, learning device, weekly learning hours, learning purpose, preferred content type, course completion rate and satisfaction score.

# # 2: Data Preparation

# ## Check missing values

# In[8]:


df.isnull().sum()


# ## Check duplicate records

# In[9]:


df.duplicated().sum()


# ## Check data types

# In[10]:


df.dtypes


# ## Check unique values

# In[11]:


print(df["age_group"].unique())
print(df["role"].unique())
print(df["primary_platform"].unique())
print(df["learning_device"].unique())
print(df["learning_purpose"].unique())
print(df["content_type_preference"].unique())


# ## Check numerical ranges

# In[12]:


print(df["hours_per_week"].min())
print(df["hours_per_week"].max())

print(df["completion_rate_percent"].min())
print(df["completion_rate_percent"].max())

print(df["satisfaction_score"].min())
print(df["satisfaction_score"].max())


# ### I checked the dataset for missing values, duplicate records, incorrect data types and unusual values. No missing values or duplicate records were found. The numerical variables were already stored in suitable formats, so no major data cleaning was required.

# # Exploratory Data Analysis

# ## 1. Distribution of users across age groups

# In[13]:


df["age_group"].value_counts()


# ### The 35-44 age group has the highest number of users with 98 users. The distribution shows that the platform is being used by people from different age groups rather than being concentrated in only one age segment.

# ## 2. Hours Spent on Learning Per Week

# In[14]:


df["hours_per_week"].describe()


# In[15]:


df["hours_per_week"].mean()


# In[16]:


print("Minimum:", df["hours_per_week"].min())
print("Maximum:", df["hours_per_week"].max())


# ### Users spend an average of approximately 10.48 hours per week on learning activities. The learning time varies between users, showing that some users spend considerably more time on the platform than others.

# # 3.Platform Usage Patterns

# In[17]:


df["primary_platform"].value_counts()


# In[18]:


df.groupby("primary_platform")["hours_per_week"].mean().sort_values(
    ascending=False
)


# ### Users are distributed across multiple online learning platforms. The average learning time also differs between platforms, which suggests that users may interact with different platforms at different levels of intensity.

# # 4.Preferred Learning Content Types

# In[19]:


df["content_type_preference"].value_counts()


# ### Users show different preferences for learning content such as text, video and other formats.

# # Engagement Analysis

# ## 1. Average learning hours by age group

# In[20]:


df.groupby("age_group")["hours_per_week"].mean().sort_values(
    ascending=False
)


# ### 25–34 is having 11.62 hours/week
# 
# So this age group has the highest average learning time.

# ## 2. Relationship Between Learning Hours and Completion Rate

# In[21]:


correlation = df["hours_per_week"].corr(
    df["completion_rate_percent"]
)

print("Correlation:", correlation)


# ### A correlation close to 0 means there is almost no linear relationship between the two variables.
# 
# So, spending more hours on the platform does not automatically mean that a user will have a higher completion rate.

# ## 3. Comparison of engagement across different platforms or devices

# In[22]:


# Engagement Across Different Platforms
platform_engagement = df.groupby(
    "primary_platform"
)["hours_per_week"].mean().sort_values(ascending=False)

print(platform_engagement)


# ### Engagement levels vary across learning platforms. Users of some platforms spend more time learning each week, which may indicate differences in course structure, content quality, user motivation or platform experience.

# In[23]:


#Engagement Across Different Devices
device_engagement = df.groupby(
    "learning_device"
)["hours_per_week"].mean().sort_values(ascending=False)

print(device_engagement)


# ### Laptop users appear to be more engaged when engagement is measured by weekly learning hours. This could indicate that users prefer larger screens for longer learning sessions.

# # Course Completion Analysis

# ### 1. Overall Course Completion Rate

# In[24]:


df["completion_rate_percent"].mean()


# In[25]:


df["completion_rate_percent"].describe()


# ### The average course completion rate is approximately 53.94%. This means that on average users complete around half of their assigned or enrolled course content.

# ### 2. Relationship Between Engagement and Completion

# In[26]:


df["engagement_level"] = pd.qcut(
    df["hours_per_week"],
    q=3,
    labels=["Low", "Medium", "High"]
)


# In[27]:


engagement_completion = df.groupby(
    "engagement_level",
    observed=True
)["completion_rate_percent"].mean()

print(engagement_completion)


# ### 3. Impact of Learning Purpose

# In[28]:


completion_purpose = df.groupby(
    "learning_purpose"
)["completion_rate_percent"].mean().sort_values(
    ascending=False
)

print(completion_purpose)


# Skill Upgrade is approximately 59.21%
# 
# This is the highest average completion rate among the learning-purpose groups.
# 
# Users learning for skill improvement have the highest average completion rate, at around 59.21%. This suggests that users with a specific and practical learning goal may be more likely to continue and complete their courses.

# # User Satisfaction Analysis

# ## 1. Relationship Between Completion Rate and Satisfaction

# In[29]:


correlation = df["completion_rate_percent"].corr(
    df["satisfaction_score"]
)

print("Correlation:", correlation)


# ### Users with higher completion rates are not necessarily more satisfied than users with lower completion rates.

# ### 2. Satisfaction by Age Group

# In[3]:


satisfaction_age = df.groupby(
    "age_group"
)["satisfaction_score"].mean().sort_values(
    ascending=False
)

print(satisfaction_age)


# ### Satisfaction varies across age groups. The 25-34 age group has the highest average satisfaction score at approximately 5.66 out of 10.

# ### 3.Satisfaction by Platform

# In[5]:


satisfaction_platform = df.groupby(
    "primary_platform"
)["satisfaction_score"].mean().sort_values(
    ascending=False
)

print(satisfaction_platform)


# ### Satisfaction differs across learning platforms. Khan Academy users have the highest average satisfaction score at approximately 5.69 out of 10. This difference could be related to content, usability or the type of learning experience offered, but the dataset alone cannot determine the exact cause.

# # Data Visualization

# ### Engagement Level Distribution

# In[6]:


df["engagement_level"] = pd.qcut(
    df["hours_per_week"],
    q=3,
    labels=["Low", "Medium", "High"]
)


# In[7]:


df["engagement_level"].value_counts()


# In[8]:


df["engagement_level"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Distribution of Users by Engagement Level")
plt.xlabel("Engagement Level")
plt.ylabel("Number of Users")
plt.xticks(rotation=0)

plt.show()


# ### This chart divides users into low, medium and high engagement groups based on their weekly learning hours. It helps identify how the overall user base is distributed according to learning activity.

# ### Completion Rate by Learning Purpose

# In[9]:


purpose_completion = df.groupby(
    "learning_purpose"
)["completion_rate_percent"].mean().sort_values(
    ascending=False
)

purpose_completion.plot(kind="bar")

plt.title("Average Completion Rate by Learning Purpose")
plt.xlabel("Learning Purpose")
plt.ylabel("Average Completion Rate (%)")

plt.show()


# ### Users learning for Skill Upgrade have the highest average completion rate. This suggests that users with a specific practical goal may be more likely to complete their learning.

# ### Satisfaction by Device

# In[10]:


device_satisfaction = df.groupby(
    "learning_device"
)["satisfaction_score"].mean().sort_values(
    ascending=False
)

device_satisfaction.plot(kind="bar")

plt.title("Average Satisfaction by Learning Device")
plt.xlabel("Learning Device")
plt.ylabel("Average Satisfaction Score")

plt.show()


# ### Mobile users have the highest average satisfaction among the device groups. This suggests that the mobile experience is relatively well received by users.

# ### Platform Usage Pattern

# In[11]:


platform_users = df["primary_platform"].value_counts()

print(platform_users)


# In[12]:


platform_users.plot(kind="bar")

plt.title("Number of Users by Learning Platform")
plt.xlabel("Learning Platform")
plt.ylabel("Number of Users")

plt.show()


# ### The chart shows how the user base is distributed across different learning platforms. This can help the company understand which platforms are commonly used by its customers.

# # Business Insights

# ### 1. The 35–44 age group has the highest number of users, with 98 users. This shows that this age segment forms an important part of the platform's user base.
# ### 2. Users aged 25–34 spend the highest average time learning, at around 11.62 hours per week. This group appears to be highly active on the platform.
# ### 3. Users whose main purpose is Skill Upgrade have the highest average completion rate, around 59.21%. Goal-oriented users appear more likely to complete their learning.
# ### 4. Desktop users have the highest average completion rate at around 58.29%, while mobile users report the highest average satisfaction score of around 5.59.
# ### 5. Users spend an average of around 10.48 hours per week learning. However, learning hours have almost no relationship with completion rate.
# ### 6. Satisfaction and completion rate are also only very weakly related. This means a user can be satisfied with the platform without necessarily completing more courses.
# ### 7. User behavior differs across age groups, platforms, devices and learning purposes. Therefore, the company should avoid using the same engagement strategy for every user.

# # Business Recommendations

# ### 1 - Improve Completion Support
# 
# The platform should introduce progress reminders, milestone notifications and smaller learning targets to encourage users to complete their courses.
# 
# ### 2 - Personalize Learning
# 
# Learning recommendations should be based on the user's learning purpose, age group, device and previous activity instead of giving every user the same content.
# 
# ### 3 - Improve Desktop Learning
# 
# Since desktop users show a relatively high completion rate, the company can study which desktop features support course completion and consider bringing similar features to other devices.
# 
# ### 4 - Focus on User Engagement
# 
# Instead of measuring engagement only through hours spent, the company should also track course progress, completed lessons, return frequency and interaction with learning content.
# 
# ### 5 - Monitor Satisfaction Separately
# 
# Satisfaction and completion should be treated as separate KPIs because the analysis does not show a strong relationship between them. Regular feedback surveys can help identify specific user experience problems.
