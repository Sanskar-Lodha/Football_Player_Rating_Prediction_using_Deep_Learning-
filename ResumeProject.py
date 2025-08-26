#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('SoccerPlayersData.csv')


# In[3]:


df.head()


# In[4]:


df.drop(columns=['name','player_id'],inplace=True)


# In[5]:


plt.figure(figsize=(18,5))
sns.heatmap(df.isnull(),yticklabels=False)


# In[6]:


df.isnull().sum()


# In[7]:


df.dropna(subset=['rating'], inplace=True)


# In[8]:


df.dropna(subset=['role'], inplace=True)


# In[9]:


df= pd.get_dummies(df,columns=['role'], dtype=int)


# In[10]:


plt.figure(figsize=(18,6))
sns.heatmap(df.isnull(),yticklabels=False)


# In[11]:


df.fillna(0,inplace=True)


# In[12]:


df.head()


# In[14]:


cols_with_percent = ["pass_success", "shot_accuracy"]


# In[15]:


for col in cols_with_percent:
    df[col] = df[col].astype(str).str.replace("%", "", regex=True, )  
    df[col] = pd.to_numeric(df[col], errors="coerce")


# In[16]:


df[["pass_success", "shot_accuracy"]]


# In[17]:


df.corr()['rating'].sort_values()


# In[21]:


correlation = df.corr()['rating'].sort_values()
plt.figure(figsize=(20,6))
correlation.plot(kind='bar')  
plt.xticks(rotation=90)       
plt.ylabel('Correlation with rating')
plt.title('Correlation of features with rating')
plt.show()


# In[22]:


sns.boxplot(data=df, x='goals', y='rating')


# In[23]:


sns.boxplot(data=df, x='chances_created', y='rating')


# In[25]:


from sklearn.model_selection import train_test_split


# In[27]:


df.columns


# In[31]:


X=df[['acted_as_sweeper', 'diving_save', 'goals_conceded', 'minutes_played',
       'punches', 'saves', 'saves_inside_box', 'throws', 'accurate_passes',
       'assists', 'chances_created', 'goals', 'pass_success', 'total_shots',
       'blocked_shots', 'shot_accuracy', 'shot_off_target', 'shot_on_target',
       'shots_woodwork', 'accurate_long_balls', 'crosses', 'key_passes',
       'long_balls', 'passes', 'touches', 'aerials_lost', 'aerials_won',
       'clearances', 'dispossessed', 'dribbles_attempted',
       'dribbles_succeeded', 'duels_lost', 'duels_won', 'fouls',
       'interceptions', 'recoveries', 'tackles_attempted', 'tackles_succeeded',
       'was_fouled', 'is_a_sub', 'was_subbed', 'yellow_card', 'red_card',
        'role_Attacker', 'role_Defender', 'role_Keeper',
       'role_Midfielder']].values
y = df['rating'].values


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# In[33]:


from sklearn.preprocessing import MinMaxScaler


# In[34]:


scaler = MinMaxScaler()


# In[35]:


scaler.fit(X_train)


# In[36]:


X_train = scaler.transform(X_train)


# In[37]:


X_test = scaler.transform(X_test)


# In[40]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# In[41]:


len(df.columns)


# In[69]:


model = Sequential()

model.add(Dense(20, activation='relu'))  
model.add(Dropout(0.25))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))


# In[70]:


from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)


# In[71]:


model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')


# In[72]:


model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=600,batch_size=16,callbacks=[early_stop])


# In[73]:


loss = pd.DataFrame(model.history.history)


# In[77]:


ax = loss.plot()
ax.set_ylim(0,1.5)


# In[78]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[79]:


predictions = model.predict(X_test)


# In[81]:


mean_absolute_error(y_test, predictions)


# In[82]:


mean_squared_error(y_test, predictions)


# In[86]:


df['rating'].describe()


# In[95]:


plt.scatter(y_test,predictions)
plt.plot(y_test,y_test,'r')


# In[108]:


new_player = [
    0,      # acted_as_sweeper
    0,      # diving_save
    0,      # goals_conceded
    90,     # minutes_played
    0,      # punches
    0,      # saves
    0,      # saves_inside_box
    5,      # throws
    50,     # accurate_passes
    0,      # assists
    2,      # chances_created  
    0,      # goals           
    80,     # pass_success (%)
    5,      # total_shots
    1,      # blocked_shots
    50,     # shot_accuracy (%)
    1,      # shot_off_target
    2,      # shot_on_target
    0,      # shots_woodwork
    15,     # accurate_long_balls
    20,     # crosses
    8,      # key_passes
    8,      # long_balls
    70,     # passes
    75,     # touches
    4,      # aerials_lost
    8,      # aerials_won
    8,      # clearances
    4,      # dispossessed
    10,     # dribbles_attempted
    8,      # dribbles_succeeded
    6,      # duels_lost
    8,      # duels_won
    2,      # fouls
    7,      # interceptions
    15,     # recoveries
    10,     # tackles_attempted
    8,      # tackles_succeeded
    3,      # was_fouled
    0,      # is_a_sub
    0,      # was_subbed
    1,      # yellow_card
    1,      # red_card
    0,      # role_Attacker
    0,      # role_Defender
    0,      # role_Keeper
    1       # role_Midfielder
]


# In[107]:


def predict_player_rating(player_stats):
    player_scaled = scaler.transform([player_stats])
    rating = model.predict(player_scaled)[0][0]
    return rating

new_rating = predict_player_rating(new_player)
print(new_rating)


# In[ ]:




