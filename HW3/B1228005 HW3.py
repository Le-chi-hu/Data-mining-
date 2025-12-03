import pandas as pd
import os
from sklearn.model_selection import GridSearchCV

dataset = pd.read_csv('NBA_2014_games.csv')
print(dataset.iloc[:5])

dataset = pd.read_csv('NBA_2014_games.csv', parse_dates=["Date"])

dataset = dataset.rename(columns={
    "Visitor/Neutral": "Visitor Team",
    "PTS": "VisitorPts",
    "Home/Neutral": "Home Team",
    "PTS.1": "HomePts"
})

print(dataset.iloc[:5])


dataset["HomeWin"] = dataset["VisitorPts"] < dataset["HomePts"]
from collections import defaultdict
won_last = defaultdict(int)

home_last_list = []
visitor_last_list = []

for index, row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    

    home_last_list.append(won_last[home_team])
    visitor_last_list.append(won_last[visitor_team])

    won_last[home_team] = row["HomeWin"]
    won_last[visitor_team] = not row["HomeWin"]


dataset["HomeLastWin"] = home_last_list
dataset["VisitorLastWin"] = visitor_last_list


print(dataset.iloc[20:25])

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=14)
import numpy as np

from sklearn.model_selection import cross_val_score
y_true = dataset["HomeWin"].values 
X_previouswins = dataset[["HomeLastWin", "VisitorLastWin"]].values

scores = cross_val_score(clf, X_previouswins, y_true,scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

data_folder=r"C:\Users\User\Downloads\資料探勘hw3"
standings_filename = os.path.join(data_folder,"leagues_NBA_2013_standings_expanded-standings.csv")
standings = pd.read_csv(standings_filename,header=0)

print(standings) 



dataset["HomeTeamRanksHigher"] = 0
for index, row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]


    if home_team == "New Orleans Pelicans":
        home_team = "New Orleans Hornets"
    elif visitor_team == "New Orleans Pelicans":
        visitor_team = "New Orleans Hornets"

    home_rank = standings[standings["Team"] ==home_team]["Rk"].values[0]
    visitor_rank = standings[standings["Team"] ==visitor_team]["Rk"].values[0]
    row["HomeTeamRanksHigher"] = int(home_rank < visitor_rank)
    dataset.iloc[index] = row

X_homehigher = dataset[["HomeLastWin", "VisitorLastWin","HomeTeamRanksHigher"]].values

clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_homehigher, y_true,scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

last_match_winner = defaultdict(int)
dataset["HomeTeamWonLast"] = 0

for index, row in dataset.iterrows():

    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    teams = tuple(sorted([home_team, visitor_team]))
    
    row["HomeTeamWonLast"] = 1 if last_match_winner[teams] ==row["Home Team"] else 0
    dataset.loc[index] = row
    
    winner = row["Home Team"] if row["HomeWin"] else row["Visitor Team"]

    last_match_winner[teams] = winner
X_lastwinner = dataset[["HomeTeamRanksHigher", "HomeTeamWonLast"]].values
clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_lastwinner, y_true,scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))


from sklearn.preprocessing import LabelEncoder
encoding = LabelEncoder()

encoding.fit(dataset["Home Team"].values)

home_teams = encoding.transform(dataset["Home Team"].values)
visitor_teams = encoding.transform(dataset["Visitor Team"].values)
X_teams = np.vstack([home_teams, visitor_teams]).T

from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder()

X_teams_expanded = onehot.fit_transform(X_teams).toarray()

clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_teams_expanded, y_true,scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=14)
scores = cross_val_score(clf, X_teams, y_true, scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

X_all = np.hstack([X_homehigher, X_teams])
clf = RandomForestClassifier(random_state=14)
scores = cross_val_score(clf, X_all, y_true, scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

parameter_space = {
 "max_features": [2, 10, 'sqrt'],
 "n_estimators": [100,],
 "criterion": ["gini", "entropy"],
 "min_samples_leaf": [2, 4, 6],
}
clf = RandomForestClassifier(random_state=14)
grid = GridSearchCV(clf, parameter_space)
grid.fit(X_all, y_true)
print("Accuracy: {0:.1f}%".format(grid.best_score_ * 100))

print(grid.best_estimator_)

RandomForestClassifier(
    criterion='entropy',
    max_features=2,
    min_samples_leaf=6,
    n_estimators=100,
    random_state=14
)




#  新增特徵 

print("\n--- Adding New Engineered Features ---")


dataset["Date"] = pd.to_datetime(dataset["Date"])
from collections import defaultdict
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


# DaysSinceLastMatch: 球隊休息天數
last_played = defaultdict(lambda: pd.Timestamp('1900-01-01'))
days_since_home, days_since_visitor = [], []

for index, row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    date = row["Date"]

    days_since_home.append((date - last_played[home_team]).days)
    days_since_visitor.append((date - last_played[visitor_team]).days)


    last_played[home_team] = date
    last_played[visitor_team] = date

dataset["DaysSinceHome"] = days_since_home
dataset["DaysSinceVisitor"] = days_since_visitor



# Last5WinRatio: 最近5場比賽勝率

recent_results = defaultdict(list)
home_win_ratio, visitor_win_ratio = [], []

for index, row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]

    home_win_ratio.append(np.mean(recent_results[home_team][-5:]) if recent_results[home_team] else 0.5)
    visitor_win_ratio.append(np.mean(recent_results[visitor_team][-5:]) if recent_results[visitor_team] else 0.5)

    recent_results[home_team].append(1 if row["HomeWin"] else 0)
    recent_results[visitor_team].append(0 if row["HomeWin"] else 1)

dataset["HomeRecentWinRatio"] = home_win_ratio
dataset["VisitorRecentWinRatio"] = visitor_win_ratio


# 客場球隊在特定主場的勝率

away_performance = defaultdict(list)
visitor_in_venue = []

for index, row in dataset.iterrows():
    visitor_team = row["Visitor Team"]
    home_team = row["Home Team"]

    key = (visitor_team, home_team)
    history = away_performance[key]
    visitor_in_venue.append(np.mean(history) if history else 0.5)

    # 更新紀錄
    result = 0 if row["HomeWin"] else 1
    away_performance[key].append(result)

dataset["VisitorVenueWinRate"] = visitor_in_venue



# 使用新的特徵重新訓練模型

feature_cols = [
    "HomeLastWin", "VisitorLastWin", "HomeTeamRanksHigher",
    "HomeRecentWinRatio", "VisitorRecentWinRatio",
    "DaysSinceHome", "DaysSinceVisitor", "VisitorVenueWinRate"
]

X_new = dataset[feature_cols].fillna(0).values
y_true = dataset["HomeWin"].values

clf = RandomForestClassifier(random_state=14)
scores = cross_val_score(clf, X_new, y_true, scoring='accuracy', cv=5)
print("\nAccuracy (with new engineered features): {0:.1f}%".format(np.mean(scores)*100))

