import os
import pandas as pd
import sys


data_folder = os.path.join(os.path.expanduser("~"), "Downloads", "ml-100k")
ratings_filename = os.path.join(data_folder, "u.data")

all_ratings = pd.read_csv(ratings_filename, delimiter="\t", header=None, names=["UserID", "MovieID", "Rating", "Datetime"])
all_ratings["Datetime"] = pd.to_datetime(all_ratings['Datetime'], unit='s')
print(all_ratings[:5])

all_ratings["Favorable"] = all_ratings["Rating"] > 3
all_ratings[10:15]
print(all_ratings[10:15])

ratings = all_ratings[all_ratings['UserID'].isin(range(200))]

favorable_ratings = ratings[ratings["Favorable"]]

favorable_reviews_by_users = dict((k, frozenset(v.values))
 for k, v in favorable_ratings.groupby("UserID")["MovieID"])

num_favorable_by_movie = ratings[["MovieID", "Favorable"]].groupby("MovieID").sum()

num_favorable_by_movie.sort_values("Favorable", ascending=False)[:5]

print(num_favorable_by_movie.sort_values("Favorable", ascending=False)[:5])

#aprior
frequent_itemsets = {}
min_support = 50

frequent_itemsets[1] = dict((frozenset((movie_id,)),row["Favorable"])
    for movie_id, row in num_favorable_by_movie.iterrows()
    if row["Favorable"] > min_support)

from collections import defaultdict
def find_frequent_itemsets(favorable_reviews_by_users, k_1_itemsets,min_support):
    counts = defaultdict(int)
    
    for user, reviews in favorable_reviews_by_users.items():
        for itemset in k_1_itemsets:
            if itemset.issubset(reviews):
                for other_reviewed_movie in reviews - itemset:
                    current_superset = itemset | frozenset((other_reviewed_movie,))
                    counts[current_superset] += 1
    return dict([(itemset, frequency) for itemset, frequency in counts.items() if frequency >= min_support])

for k in range(2, 20):
    cur_frequent_itemsets =find_frequent_itemsets(favorable_reviews_by_users,frequent_itemsets[k-1],min_support)
    frequent_itemsets[k] = cur_frequent_itemsets
    if len(cur_frequent_itemsets) == 0:
        print("Did not find any frequent itemsets of length {}".format(k))
        sys.stdout.flush()
        break
    else:
        print("I found {} frequent itemsets of length{}".format(len(cur_frequent_itemsets), k))
        sys.stdout.flush()
del frequent_itemsets[1]

candidate_rules = []
for itemset_length, itemset_counts in frequent_itemsets.items():
    for itemset in itemset_counts.keys():
        for conclusion in itemset:
            premise = itemset - set((conclusion,))
            candidate_rules.append((premise, conclusion))
print(candidate_rules[:5])

correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)

for user, reviews in favorable_reviews_by_users.items():
    for candidate_rule in candidate_rules:
        premise, conclusion = candidate_rule
        if premise.issubset(reviews):
            if premise.issubset(reviews):
                if conclusion in reviews:
                    correct_counts[candidate_rule] += 1
                else:
                    incorrect_counts[candidate_rule] += 1

rule_confidence = {candidate_rule: correct_counts[candidate_rule]/ float(correct_counts[candidate_rule] +incorrect_counts[candidate_rule])
    for candidate_rule in candidate_rules}

from operator import itemgetter

sorted_confidence = sorted(rule_confidence.items(),
    key=itemgetter(1), reverse=True)
for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    print("Rule: If a person recommends {0} they will alsorecommend {1}".format(premise, conclusion))
    print(" - Confidence:{0:.3f}".format(rule_confidence[(premise, conclusion)]))
    print("")


movie_name_filename = os.path.join(data_folder, "u.item")
movie_name_data = pd.read_csv(movie_name_filename, delimiter="|",header=None, encoding = "mac-roman")

movie_name_data.columns = ["MovieID", "Title", "Release Date","Video Release", "IMDB", "<UNK>", "Action", "Adventure","Animation", "Children's", "Comedy", "Crime", "Documentary","Drama", "Fantasy", "Film-Noir","Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller","War", "Western"]

def get_movie_name(movie_id):
    title_object = movie_name_data[movie_name_data["MovieID"] ==movie_id]["Title"]
    title = title_object.values[0]
    return title

for index in range(10):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0] 
    premise_names = ", ".join(get_movie_name(idx) for idx in premise)
    conclusion_name = get_movie_name(conclusion)
    print("Rule: If a person recommends {0} they will also recommend {1}".format(premise_names, conclusion_name))
    print(" - Confidence: {0:.3f}".format(rule_confidence[(premise,conclusion)]))
    print("")


test_dataset =all_ratings[~all_ratings['UserID'].isin(range(200))]
test_favorable = test_dataset[test_dataset["Favorable"]]
test_favorable_by_users = dict((k, frozenset(v.values)) for k, v in test_favorable.groupby("UserID")["MovieID"])

correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)
for user, reviews in test_favorable_by_users.items():
    for candidate_rule in candidate_rules:
        premise, conclusion = candidate_rule
        if premise.issubset(reviews):
            if conclusion in reviews:
                correct_counts[candidate_rule] += 1
            else:
                incorrect_counts[candidate_rule] += 1

test_confidence = {candidate_rule: correct_counts[candidate_rule]/ float(correct_counts[candidate_rule] + incorrect_counts[candidate_rule])
for candidate_rule in rule_confidence}
for index in range(10):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    premise_names = ", ".join(get_movie_name(idx) for idx in premise)
    conclusion_name = get_movie_name(conclusion)
    print("Rule: If a person recommends {0} they will also recommend {1}".format(premise_names, conclusion_name))
    print(" - Train Confidence:{0:.3f}".format(rule_confidence.get((premise, conclusion),-1)))
    print(" - Test Confidence:{0:.3f}".format(test_confidence.get((premise, conclusion),-1)))
    print("")


from math import sqrt
from scipy.stats import chi2_contingency

print("\n=== 前 10 名推薦規則的各種指標分析 ===\n")
print(f"{'Rule':60} | Conf  | Supp  | Lift  | Chi²   | AllConf | MaxConf | Kulcz | Cosine")
print("-" * 120)

total_users = len(favorable_reviews_by_users)

top_rules = [item[0] for item in sorted_confidence[:10]]

for premise, conclusion in top_rules:
    premise_users = set(
        user for user, reviews in favorable_reviews_by_users.items()
        if premise.issubset(reviews)
    )
    conclusion_users = set(
        user for user, reviews in favorable_reviews_by_users.items()
        if conclusion in reviews
    )
    both_users = set(
        user for user, reviews in favorable_reviews_by_users.items()
        if premise.issubset(reviews) and conclusion in reviews
    )

    a = len(both_users)
    b = len(premise_users - both_users)
    c = len(conclusion_users - both_users)
    d = total_users - a - b - c

    
    table = [[a, b], [c, d]]
    try:
        chi2, p_val, dof, expected = chi2_contingency(table)
    except:
        chi2 = 0  

    # 指標計算
    support = a / total_users if total_users else 0
    conf = rule_confidence.get((premise, conclusion), 0)
    conf_reverse = rule_confidence.get((frozenset([conclusion]), next(iter(premise))) if len(premise)==1 else (conclusion, next(iter(premise))), 0)
    supp_p = len(premise_users) / total_users
    supp_c = len(conclusion_users) / total_users
    lift = conf / supp_c if supp_c else 0
    all_conf = support / max(supp_p, supp_c) if max(supp_p, supp_c) else 0
    max_conf = max(conf, conf_reverse)
    kulczynski = 0.5 * (conf + (a / len(conclusion_users)) if len(conclusion_users) else 0)
    cosine = support / sqrt(supp_p * supp_c) if supp_p and supp_c else 0

    
    premise_names = ", ".join(get_movie_name(mid) for mid in premise)
    conclusion_name = get_movie_name(conclusion)
    rule_str = f"If [{premise_names}] → [{conclusion_name}]"

    print(f"{rule_str:60} | {conf:.3f} | {support:.3f} | {lift:.3f} | {chi2:6.2f} | {all_conf:.3f}  | {max_conf:.3f}   | {kulczynski:.3f} | {cosine:.3f}")
