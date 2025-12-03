import numpy as np
import csv
import os
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

download = r"C:\Users\User\Downloads"
data_filename = os.path.join(download, "Ionosphere","ionosphere.data")

X = np.zeros((351, 34), dtype='float')
y = np.zeros((351,), dtype='bool')

with open(data_filename, 'r') as input_file:
    reader = csv.reader(input_file)
    for i, row in enumerate(reader):
        data = [float(datum) for datum in row[:-1]]
        X[i] = data
        y[i] = row[-1] == 'g'

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14)

from sklearn.neighbors import KNeighborsClassifier
estimator = KNeighborsClassifier()

estimator.fit(X_train, y_train)

y_predicted = estimator.predict(X_test)
accuracy = np.mean(y_test == y_predicted) * 100
print("The accuracy is {0:.1f}%".format(accuracy))

scores = cross_val_score(estimator, X, y, scoring='accuracy')
average_accuracy = np.mean(scores) * 100
print("The average accuracy is {0:.1f}%".format(average_accuracy))

avg_scores = []
all_scores = []
parameter_values = list(range(1, 21)) # Include 20
for n_neighbors in parameter_values:
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(estimator, X, y, scoring='accuracy')
    avg_scores.append(np.mean(scores))
    all_scores.append(scores)


from matplotlib import pyplot as plt 
plt.plot(parameter_values,avg_scores, '-o')
plt.show()

X_broken = np.array(X)
X_broken[:,::2] /= 10

estimator = KNeighborsClassifier()
original_scores = cross_val_score(estimator, X, y,
 scoring='accuracy')
print("The original average accuracy for is{0:.1f}%".format(np.mean(original_scores) * 100))
broken_scores = cross_val_score(estimator, X_broken, y,
 scoring='accuracy')
print("The 'broken' average accuracy for is{0:.1f}%".format(np.mean(broken_scores) * 100))

X_transformed = MinMaxScaler().fit_transform(X_broken)
estimator = KNeighborsClassifier()
transformed_scores = cross_val_score(estimator, X_transformed, y,scoring='accuracy')
print("The average accuracy for is {0:.1f}%".format(np.mean(transformed_scores) * 100))

from sklearn.pipeline import Pipeline

scaling_pipeline = Pipeline([('scale', MinMaxScaler()),('predict', KNeighborsClassifier())])

scores = cross_val_score(scaling_pipeline, X_broken, y,scoring='accuracy')
print("The pipeline scored an average accuracy for is {0:.1f}%".format(np.mean(transformed_scores) * 100))


#第三題
from sklearn.metrics import accuracy_score

def condensed_nn(X_train, y_train):
    condensed_X = []
    condensed_y = []

    for label in np.unique(y_train):
        idx = np.where(y_train == label)[0][0]
        condensed_X.append(X_train[idx])
        condensed_y.append(y_train[idx])

    condensed_X = np.array(condensed_X)
    condensed_y = np.array(condensed_y)

    changed = True
    while changed:
        changed = False
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(condensed_X, condensed_y)

        for i in range(len(X_train)):
            x_i = X_train[i].reshape(1, -1)
            y_i = y_train[i]
            y_pred = knn.predict(x_i)[0]

            if y_pred != y_i:
                condensed_X = np.vstack([condensed_X, X_train[i]])
                condensed_y = np.append(condensed_y, y_i)
                changed = True

    return condensed_X, condensed_y


condensed_X, condensed_y = condensed_nn(X_train, y_train)


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(condensed_X, condensed_y)
y_pred = knn.predict(X_test)
condensed_accuracy = accuracy_score(y_test, y_pred) * 100
scores = cross_val_score(KNeighborsClassifier(n_neighbors=1), condensed_X, condensed_y, cv=10)

print(f"第三題 condensed prototypes 訓練 1-NN：")
print(f"原始訓練樣本數：{len(X_train)}")
print(f"Condensed 選出樣本數：{len(condensed_X)}")
print(f"測試集準確率：{condensed_accuracy:.2f}%")
print("Condensed set 上交叉驗證平均準確率：", np.mean(scores)*100)