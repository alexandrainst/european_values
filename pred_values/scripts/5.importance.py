import datasets
import random
import pickle
import sys
import csv
import datasets
with open(sys.argv[1], 'rb') as picklefile:
    train_x = pickle.load(picklefile)
    train_y = pickle.load(picklefile)
    dev_x = pickle.load(picklefile)
    dev_y = pickle.load(picklefile)


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=8446)

european = set()
with open('all.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    first = True
    for row in reader:
        if first:
            first = False
            continue
        print(row[5], row[1])
        if row[5] == 'Europe':
            european.add(row[1])
    #print(european)
    #print(dev_y[:100])
train_y = [label in european for label in train_y]
dev_y = [label in european for label in dev_y]
print(len(set(train_y+dev_y)))
print(len(train_x))
print('training')

print(train_y.count(True), len(train_y))
clf.fit(train_x, train_y)

importances = clf.feature_importances_
data = datasets.load_dataset('EuropeanValuesProject/european_values_survey', 'evs_wvs_data_2017_2022', token=open('token').readline().strip())
data = data.remove_columns('question_e069_18_confidence__the_european_union')
data = data.remove_columns('question_e069_18a_confidence__major_regional_organization__combined_from_country_specific')

names = data.column_names['train'][3:]

scores = {}
for importance, name in zip(importances, names):
    scores[name] = importance

for item in sorted(scores.items(), key=lambda item: item[1]):
    print(item)



