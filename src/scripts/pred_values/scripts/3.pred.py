import datasets
import random
import pickle
import sys
import csv
with open(sys.argv[2], 'rb') as picklefile:
    train_x = pickle.load(picklefile)
    train_y = pickle.load(picklefile)
    dev_x = pickle.load(picklefile)
    dev_y = pickle.load(picklefile)


if sys.argv[1] == 'nb':
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
if sys.argv[1] == 'rf':
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=8446)

if len(sys.argv) > 3 and sys.argv[3] == 'bin':
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
    #print(dev_y[:100])
print(len(set(train_y+dev_y)))
print(len(train_x))
print('training')

clf.fit(train_x, train_y)

print('predicting')
pred_y = clf.predict(dev_x)
print(set(train_y))
cor = 0
for pred, gold in zip(pred_y, dev_y):
    if pred == gold:
        cor += 1
    else:
        print(pred, gold)

print(cor/len(dev_y))


