import datasets
import random
import pickle

for dataset_name in ['evs_wvs_data_2017_2022', 'evs_trend_data_1981_2017']:
    print('loading ' + dataset_name )
    dataset = datasets.load_dataset('EuropeanValuesProject/european_values_survey', dataset_name, token=open('token').readline().strip())

    dataset = dataset.remove_columns('question_e069_18_confidence__the_european_union')
    dataset = dataset.remove_columns('question_e069_18a_confidence__major_regional_organization__combined_from_country_specific')
    
    tgt_column = 1
    
    print('transforming')
    data1 = []
    data2 = []
    for rowIdx, row in enumerate(dataset['train']):
        #if len(data) > 1000:
        #    break
        row= list(row.values())
        label = row[tgt_column]
        feats = row[3:len(row)]
        # TODO fix nicer?
        # Use None value somehow
        # or remove least common features
        if None not in feats:
            data1.append((feats, label))
        data2.append((feats, label))
    random.seed(8446)
    random.shuffle(data1)
    random.shuffle(data2)
    
    for data, path in zip([data1, data2], [dataset_name + '-filtered.pickle', dataset_name + '.pickle']):
        split1 = int(.8 * len(data))
        split2 = int(.9 * len(data))
        train_x = [x[0] for x in data[:split1]]
        train_y = [x[1] for x in data[:split1]]
    
        dev_x = [x[0] for x in data[split1:split2]]
        dev_y = [x[1] for x in data[split1:split2]]
        with open(path, 'wb') as picklefile:
            pickle.dump(train_x, picklefile)
            pickle.dump(train_y, picklefile)
            pickle.dump(dev_x, picklefile)
            pickle.dump(dev_y, picklefile)

