

import os

for dataset in ['evs_wvs_data_2017_2022', 'evs_trend_data_1981_2017']:
    print(dataset)
    print(' setup & nb & rf \\\\')
    table = []
    for model in ['nb', 'rf']:
        for classes in ['', '.bin']:
            for setup in ['', '-filtered']:
                path = 'preds/' + model + '.' + dataset + setup + classes 
                if not os.path.isfile(path) or os.stat(path).st_size == 0:
                    score = 0.0
                else:
                    score = open(path).readlines()[-1].strip()
                    if score[0] != '0':
                        score = 0.0
                    else:
                        score = float(score)
                table.append((model + setup + classes, score*100))
    for i in range(4):
        print(table[i][0][3:].replace('out', '') + ' & {:.2f} '.format(table[i][1]) + '& {:.2f} \\\\'.format(table[4+i][1]))
    print()

