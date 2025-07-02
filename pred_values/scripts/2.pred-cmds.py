for classifier in ['nb', 'rf']:
    for dataset in ['evs_wvs_data_2017_2022', 'evs_trend_data_1981_2017']:
        for filtered in ['', '-filtered']:
            for binary in [' bin', '']:
                out = 'preds/' + classifier + '.' + dataset + filtered + binary.replace(' ', '.')
                cmd = 'python3 scripts/3.pred.py ' + classifier +' '  + dataset + filtered + '.pickle' + binary + ' > ' + out
                print(cmd)
