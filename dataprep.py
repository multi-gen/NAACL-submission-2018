
import pickle
import pandas as pd


for k1 in ["ar"]:

    f = open('./%s/splitDataset_with_targets.p' % k1, 'rb')
    v1 = pickle.load(f)

    print("processing %s ..." % k1)
    for k2, v2 in v1.items():
        print("processing %s ..." % k2)

        df = pd.DataFrame(v2)
        df.rename(columns={'item': 'Main-Item',
                           'original_summary': 'Target',
                           'final_triples_with_types_reduced': 'triples',
                           'summary_with_surf_forms_and_types': 'summary_prep'
                           },
                  inplace=True
                  )

        df['Number of Triples'] = df.apply(lambda a: len(a['triples']), axis=1)
        df.to_csv("%s_%s.csv" % (k1, k2),
                  columns=['Main-Item', 'Target', 'triples', 'summary_prep', 'Number of Triples'],
                  index=False
                  )