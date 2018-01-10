import pandas
import json
import pickle
import csv
import io

# 214 missing for Esperanto

data = pandas.read_csv('../../data/ar_validate.csv', encoding='utf-8')

en_abstracts = pickle.load(open("en_sentences.pkl", "rb"))

counter_no_en = 0

result = {}
for key, wdid in data['Main-Item'].iteritems():
    if wdid in en_abstracts:
        en_sentence = en_abstracts[wdid]
        result[wdid] = [data['Number of Triples'][key], en_sentence, data['Target'][key]]
    else:
        counter_no_en += 1

with open('translate_input_ar_validate.json', 'w') as csvfile:
    csvfile.write(json.dumps(result))

print counter_no_en
