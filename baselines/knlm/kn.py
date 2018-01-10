import os
import argparse
import pandas as pd

from nltk.corpus import PlaintextCorpusReader
from kneserney import *

parser = argparse.ArgumentParser(description='train and generate sentences for knserney language model baseline')
parser.add_argument('-i', '--inputfolder', help='input folder where data is there', required=True)
parser.add_argument('-p', '--prefix', help='ar or eo', required=True)
parser.add_argument('-t', '--template', help='either use templates or not using it', action='store_true')
parser.add_argument('-o', '--out', help='output file path to save all  files in', required=True)
args = parser.parse_args()

__NGRAMS__ = 5

print("loading training data..")
train = pd.read_csv(os.path.join(args.inputfolder, './%s_train.csv' % args.prefix), encoding="utf-8")

if args.template:
    sents = train['summary_prep'].values
else :
    sents = train['Target'].values

sents = [i.split() for i in sents]

print("training model  %s" % args.prefix)
model = KneserNeyNGram(sents, n=__NGRAMS__, D=float(0.0), corpus='eo')
g = NGramGenerator(model)
print("Done model")

print("Testing")
test = pd.read_csv(os.path.join(args.inputfolder, './%s_test.csv' % args.prefix), encoding="utf-8")

generated_summaries = []
for i in range(test.shape[0]):
    print("summary %s out of %s" % (i, test.shape[0]))

    generated_summary = None
    while generated_summary is None:
        try:
            generated_summary = g.generate_sent()
        except Exception as e:
            pass
    generated_summaries.append(" ".join(generated_summary))

test['Summary 1'] = generated_summaries

# instead of replacing the placeholders
if args.template:
    test['Target'] = test['summary_prep']

model.perplexity([i.split() for i in generated_summaries])

# test.to_csv(args.out, columns=["Main-Item", "Number of Triples", "Summary 1", "Target"], index=False)






