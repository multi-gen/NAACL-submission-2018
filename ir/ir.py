#######################################
# Baseline for running Information Retrieval baseline
# python ir.py -i ../../data/ -p eo -o ../../results/ir/eo_ir_test.csv -setup test
# python ir.py -i ../../data/ -p eo -o ../../results/ir/eo_ir_valid.csv -setup validate
# python ir.py -i ../../data/ -p ar -o ../../results/ir/ar_ir_test.csv -setup test
# python ir.py -i ../../data/ -p ar -o ../../results/ir/ar_ir_valid.csv -setup validate
# python ir.py -i ../../data/ -t -p eo -o ../../results/ir/eo_template_ir_test.csv -setup test
# python ir.py -i ../../data/ -t -p eo -o ../../results/ir/eo_template_ir_valid.csv -setup validate
# python ir.py -i ../../data/ -t -p ar -o ../../results/ir/ar_template_ir_test.csv -setup test
# python ir.py -i ../../data/ -t -p ar -o ../../results/ir/ar_template_ir_valid.csv -setup validate
#######################################

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

parser = argparse.ArgumentParser(description='train and generate sentences for Information retrieval baseline')
parser.add_argument('-i', '--inputfolder', help='input folder where data is there', required=True)
parser.add_argument('-p', '--prefix', help='ar or eo', required=True)
parser.add_argument('-t', '--template', help='either use templates or not using it', action='store_true')
parser.add_argument('-o', '--out', help='output file path to save all  files in', required=True)
parser.add_argument('-setup', '--setup', help='test or validate', required=True)
args = parser.parse_args()

_K_ = 5
_RADIUS_ = 0.4
_N_COMPONENTS_ = 200

def preprocess_triples(i):

    s = i.replace("http://www.wikidata.org/prop/direct/", "")
    s = s.replace("http://www.wikidata.org/entity/", "")
    s = s[1:-1].split()
    s = " ".join(s)
    return s

print("loading training data..")
train = pd.read_csv(os.path.join(args.inputfolder, './%s_train.csv' % args.prefix), encoding="utf-8")

train['triples_prep'] = train.apply(lambda i: preprocess_triples(i['triples']), axis=1)
train['Number of Triples'] = train.apply(lambda i: len(i['triples'][1:-1].split(",")), axis=1)

print("loading test data..")
test = pd.read_csv(os.path.join(args.inputfolder, './%s_%s.csv' % (args.prefix, args.setup)), encoding="utf-8")
test['triples_prep'] = test.apply(lambda i: preprocess_triples(i['triples']), axis=1)
test['Number of Triples'] = test.apply(lambda i: len(i['triples'].split(",")), axis=1)

# VECTORIZATION
print("vectorization..")
X = np.concatenate([train['triples_prep'], test['triples_prep']])

if args.template:
    X_text = np.concatenate([train['summary_prep'], test['summary_prep']])
else:
    X_text = np.concatenate([train['Target'], test['Target']])

count_vect = CountVectorizer().fit(X[:train.shape[0]])
X = count_vect.transform(X)

tf_transformer = TfidfTransformer().fit(X[:train.shape[0]])
X = tf_transformer.transform(X)

svd = TruncatedSVD(n_components=_N_COMPONENTS_).fit(X[:train.shape[0]])
X = svd.transform(X)

# CLUSTERING
print("clustering..")
neigh = NearestNeighbors(_K_, _RADIUS_)
neigh.fit(X[:train.shape[0]])  # clustering only training set

# TESTING
print("testing..")
# iterating over the test set
def get_text(v):

    dist, id = neigh.kneighbors(v, 1)
    text = X_text[id[0][0]]    # pick the closes summary from the training set
    return text

y = []
for v in X[train.shape[0]:]:
    y.append(get_text(v))


test['Target'] = X_text[train.shape[0]:]
test['Summary 1'] = y

test.to_csv(args.out, columns=["Main-Item", "Number of Triples", "Summary 1", "Target"], index=False, encoding="utf-8")

