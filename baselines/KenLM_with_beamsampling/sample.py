
# coding: utf-8

# In[1]:

import json
import cPickle as pickle
import numpy as np
import h5py
import numpy
import pandas as pd
from evaluator.eval import Evaluator


# In[2]:
num_samples = 10
batch_size = 85
state = 'validate'
# summaries_type = 'summary_with_URIs'
summaries_type = 'summary_with_surf_forms'
dataset_load_location = '../Datasets/eo/Dataset/'
processed_dataset_location = dataset_load_location + 'with-URIs/' if summaries_type == 'summary_with_URIs' else  dataset_load_location + 'with-Surface-Forms/'



# In[3]:
if state == 'test':
    if 'ar' in dataset_load_location:
        if summaries_type == 'summary_with_URIs':
            summaries_filename = '../triples-to-gru/Checkpoints/ar/with-URIs/Summaries-Testing-Beam-4.h5'
        elif summaries_type == 'summary_with_surf_forms':
            summaries_filename = '../triples-to-gru/Checkpoints/ar/with-Surface-Forms/Summaries-Testing-Beam-4.h5'
    elif 'eo' in dataset_load_location:
        if summaries_type == 'summary_with_URIs':
            summaries_filename = '../triples-to-gru/Checkpoints/eo/with-URIs/Summaries-Testing-Beam-4.h5'
        elif summaries_type == 'summary_with_surf_forms':
            summaries_filename = '../triples-to-gru/Checkpoints/eo/with-Surface-Forms/Summaries-Testing-Beam-4.h5'
elif state == 'validate':
    if 'ar' in dataset_load_location:
        if summaries_type == 'summary_with_URIs':
            summaries_filename = '../triples-to-gru/Checkpoints/ar/with-URIs/Summaries-Validation-Beam-4.h5'
        elif summaries_type == 'summary_with_surf_forms':
            summaries_filename = '../triples-to-gru/Checkpoints/ar/with-Surface-Forms/Summaries-Validation-Beam-4.h5'
    elif 'eo' in dataset_load_location:
        if summaries_type == 'summary_with_URIs':
            summaries_filename = '../triples-to-gru/Checkpoints/eo/with-URIs/Summaries-Validation-Beam-5.h5'
        elif summaries_type == 'summary_with_surf_forms':
            summaries_filename = '../triples-to-gru/Checkpoints/eo/with-Surface-Forms/Summaries-Validation-Beam-4.h5'        
summaries = h5py.File(summaries_filename, 'r')

# Loading the sampled KenLM tempates.
summary_templates_location = './ar_without_surf_forms_templates.p' if 'ar' in dataset_load_location else './eo_without_surf_forms_templates.p'
with open(summary_templates_location, 'rb') as f:
    summary_templates = pickle.load(f)
    templates = summary_templates['sentences']
    prob_distribution = summary_templates['prob_distribution']


print (state)
print (dataset_load_location)
print (processed_dataset_location)
print (summaries_filename)

# In[4]:

summaries_dictionary = processed_dataset_location + 'summaries_dictionary.json'
with open(summaries_dictionary, 'r') as f:
    dictionary = json.load(f, 'utf-8')
    id2word = dictionary['id2word']
    id2word = {int(key): id2word[key] for key in id2word}
    word2id = dictionary['word2id']
    f.close()
    
# triples_dictionary = processed_dataset_location + 'triples_dictionary.json'
# with open(triples_dictionary, 'r') as f:
#     dictionary = json.load(f, 'utf-8')
#     id2item = dictionary['id2item']
#     id2item = {int(key): id2item[key] for key in id2item}
#     item2id = dictionary['item2id']
#     numAlignedTriples = dictionary['max_num_triples']
#     f.close()
    
# Loading supporting inverse dictionaries for surface forms and instance types
with open(processed_dataset_location + 'inv_surf_forms_dictionary.json', 'r') as f:
    inv_surf_forms_tokens = json.load(f, encoding='utf-8')
with open(processed_dataset_location + 'inv_instance_types_with_predicates.json', 'r') as f:
    inv_instancetypes_with_pred_dict = json.load(f, encoding='utf-8')
with open(processed_dataset_location + 'splitDataset_with_targets.p', 'rb') as f:
    splitDataset = pickle.load(f)
# with open(processed_dataset_location + 'inv_instance_types_dictionary.json', 'r') as f:
#     inv_instancetypes_dict = json.load(f, encoding='utf-8')
with open(dataset_load_location + 'labels_dict.p', 'rb') as f:
    labels = pickle.load(f)
    
if summaries_type == 'summary_with_URIs':
    with open(dataset_load_location + 'URI-Counts.p', 'rb') as f:
        surf_form_counts = pickle.load(f)
elif summaries_type == 'summary_with_surf_forms':
    with open(dataset_load_location + 'Surface-Forms-Counts.p', 'rb') as f:
        surf_form_counts = pickle.load(f)
    



# ```python
# surf_form_counts[u'http://www.wikidata.org/entity/Q46611']: {u'Apollo-Programo': 10, u'Projekto Apollo': 6, u'projekto Apollo': 2}
# inv_surf_forms_tokens[u'#surFormToken71849']: [u'http://www.wikidata.org/entity/Q832222', u'Caprivi-streko']
# inv_instancetypes_with_pred_dict[u'#instanceTypeWithPredicate11']: u'http://www.wikidata.org/prop/direct/P138'
# ```

# In[6]:

most_frequent_surf_form = {}
for entity in surf_form_counts:
    most_frequent_surf_form[entity] = sorted(surf_form_counts[entity], key=lambda k: surf_form_counts[entity][k], reverse=True)[0]

def match_predicate_to_entity(token, triples, expressed_triples):
    matched_entities = []
    
    for tr in range(0, len(triples)):
        if tr not in expressed_triples:
            tempPredicate = triples[tr].split()[1]
            if tempPredicate == token:
                tempEntity = triples[tr].split()[-1]
                if tempEntity == "<item>":
                    tempEntity == triples[tr].split()[0]
                if tempEntity not in matched_entities:
                    matched_entities.append(tempEntity.decode('utf-8'))

    if len(matched_entities) == 0:
        token = '<resource>'
    else:
        
        random_selection = random.choice(matched_entities)
        while random_selection not in labels and len(matched_entities) > 1:
            matched_entities.remove(random_selection)
            random_selection = random.choice(matched_entities)
        if random_selection in labels:
            
            token = labels[random_selection]
            expressed_triples.append(random_selection)
        else:
            token = token
        
    return token


# In[16]:

def token_to_word(token, main_entity, triples, expressed_triples):
    main_entity = main_entity.decode('utf-8')
    if "#surFormToken" in token:
        word = inv_surf_forms_tokens[token[1:]][1] if "##surFormToken" in token else inv_surf_forms_tokens[token][1]
    elif "#instanceTypeWithPredicate" in token:
        # word = inv_instancetypes_with_pred_dict[token]
        word = match_predicate_to_entity(inv_instancetypes_with_pred_dict[token], triples, expressed_triples)
    elif "#instanceType" in token:
        word = inv_instancetypes_dict[token]
    elif token == "<item>":
        word = sorted(surf_form_counts[main_entity], key=lambda k: surf_form_counts[main_entity][k], reverse=True)[0]
    else:
        word = token
    return word


evaluations = {'CIDEr': [], 'Bleu_4': [], 'Bleu_3': [], 'Bleu_2': [], 'Bleu_1': [], 'ROUGE_L': [], 'METEOR': []}

# In[10]:

batch_size = 85
dataset_indices = ['train', 'validate', 'test']


for sampleidx in range(0, num_samples):
    output = {'Main-Item': [], 'Target': [], 'Random': []}
    for batchidx in range(0, len(summaries['triples'])):
    # for batchidx in range(0, 101):
        # print('Generating summaries for %d. Batch...' % (batchidx + 1))
        for instance in range(0, batch_size):
            # Pay attention to the Python division at the np.round() function -- can seriously mess things up!
            splitDatasetIndex = int(np.round(instance * len(splitDataset[state]['item']) / float(batch_size)) + batchidx)
            mainItem = splitDataset[state]['item'][splitDatasetIndex]
            output['Main-Item'].append(mainItem)


            output['Target'].append(splitDataset[state]['actual_target'][splitDatasetIndex].encode('utf-8'))



            # We will only be sampling from our training set.
            assert(len(prob_distribution) == len(templates))
            rand_instance = np.random.choice(np.arange(len(prob_distribution)), p=np.ndarray.tolist(prob_distribution))
            
            expressed_triples = []
            target = ['<start>']
            tempTarget = templates[rand_instance]
            print (tempTarget)
            for word in tempTarget.split():
                target.append(token_to_word(word,
                                            mainItem,
                                            splitDataset[state]['triples'][splitDatasetIndex],
                                            expressed_triples))
            target = target + ['<end>']
            output['Random'].append(' '.join(target).encode('utf-8'))
            
    e = Evaluator(output['Random'], output['Target'])
    e.evaluate()

    for method in e.overall_eval:
        evaluations[method].append(e.overall_eval[method])
        print(evaluations[method])

for method in evaluations:
    tempArray = numpy.asarray(evaluations[method])
    print('%s: %.3f (%.3f)' % (method, tempArray.mean() * 100, tempArray.std() * 100))

output_filename = ('./ar_without_surf_forms_%s.csv' % state) if 'ar' in dataset_load_location else ('./eo_without_surf_forms_%s.csv' % state)
out_df = pd.DataFrame(output)
out_df.to_csv(output_filename, index=False, encoding = 'utf-8')
