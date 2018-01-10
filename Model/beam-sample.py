
# coding: utf-8

# In[1]:


import json
import cPickle as pickle
import numpy as np
import h5py
import pandas as pd
import re
import random


# In[28]:


# IMPORTANT: Leave the batch size unchanged
# It's the one with which we trained the model, and it should be 
# the same with the one of the pre-trained model.
batch_size = 85

state = 'test'
# state = 'validate'
if state == 'test':
    summaries_filename = './Summaries-Testing-Beam-4.h5'
elif state == 'validate':
    summaries_filename = './Summaries-Validation-Beam-4.h5'
else:
    raise Exception('The state variable can be equal either to test or validation!')
    
# IMPORTANT: The language of the dataset location should match the language of the sampled summaries file
# that is defined above.
dataset_load_location = './data/ar/'


# In[22]:


summaries = h5py.File(summaries_filename, 'r')
beam_size = int(re.findall(r'-Beam-(.*?).h5', summaries_filename)[0])
dataset_dump_location = './%s_GRU_%s_Beam_%s' % (dataset_load_location[:-1].replace('./data/', ''), state, str(beam_size))


# In[15]:


summaries_dictionary = dataset_load_location + 'summaries_dictionary.json'
with open(summaries_dictionary, 'r') as f:
    dictionary = json.load(f, 'utf-8')
    id2word = dictionary['id2word']
    id2word = {int(key): id2word[key] for key in id2word}
    word2id = dictionary['word2id']
    f.close()

triples_dictionary = dataset_load_location + 'triples_dictionary.json'
with open(triples_dictionary, 'r') as f:
    dictionary = json.load(f, 'utf-8')
    id2item = dictionary['id2item']
    id2item = {int(key): id2item[key] for key in id2item}
    item2id = dictionary['item2id']
    numAlignedTriples = dictionary['max_num_triples']
    f.close()
    
# Loading supporting inverse dictionaries for surface forms and instance types
with open(dataset_load_location + 'inv_surf_forms_dictionary.json', 'r') as f:
    inv_surf_forms_tokens = json.load(f, encoding='utf-8')
with open(dataset_load_location + 'inv_instance_types_with_predicates.json', 'r') as f:
    inv_instancetypes_with_pred_dict = json.load(f, encoding='utf-8')
with open(dataset_load_location + 'splitDataset.p', 'rb') as f:
    splitDataset = pickle.load(f)
with open(dataset_load_location + 'surf_forms_counts.p', 'rb') as f:
    surf_form_counts = pickle.load(f)
with open(dataset_load_location + 'labels_dict.p', 'rb') as f:
    labels = pickle.load(f)


# In[16]:


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


# In[17]:


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


# In[18]:


output = {'Main-Item': [], 'Target': []}
# for beamidx in range(0, beam_size):
for beamidx in range(0, 1):
    output[('Summary %d' % (beamidx + 1))] = []


# In[19]:


for batchidx in range(0, len(summaries['triples'])):
    print('Generating summaries for %d. Batch...' % (batchidx + 1))
    for instance in range(0, batch_size):
        splitDatasetIndex = int(np.round(instance * len(splitDataset[state]['item']) / float(batch_size)) + batchidx)
        mainItem = splitDataset[state]['item'][splitDatasetIndex]
        output['Main-Item'].append(mainItem)
        
        output['Target'].append(splitDataset[state]['actual_target'][splitDatasetIndex])
        # We are only sampling from the first (i.e. most probable) beam.
        for beamidx in range(0, 1):
            expressed_triples = []
            summary = ''
            i = 0
            while summaries['summaries'][beamidx][batchidx * batch_size + instance][i] != word2id['<end>']:
                summary += ' ' + token_to_word(id2word[summaries['summaries'][beamidx][batchidx * batch_size + instance][i]],                                               mainItem,                                               splitDataset[state]['triples'][splitDatasetIndex],                                               expressed_triples)
                if i == len(summaries['summaries'][beamidx][batchidx * batch_size + instance]) - 1:
                    break
                else:
                    i += 1       
            summary += ' ' + token_to_word(id2word[summaries['summaries'][beamidx][batchidx * batch_size + instance][i]],                                           mainItem,                                           splitDataset[state]['triples'][splitDatasetIndex],                                           expressed_triples)
            output['Summary %d' % (beamidx + 1)].append(summary[1:])


# In[23]:


output_filename = dataset_dump_location + '.csv'
out_df = pd.DataFrame(output)
out_df.to_csv(output_filename, index=False, encoding = 'utf-8')

