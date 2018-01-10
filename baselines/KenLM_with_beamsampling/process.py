import cPickle as pickle

# dataset_file_location = '../Datasets/eo/Dataset/with-Surface-Forms/splitDataset_with_targets.p'
dataset_file_location = '../Datasets/ar/Dataset/with-Surface-Forms/splitDataset_with_targets.p'


with open(dataset_file_location, 'rb') as f:
    dataset = pickle.load(f)

for i in range(0, len(dataset['train']['summary_with_surf_forms_and_types'])):
    print(dataset['train']['original_summary'][i].replace('<start> ', '').replace(' <end>', '').encode('utf-8'))
    # Comment-in for using surface form tuples and property placeholders.
    # print(dataset['train']['summary_with_surf_forms_and_types'][i].replace('<start> ', '').replace(' <end>', '').encode('utf-8'))