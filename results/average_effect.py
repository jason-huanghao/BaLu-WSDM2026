import os
import torch
import json

datasets = ['Syn_M=None_SimRel=1_Rel=4',  'BlogCatalog1_M=20_SimRel=1_Rel=4', 'Flickr1_M=20_SimRel=1_Rel=4', 'Youtube_M=20_SimRel=1_Rel=4']
exp_dir = '/mnt/vast-kisski/projects/kisski-tib-activecl/BaLu/datasets/exps/'

rest = {dataset: {} for dataset in datasets}

for data_dir in os.listdir(exp_dir):
    if data_dir not in datasets:
        continue
    files_dir = os.path.join(exp_dir, data_dir, 'full')
    for fn in os.listdir(files_dir):
        data = torch.load(os.path.join(files_dir, fn), weights_only=False)
        rest[data_dir][fn[:-3]] = data.true_effect.mean().item() 

try:
    with open("ave_effect.json", 'w') as json_file:
        json.dump(rest, json_file, indent=4)  # indent for pretty printing
    print(f"Successfully saved dictionary to ave_effect.json")
except IOError:
    print(f"Error: Could not write to file ave_effect.json")



