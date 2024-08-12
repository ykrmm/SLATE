import torch 
import numpy as np
import pandas as pd
import pickle
import argparse
from os.path import join,isfile

def read_dynamic(path: str) -> tuple[torch.LongTensor,torch.FloatTensor]:
    """
    Read the dynamic graph from the file.
    (u,v,t,w) is the edge from node u to node v at time t with weight w.
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    n = len(lines)
    dynamic_graph = torch.zeros((n, 3) ,dtype=torch.long)
    weights = torch.zeros(n)
    for i,line in enumerate(lines):
        line = line.split(',')
        u = int(line[0])
        v = int(line[1])
        t = int(line[2])
        w = float(line[3])
        dynamic_graph[i] = torch.LongTensor([u, v, t])
        weights[i] = w
    return (dynamic_graph,weights)

def read_dynamic_dgb(path:str) -> tuple[torch.LongTensor,torch.FloatTensor]:
    """
    Read the dynamic graph from the csv ml file.
    """
    df_data = pd.read_csv(path)
    src = torch.LongTensor(df_data.u.values)
    dst = torch.LongTensor(df_data.i.values)
    ts = df_data.ts.values
    convert_t = {t:i for i,t in enumerate(np.unique(ts))}
    convert_node = {n:i for i,n in enumerate(np.unique(np.concatenate((src,dst))))}
    src = torch.LongTensor([convert_node[int(n)] for n in src])
    dst = torch.LongTensor([convert_node[int(n)] for n in dst])
    ts = torch.LongTensor([convert_t[t] for t in ts])
    weights = torch.FloatTensor(df_data.label.values)
    dynamic_graph = torch.stack([src,dst,ts],dim=1)
    return (dynamic_graph,weights)

def read_features(path: str,T: int) -> torch.Tensor:
    """
    Read the features of the nodes from the file.
    T is the total number of time steps.
    """
    with open(path, 'r') as f:
        lines = f.read().splitlines() 
    n = len(lines)
    features = torch.zeros((n, T), dtype=torch.float)
    for i,line in enumerate(lines):
        line = line.split(',')
        features[i] = torch.tensor([float(x) for x in line[2:]])
    # Normalize the features
    #features = torch.nn.functional.normalize(features,dim=1)

    return features


def split_train_test(datadir,dataname,train_ratio: float = 0.8):
    """
    Split the dataset into train and test. Create numpy arrays for train and test containing nodes and labels.
    Process the data to make sure that labels in test are also in train. 
    """

    label_path = join(datadir,'Node-Label.csv')
    features_path = join(datadir,dataname,'Node_Features.txt')

    if isfile(join(datadir,dataname,'train_classif.npy')) and isfile(join(datadir,dataname,'test_classif.npy')):
        print('Found existing train and test files for classification in',join(datadir,dataname))
        return
    
    elif isfile(join(datadir,dataname,'train_classif.txt')) and isfile(join(datadir,dataname,'test_classif.txt')):
        print('Founding existing .txt files for classification in',join(datadir,dataname))
        
        train_nodes_all = []
        train_labels_all = []
        test_nodes_all = []
        test_labels_all = []

        lines = open(join(datadir,dataname,'train_classif.txt'), "r").readlines()
        for l in lines:
            line = l.split(',')
            node, label = line[0], line[1]
            train_nodes_all.append(node)
            train_labels_all.append(label)
        np.save(join(datadir,dataname,'train_classif.npy'), (np.array(train_nodes_all),np.array(train_labels_all)))

        lines = open(join(datadir,dataname,'test_classif.txt'), "r").readlines()
        for l in lines:
            line = l.split(',')
            node, label = line[0], line[1]
            test_nodes_all.append(node)
            test_labels_all.append(label)
        np.save(join(datadir,dataname,'test_classif.npy'), (np.array(test_nodes_all),np.array(test_labels_all)))

        print('Successfully created numpy train and test files for classification in',join(datadir,dataname))
            
    
    df_labels = pd.read_csv(label_path,delimiter=';')
    df_features = pd.read_csv(features_path,header=None)
    node_id_to_name = dict() # Node id to node name
    name_to_label = dict() # Node name to label name
    unique_label = set() # Set of unique labels
    
    for i,id in enumerate(df_features[0].values):
        node_id_to_name[id] = str(df_features[1].values[i])

    for i,name in enumerate(df_labels['Node']):
        name_to_label[name] = df_labels['Label'][i]

    for n,l in name_to_label.items():
        if l is not np.nan:
            unique_label.add(l)

    if isfile(join(datadir,'label_to_id.pkl')): # To share the same label id between all the datasets
        with open(join(datadir,'label_to_id.pkl'),'rb') as f:
            print('loading existing label_to_id.pkl')
            label_to_id = pickle.load(f)
    else:
        label_to_id = dict()
        for i,l in enumerate(unique_label):
            label_to_id[l] = i
        with open(join(datadir,'label_to_id.pkl'),'wb') as f:
            pickle.dump(label_to_id,f)

    node_label = [] # node id to label id
    for i,node in enumerate(df_features[0].values):
        try:
            node_label.append([node,label_to_id[name_to_label[node_id_to_name[node]]]])
        except: 
            node_label.append([node,-1]) # -1 is the label for unlabelled nodes
    node_label = np.array(node_label)
    unique_label_dts,count = np.unique(node_label[:,1],return_counts=True)   

    # Take train_ratio of each label in the dataset 
    train_nodes_all = []
    train_labels_all = []
    test_nodes_all = []
    test_labels_all = []
    for i,l_id in enumerate(unique_label_dts):
        if l_id != -1 and count[i] > 5: # at least 5 nodes for each label
            nb_train = int(train_ratio*count[i])
            train_nodes = np.random.choice(np.array(node_label)[node_label[:,1]==l_id][:,0],nb_train,replace=False)
            train_labels = np.array([l_id]*nb_train)
            test_nodes = np.array(node_label)[node_label[:,1]==l_id][:,0][~np.isin(np.array(node_label)[node_label[:,1]==l_id][:,0],train_nodes)]
            test_labels = np.array([l_id]*len(test_nodes))
            if i == 0:
                train_nodes_all = train_nodes
                train_labels_all = train_labels
                test_nodes_all = test_nodes
                test_labels_all = test_labels
            else:
                train_nodes_all = np.concatenate((train_nodes_all,train_nodes))
                train_labels_all = np.concatenate((train_labels_all,train_labels))
                test_nodes_all = np.concatenate((test_nodes_all,test_nodes))
                test_labels_all = np.concatenate((test_labels_all,test_labels))

        

    
    np.save(join(datadir,dataname,'train_classif.npy'), (train_nodes_all,train_labels_all))
    np.save(join(datadir,dataname,'test_classif.npy'), (test_nodes_all,test_labels_all))


    # writing train_node all and train_labels_all in .txt files for reproducibility
    with open(join(datadir,dataname,'train_classif.txt'),'w') as f:
        for i,n in enumerate(train_nodes_all):
            f.write(str(int(n))+','+str(int(train_labels_all[i]))+'\n')

    with open(join(datadir,dataname,'test_classif.txt'),'w') as f:
        for i,n in enumerate(test_nodes_all):
            f.write(str(int(n))+','+str(int(test_labels_all[i]))+'\n')

    print('train and test successfully saved in ',join(datadir,dataname,'train_classif.npy'),' and test_classif.npy')



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--datadir', type=str, default='/share/DEEPLEARNING/datasets/graph_datasets/DPPIN/', help='Path to the datadir of DPPIN')
    parser.add_argument('--dataname', type=str, default='DPPIN-Ito', help='Name of the dataset')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of train nodes for each label')
    args = parser.parse_args()

    
    split_train_test(args.datadir,args.dataname,args.train_ratio)