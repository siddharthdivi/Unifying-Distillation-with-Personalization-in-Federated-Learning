import json
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import trange
import numpy as np
import random
import pickle

# MNIST related params.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CHANNELS = 1

# CIFAR related params.
IMAGE_SIZE_CIFAR = 32
NUM_CHANNELS_CIFAR = 3

def suffer_data(data):
    data_x = data['x']
    data_y = data['y']
        # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)
    return (data_x, data_y)
    
def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)


def get_random_batch_sample(data_x, data_y, batch_size):
    num_parts = len(data_x)//batch_size + 1
    if(len(data_x) > batch_size):
        batch_idx = np.random.choice(list(range(num_parts +1)))
        sample_index = batch_idx*batch_size
        if(sample_index + batch_size > len(data_x)):
            return (data_x[sample_index:], data_y[sample_index:])
        else:
            return (data_x[sample_index: sample_index+batch_size], data_y[sample_index: sample_index+batch_size])
    else:
        return (data_x,data_y)


def get_batch_sample(data, batch_size):
    data_x = data['x']
    data_y = data['y']

    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    batched_x = data_x[0:batch_size]
    batched_y = data_y[0:batch_size]
    return (batched_x, batched_y)

# def read_mnist_data(args=None):
#     # Extract the required variables.
#     # Sample file name format: data_mnist_IID_False_U_10_K_4_R_0_DS_1 
    
#     if (args.dataset == 'Mnist'):
#         datasetName = 'mnist'
        
#     if (not args):
#             # Raise an error, args cannot be empty.
#             print ('Args parameter cannot be empty!!')
#     else:
# #             if (args.dataSplitStrategy == 1):
                
# #                 dataFileName = 'data_{}_IID_{}_U_{}_K_{}_R_{}_DS_{}.pkl'.format(datasetName, 'False', args.numusers, str(4), args.round_num, args.dataSplitStrategy)
# #             else:
#             # Replace the number 5 in the following line with the following: args.numusers
#             dataFileName = 'data_{}_IID:{}_U:{}_K:{}_R:{}_DS:{}.pkl'.format(datasetName, 'False', 20 , str(4), args.round_num, args.dataSplitStrategy)
                
#             directoryPath = './Datasplits_MNIST/Datasplit-{}/'.format(args.dataSplitStrategy)
#             print ('Datafilename : %s.' % (dataFileName))
          
#     mnist_data_image = []
#     mnist_data_label = []

#     # Load the information about the train, val and test sets' information of users.
#     u = pickle.load(open(directoryPath + dataFileName, 'rb'))

#     if (args.dataSplitStrategy == 1):
#         # Load the actual data.
#         f = open(directoryPath + 'DS-1_MNIST.pkl','rb')
#         trainset = pickle.load(f)
        
#         mnist_data_image = trainset.data
#         mnist_data_label = trainset.targets
        
#     elif ((args.dataSplitStrategy == 2) or (args.dataSplitStrategy == 3)):
#         data = u['data']
        
#         test_data = u['test_data']
        
#         mnist_data_image = np.array(data['x'])
#         mnist_data_label = np.array(data['y'])
        
#         mnist_testData_image = np.array(test_data['x'])
#         mnist_testData_label = np.array(test_data['y'])
        
#     NUM_USERS = len(u['train_data_users'])
    
#     # The following parameter is really inconsequential, this parameter is not required.
#     # NUM_LABELS = 3 
    
#     train_data_users = u['train_data_users']
#     test_data_users = u['test_data_users']
#     valid_data_users = u['val_data_users']
    
#     random.seed(1)
#     np.random.seed(1)
    
#     mnist_data_image = np.array(mnist_data_image)
#     mnist_data_label = np.array(mnist_data_label)
    
#     mnist_testData_image = np.array(mnist_testData_image)
#     mnist_testData_label = np.array(mnist_testData_label)
    
#     print('0')

#     X_train = [[] for _ in range(NUM_USERS)]
#     y_train = [[] for _ in range(NUM_USERS)]
#     X_test = [[] for _ in range(NUM_USERS)]
#     y_test = [[] for _ in range(NUM_USERS)]
#     #X_valid = [[] for _ in range(NUM_USERS)]
#     #y_valid = [[] for _ in range(NUM_USERS)]
    
#     length = 1 
#     for user in range(NUM_USERS):
#         for j in train_data_users[user]:  # 3 labels for each users
#             #print(cifa_data_image[j], len(cifa_data_image[j]))
#             X_train[user] += mnist_data_image[j].tolist()
#             y_train[user] += (mnist_data_label[j] * np.ones(length)).tolist()
#         for j in test_data_users[user]:
#             X_test[user] += mnist_testData_image[j].tolist()
#             y_test[user] += (mnist_testData_label[j] * np.ones(length)).tolist()
#     print("1")
    
#     #print(len(X_train[0]), len(cifa_data_image), len(cifa_data_image[j]))
#     # Create data structure
#     train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
#     test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

#     # Setup 5 users
#     # for i in trange(5, ncols=120):
#     for i in range(NUM_USERS):
#         uname = 'f_{0:05d}'.format(i)
        
#         train_len = len(X_train)
#         test_len = len(X_test)
        
#         train_data['users'].append(uname) 
#         train_data['user_data'][uname] = {'x': X_train[i], 'y': y_train[i]}
#         train_data['num_samples'].append(train_len)
#         test_data['users'].append(uname)
#         test_data['user_data'][uname] = {'x': X_test[i], 'y': y_test[i]}
#         test_data['num_samples'].append(test_len)
#     print("2")
    
#     print ("Inside the read_mnist_data() function : ")
#     from collections import Counter
#     print ("Train set stats.")
#     for i in range(NUM_USERS):
#         uname = 'f_{0:05d}'.format(i)
#         print ('%s : %s' % (uname, Counter(train_data['user_data'][uname]['y'])) )
        
#     print ("Test set stats.")
#     for i in range(NUM_USERS):
#         uname = 'f_{0:05d}'.format(i)
#         print ('%s : %s' % (uname, Counter(test_data['user_data'][uname]['y'])) )
    
#     return train_data['users'], 1 , train_data['user_data'], test_data['user_data']

def read_mnist_data(args=None):
    # Extract the required variables.
    # Sample file name format: data_mnist_IID_False_U_10_K_4_R_0_DS_1 
    
    if (args.dataset == 'Mnist'):
        datasetName = 'mnist'
        
    if (not args):
            # Raise an error, args cannot be empty.
            print ('Args parameter cannot be empty!!')
    else:
#             if (args.dataSplitStrategy == 1):
                
#                 dataFileName = 'data_{}_IID_{}_U_{}_K_{}_R_{}_DS_{}.pkl'.format(datasetName, 'False', args.numusers, str(4), args.round_num, args.dataSplitStrategy)
#             else:
            # Replace the number 5 in the following line with the following: args.numusers
            dataFileName = 'data_{}_IID:{}_U:{}_K:{}_R:{}_DS:{}.pkl'.format(datasetName, 'False', args.numusers , str(4), args.round_num, args.dataSplitStrategy)
                
            directoryPath = './Datasplits_MNIST/Datasplit-{}/'.format(args.dataSplitStrategy)
            print ('Datafilename : %s.' % (dataFileName))
          
    mnist_data_image = []
    mnist_data_label = []

    # Load the information about the train, val and test sets' information of users.
    u = pickle.load(open(directoryPath + dataFileName, 'rb'))

    if (args.dataSplitStrategy == 1):
        # Load the actual data.
        f = open(directoryPath + 'DS-1_MNIST.pkl','rb')
        trainset = pickle.load(f)
        
        mnist_data_image = trainset.data
        mnist_data_label = trainset.targets
        
    elif ((args.dataSplitStrategy == 2) or (args.dataSplitStrategy == 3)):
        data = u['data']
        mnist_data_image = np.array(data['x'])
        mnist_data_label = np.array(data['y'])
        
    NUM_USERS = len(u['train_data_users'])
    
    # The following parameter is really inconsequential, this parameter is not required.
    # NUM_LABELS = 3 
    
    train_data_users = u['train_data_users']
    test_data_users = u['test_data_users']
    valid_data_users = u['val_data_users']
    
    random.seed(1)
    np.random.seed(1)
    
    mnist_data_image = np.array(mnist_data_image)
    mnist_data_label = np.array(mnist_data_label)
    print('0')

    X_train = [[] for _ in range(NUM_USERS)]
    y_train = [[] for _ in range(NUM_USERS)]
    X_test = [[] for _ in range(NUM_USERS)]
    y_test = [[] for _ in range(NUM_USERS)]
    #X_valid = [[] for _ in range(NUM_USERS)]
    #y_valid = [[] for _ in range(NUM_USERS)]
    
    length = 1 
    for user in range(NUM_USERS):
        for j in train_data_users[user]:  # 3 labels for each users
            #print(cifa_data_image[j], len(cifa_data_image[j]))
            X_train[user] += mnist_data_image[j].tolist()
            y_train[user] += (mnist_data_label[j] * np.ones(length)).tolist()
        for j in test_data_users[user]:
            X_test[user] += mnist_data_image[j].tolist()
            y_test[user] += (mnist_data_label[j] * np.ones(length)).tolist()
    print("1")
    
    #print(len(X_train[0]), len(cifa_data_image), len(cifa_data_image[j]))
    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    # Setup 5 users
    # for i in trange(5, ncols=120):
    for i in range(NUM_USERS):
        uname = 'f_{0:05d}'.format(i)
        
        train_len = len(X_train)
        test_len = len(X_test)
        
        train_data['users'].append(uname) 
        train_data['user_data'][uname] = {'x': X_train[i], 'y': y_train[i]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X_test[i], 'y': y_test[i]}
        test_data['num_samples'].append(test_len)
    print("2")
    
    print ("Inside the read_mnist_data() function : ")
    from collections import Counter
    for i in range(NUM_USERS):
        uname = 'f_{0:05d}'.format(i)
        print ( Counter(train_data['user_data'][uname]['y']) )
    
    return train_data['users'], 1 , train_data['user_data'], test_data['user_data']
    
    
def read_cifa_data(args=None):
    
    # Extract the required variables.
    # Sample file name format: data_cifar_IID_False_U_10_K_4_R_0_DS_1 
    if (args.dataset == 'Cifar10'):
        datasetName = 'cifar'

    if (not args):
            # Raise an error, args cannot be empty.
            print ('Args parameter cannot be empty!!')
    else:
            if (args.dataSplitStrategy == 1):
                
                dataFileName = 'data_{}_IID_{}_U_{}_K_{}_R_{}_DS_{}.pkl'.format(datasetName, 'False', args.numusers, str(4), args.round_num, args.dataSplitStrategy)
            else:
                dataFileName = 'data_{}_IID:{}_U:{}_K:{}_R:{}_DS:{}.pkl'.format(datasetName, 'False', args.numusers, str(4), args.round_num, args.dataSplitStrategy)
                
            directoryPath = './Datasplit-{}/'.format(args.dataSplitStrategy)
            print ('Datafilename : %s.' % (dataFileName))
          
    cifa_data_image = []
    cifa_data_label = []

    # Load the information about the train, val and test sets' information of users.
    u = pickle.load(open(directoryPath + dataFileName, 'rb'))

    if (args.dataSplitStrategy == 1):
        # Load the actual data.
        f = open(directoryPath + 'DS_1.pkl','rb')
        trainset = pickle.load(f)
        
        cifa_data_image = trainset.data
        cifa_data_label = trainset.targets
        
    elif ((args.dataSplitStrategy == 2) or (args.dataSplitStrategy == 3)):
        data = u['data']
        cifa_data_image = np.array(data['x'])
        cifa_data_label = np.array(data['y'])
        
    NUM_USERS = len(u['train_data_users'])
    NUM_LABELS = 3
    
    train_data_users = u['train_data_users']
    test_data_users = u['test_data_users']
    valid_data_users = u['val_data_users']
    
    random.seed(1)
    np.random.seed(1)
    
    cifa_data_image = np.array(cifa_data_image)
    cifa_data_label = np.array(cifa_data_label)
    print('0')
#     cifa_data = []
#     for i in trange(NUM_USERS):
#         idx = cifa_data_label==i
#         cifa_data.append(cifa_data_image[idx])

    X_train = [[] for _ in range(NUM_USERS)]
    y_train = [[] for _ in range(NUM_USERS)]
    X_test = [[] for _ in range(NUM_USERS)]
    y_test = [[] for _ in range(NUM_USERS)]
    #X_valid = [[] for _ in range(NUM_USERS)]
    #y_valid = [[] for _ in range(NUM_USERS)]
    
    length = 1 
    for user in range(NUM_USERS):
        for j in train_data_users[user]:  # 3 labels for each users
            #print(cifa_data_image[j], len(cifa_data_image[j]))
            X_train[user] += cifa_data_image[j].tolist()
            y_train[user] += (cifa_data_label[j] * np.ones(length)).tolist()
        for j in test_data_users[user]:
            X_test[user] += cifa_data_image[j].tolist()
            y_test[user] += (cifa_data_label[j] * np.ones(length)).tolist()
    print("1")
    #print(len(X_train[0]), len(cifa_data_image), len(cifa_data_image[j]))
    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    # Setup 5 users
    # for i in trange(5, ncols=120):
    for i in range(NUM_USERS):
        uname = 'f_{0:05d}'.format(i)
        
        train_len = len(X_train)
        test_len = len(X_test)
        
        train_data['users'].append(uname) 
        train_data['user_data'][uname] = {'x': X_train[i], 'y': y_train[i]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X_test[i], 'y': y_test[i]}
        test_data['num_samples'].append(test_len)
    print("2")
    
    return train_data['users'], 1 , train_data['user_data'], test_data['user_data']

def read_data(dataset, args=None):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''

    if(dataset == "Cifar10"):
        clients, groups, train_data, test_data = read_cifa_data(args=args)
        return clients, groups, train_data, test_data

    # We've already generated the datasplits, so we just have to load them off the pickle files that they're saved in.
    elif (dataset == "Mnist"):
        clients, groups, train_data, test_data = read_mnist_data(args=args)
        return clients, groups, train_data, test_data
        
    train_data_dir = os.path.join('data',dataset,'data', 'train')
    test_data_dir = os.path.join('data',dataset,'data', 'test')
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data

def read_user_data(index,data,dataset):
    id = data[0][index]
    train_data = data[2][id]
    test_data = data[3][id]
    X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
    if(dataset == "Mnist"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    elif(dataset == "Cifar10"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    else:
        X_train = torch.Tensor(X_train).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    
    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    return id, train_data, test_data

class Metrics(object):
    def __init__(self, clients, params):
        self.params = params
        num_rounds = params['num_rounds']
        self.bytes_written = {c.id: [0] * num_rounds for c in clients}
        self.client_computations = {c.id: [0] * num_rounds for c in clients}
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}
        self.accuracies = []
        self.train_accuracies = []

    def update(self, rnd, cid, stats):
        bytes_w, comp, bytes_r = stats
        self.bytes_written[cid][rnd] += bytes_w
        self.client_computations[cid][rnd] += comp
        self.bytes_read[cid][rnd] += bytes_r

    def write(self):
        metrics = {}
        metrics['dataset'] = self.params['dataset']
        metrics['num_rounds'] = self.params['num_rounds']
        metrics['eval_every'] = self.params['eval_every']
        metrics['learning_rate'] = self.params['learning_rate']
        metrics['mu'] = self.params['mu']
        metrics['num_epochs'] = self.params['num_epochs']
        metrics['batch_size'] = self.params['batch_size']
        metrics['accuracies'] = self.accuracies
        metrics['train_accuracies'] = self.train_accuracies
        metrics['client_computations'] = self.client_computations
        metrics['bytes_written'] = self.bytes_written
        metrics['bytes_read'] = self.bytes_read
        metrics_dir = os.path.join('out', self.params['dataset'], 'metrics_{}_{}_{}_{}_{}.json'.format(
            self.params['seed'], self.params['optimizer'], self.params['learning_rate'], self.params['num_epochs'], self.params['mu']))
        #os.mkdir(os.path.join('out', self.params['dataset']))
        if not os.path.exists('out'):
            os.mkdir('out')
        if not os.path.exists(os.path.join('out', self.params['dataset'])):
            os.mkdir(os.path.join('out', self.params['dataset']))
        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf)