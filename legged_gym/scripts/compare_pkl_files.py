import pickle as pkl

file1 = '/home/vamshi/Downloads/legged_gym-libtraj/logs/go1_basic7/Jan08_16-27-17_/config.pkl'
file2 = '/home/vamshi/Downloads/legged_gym-libtraj/legged_gym/scripts/vamshi/scratch/2024/01-08/224429/parameters.pkl'

data1 = pkl.load(open(file1, 'rb'))
data2 = pkl.load(open(file2, 'rb'))

keys1 = data1.keys()
keys2 = data2['Cfg'].keys()

for key in keys2:
    print(key, data2['Cfg'][key])
    print(key, data1[key])
