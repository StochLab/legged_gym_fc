import pickle

file = open('../../runs/gait-conditioned-agility/pretrain-v0/train/025417.456545/parameters2.pkl','rb')
config = pickle.load(file)
file.close()
print(config['Cfg']['control']['stiffness'])
# config['Cfg']['sim']['dt']=0.0075
# file = open('../../runs/gait-conditioned-agility/pretrain-v0/train/025417.456545/parameters3.pkl','wb')
# pickle.dump(config,file)
# file.close()
