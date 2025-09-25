# train
epochs = 200
batch_size = 48
device = 'cuda:0' 
id_dim = 16

labels_path = "/TCdata/NameTime_Idx_BST.pkl"
k8_sta_path = "/opt/data/private/norm_data_npy/Full2015_2023_Dataets/newk8_norm_values_split.pkl"
p12hpr_pth = '/opt/data/private/Auxiliay_StatisData/2015_2023/pre12h_labels.pkl'

train_k8_path = '/TCdata/k89_4ch1/train/'
valid_k8_path = '/TCdata/k89_4ch1/valid/'
predict_k8_path = '/TCdata/k89_4ch1/test/'

model_output_dir = '/opt/data/private/model/'
predict_model = '/opt/data/private/model/epoch_200.pth'
save_fig_dir = '/opt/data/private/model/exp_img/'

num_workers = 4  
save_model_iter = 20 

data_format = 'npy'

