from Environment import Environment 
from Depeweg_BNN import BayesianNeuralNetwork as Depeweg_BNN
from Killian_BNN import BayesianNeuralNetwork as Killian_BNN
import matplotlib.pyplot as plt
import numpy as np
import torch


relu = lambda x: torch.max(x, torch.tensor(0))
lin = lambda x:x
num_dims = 1
num_wb = 5
bnn_hidden_layer_size = 50
bnn_num_hidden_layers = 2
out_dim = 1
N = 1000

param_set_kil = {
    'bnn_layer_sizes': [num_dims+num_wb]+[bnn_hidden_layer_size]*bnn_num_hidden_layers+[out_dim],
    'weight_count': num_wb,
    'num_state_dims': num_dims,
    'bnn_num_samples': 50,
    'bnn_batch_size': (N//20),
    'num_strata_samples': 1,
    'bnn_training_epochs': 1,
    'bnn_v_prior': 1.0,
    'bnn_learning_rate': 0.005,
    'bnn_alpha': 0.5,
    'wb_num_epochs':1,
    'wb_learning_rate':0.01 
}


param_set_dep = {
                'bnn_layer_sizes': [num_dims+1]+[bnn_hidden_layer_size]*bnn_num_hidden_layers+[out_dim],
                'weight_count': 1,
                'num_state_dims': out_dim,
                'bnn_num_samples': 50,
                'bnn_batch_size': (N//10),
                'bnn_training_epochs': 1,
                'bnn_v_prior': 1, # 
                'bnn_learning_rate': 0.001, 
                'bnn_alpha': 0.5,
                'gamma': 0.00045,
                'lambda': -10,
                'theta': 1,
                'N_train': N
            }

env_param_1 = {'success_prob': 0.5,
             'magnitude_1': 10,
             'magnitude_2': 10,
             'noise':1}
env = Environment(N,env_param=env_param_1)
dep_bnn = Depeweg_BNN(param_set_dep, nonlinearity=relu)

general_buffer = []
a = np.array(np.random.normal(0,1,[N,1]))
for i in range(0,N,1):
    general_buffer.append( np.array([env.x[i],[], np.array(env.y[i]),np.array(0)],dtype=object))
general_buffer_np = np.stack(general_buffer)
inst_indices = np.zeros([general_buffer_np.shape[0],])
inst_indices = inst_indices.astype(int)
exp_dict = {}
for idx in range(1):
    exp_dict[idx] = general_buffer_np[inst_indices == idx]
X = np.array([np.hstack([general_buffer_np[tt,0],general_buffer_np[tt,1]]) for tt in range(general_buffer_np.shape[0])])
network = Killian_BNN(param_set=param_set_kil, nonlinearity=lin)
full_task_weights = np.random.normal(0,1,(1,num_wb))
iteration = 2000
plt.ion()
for it in range(iteration):
    dep_bnn.fit_network(env.x, env.y)
    output_dep = dep_bnn.feed_forward(env.x)
    env.plot_array()
    # Update BNN network weights
    network.fit_network(general_buffer_np, full_task_weights, 0, state_diffs=False,
                        use_all_exp=True)
    ## Latent variable optimization - this is used for transfering the inference between environments. Uncomment if you have multiple environments
    # full_task_weights = network.optimize_latent_weighting_stochastic(
    #             general_buffer_np,np.atleast_2d(full_task_weights),0,state_diffs=False,use_all_exp=True)
    # print(full_task_weights)
    print(f'finished BNN update {it}')
    ## Uncomment following two lines to prompt the error statistics
    # l2_errors = network.get_td_error(np.hstack((X,full_task_weights[inst_indices])), np.array(env.y).reshape(env.y.shape[0],1), location=0.0, scale=1.0, by_dim=False,p=False)
    # print ("After Latent update: iter: {}, Mean Error: {}, Std Error: {}".format(i,np.mean(l2_errors),np.std(l2_errors)))
    x_h = env.x
    output_kil = np.zeros([N,1])
    for i in range(len(x_h)):
        aug_reward = np.hstack([x_h[i], [], full_task_weights.reshape(full_task_weights.shape[1],)]).reshape((1,-1))
        output_kil[i] = (network.feed_forward(aug_reward).flatten())
    plt.plot(env.x, output_kil,".",color='g',label="Killian")
    plt.plot(env.x, output_dep.detach().numpy(),"+",color='r',label="Depeweg")
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title(f'$A_1$ : {env.A1}, $A_2$ : {env.A2}, $\gamma$: {env.alpha}, $\sigma^2 $ : {env.sigma},')
    plt.grid(True)
    plt.legend()
    plt.draw()
    plt.pause(0.0001)
    if it < iteration-1:
        plt.clf()
plt.ioff() 
plt.savefig("Figure.png", format='png')