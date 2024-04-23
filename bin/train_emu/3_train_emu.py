'''

script for training neural emulator in jax 


'''
import os, sys
import numpy as np

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
from jax.nn import sigmoid
from jax.tree_util import tree_map

import optax
import functools
import optuna 

import torch
from torch.utils import data

import util as U


########################################################################################
# input 
########################################################################################
iwave = int(sys.argv[1])

########################################################################################
# load data
########################################################################################
_thetas = []
for i in range(10):
    _thetas.append(np.load(os.path.join(U.data_dir(), 'seds', 'modelb', 'train_sed.modelb.%i.thetas_sps.npz' % iwave))['arr_0'])
_thetas = np.concatenate(_thetas, axis=0)

# whiten thetas  
avg_thetas = np.mean(_thetas, axis=0)
std_thetas = np.std(_thetas, axis=0)
thetas = (_thetas - avg_thetas)/std_thetas
np.save(os.path.join(U.data_dir(), 'seds', 'modelb', 'train_sed.modelb.%i.avg_thetas_sps.npy' % iwave), avg_thetas)
np.save(os.path.join(U.data_dir(), 'seds', 'modelb', 'train_sed.modelb.%i.std_thetas_sps.npy' % iwave), std_thetas)


_x_pca = np.load(os.path.join(U.data_dir(), 'seds', 'modelb', 'train_sed.modelb.x_pca.w%i.npy' % iwave))

# whiten pcas 
avg_x_pca = np.mean(_x_pca, axis=0)
std_x_pca = np.std(_x_pca, axis=0)
x_pca = (_x_pca - avg_x_pca)/std_x_pca
np.save(os.path.join(U.data_dir(), 'seds', 'modelb', 'train_sed.modelb.%i.avg_x_pca.npy' % iwave), avg_x_pca)
np.save(os.path.join(U.data_dir(), 'seds', 'modelb', 'train_sed.modelb.%i.std_x_pca.npy' % iwave), std_x_pca)

# set up pytorch data loaders
def numpy_collate(batch):
    return tree_map(np.asarray, data.default_collate(batch))

class NumpyLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn)

class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))

N_train = int(0.9*x_pca.shape[0])
train_dataloader = NumpyLoader(data.TensorDataset(torch.tensor(thetas[:N_train]), torch.tensor(x_pca[:N_train])), batch_size=500)
valid_dataloader = NumpyLoader(data.TensorDataset(torch.tensor(thetas[N_train:]), torch.tensor(x_pca[N_train:])), batch_size=500)

########################################################################################
# set up MLP in jax
########################################################################################
def nonlin_act(x, beta, gamma):
    return (gamma + sigmoid(beta * x) * (1 - gamma)) * x

def init_mlp_params(layer_sizes, scale=1e-2):
    keys = random.split(random.PRNGKey(1), len(layer_sizes))

    params = []
    for i, key in zip(np.arange(len(layer_sizes)-2), keys):
        m, n = layer_sizes[i], layer_sizes[i+1]
        w_key, b_key, _a_key, _b_key = random.split(key, num=4)
        params.append([scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,)),
                      scale * random.normal(_a_key, (n,)), scale * random.normal(_b_key, (n,))])

    m, n = layer_sizes[-2], layer_sizes[-1]
    w_key, b_key, _a_key, _b_key = random.split(keys[-1], num=4)
    params.append([scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))])
    return params

@functools.partial(jax.vmap, in_axes=(None, 0))
def forward(params, inputs):
    activations = inputs
    for w, b, beta, gamma in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = nonlin_act(outputs, beta, gamma) #relu(outputs)#

    final_w, final_b = params[-1]
    return jnp.dot(final_w, activations) + final_b

def mse_loss(params, inputs, targets):
    preds = forward(params, inputs)
    return jnp.mean((preds - targets) ** 2)

@jit
def update(params, opt_state, inputs, targets):
    loss, grads = jax.value_and_grad(mse_loss)(params, inputs, targets)
    updates, opt_state = gradient_transform.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

##################################################################################
# OPTUNA
##################################################################################
# Optuna Parameters
n_trials    = 1000
study_name  = 'emu_pca.modelb.w%i' % iwave
n_jobs     = 1
if not os.path.isdir(os.path.join(U.data_dir(), 'emus', 'modelb', study_name)):
    os.system('mkdir %s' % os.path.join(U.data_dir(), 'emus', 'modelb', study_name))
storage    = 'sqlite:///%s/%s/%s.db' % (U.data_dir(), 'emus', 'modelb', study_name, study_name)
n_startup_trials = 20

n_layers_min, n_layers_max = 2, 10
n_hidden_min, n_hidden_max = 64, 1024 
decay_rate_min, decay_rate_max = 0., 1.

def Objective(trial):
    ''' bojective function for optuna 
    '''
    # Generate the model                                         
    n_layers = trial.suggest_int("n_layers", n_layers_min, n_layers_max)
    n_hidden = trial.suggest_int("n_hidden", n_hidden_min, n_hidden_max)
    decay_rate = trial.suggest_float("lr", decay_rate_min, decay_rate_max) 

    # train MLP 
    layer_sizes = [thetas.shape[1]] + [n_hidden for range(n_layers)] + [x_pca.shape[1]]
    learning_rate = 1e-3

    # Initialize the MLP, optimizer, and scheduler
    params = init_mlp_params(layer_sizes)
    optimizer = optax.adam(learning_rate)

    total_steps = n_epochs*len(train_dataloader)
    scheduler = optax.exponential_decay(init_value=learning_rate, 
                                        transition_steps=total_steps, 
                                        transition_begin=int(total_steps*0.25),
                                        decay_rate=decay_rate)

    # Combining gradient transforms using `optax.chain`.
    gradient_transform = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
        optax.scale_by_adam(),  # Use the updates from adam.
        optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
        # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
        optax.scale(-1.0)
    )

    # Initialize optimizer state
    opt_state = gradient_transform.init(params)

    # Training loop
    train_loss, valid_loss = [], []
    for epoch in range(100):
        epoch_loss = 0.0
        for x, y in train_dataloader:
            params, opt_state, loss = update(params, opt_state, x, y)
            epoch_loss += loss
        epoch_loss /= len(train_dataloader)
        train_loss.append(epoch_loss)

        _loss = 0
        for x, y in valid_dataloader:
            loss = mse_loss(params, x, y)
            _loss += loss
        valid_loss.append(_loss/len(valid_dataloader))

        # early stopping after 20 epochs
        if valid_loss[-1] < best_valid_loss:
            best_valid_loss = valid_loss[-1]
            best_epoch = epoch
            best_params = params.copy() 

        if epoch > best_epoch + 20:
            print(f"Epoch {epoch}, Loss: {epoch_loss}, Valid Loss: {valid_loss[-1]}")
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss}, Valid Loss: {valid_loss[-1]}")

    # save trained parameters 
    with open(os.path.join(U.data_dir(), 'emus', 'modelb', '%s.%i.pkl' % (study_name, trial)), 'wb') as f:
        pickle.dump(best_params, f)
    return best_valid_loss

sampler     = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials) 
study       = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage, directions=["minimize"], load_if_exists=True) 

study.optimize(Objective, n_trials=n_trials, n_jobs=n_jobs)
print("  Number of finished trials: %i" % len(study.trials))
