import numpy as np
import scipy.sparse as sc
import pandas as pd
from _cython._mf import FunkSVD_sgd
import logging

from evaluate import evaluate

INPUT_INTERACTIONS = "data_modified/interactions_train.csv"
TARGET_USERS_INPUT = "data_modified/target_users.csv"
OUTPUT = "data_modified/result.csv"

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")


def load_active_during_test():
    items_data = pd.read_csv("data_modified/item_profile.csv",
                             delim_whitespace=True,
                             dtype={'id': int, 'active_during_test': str},
                             usecols=['id', 'active_during_test'])

    return set(items_data[items_data.active_during_test == '1']['id'].values)


def load_target_users():
    return pd.read_csv(TARGET_USERS_INPUT,
                       delim_whitespace=True,
                       dtype={'user_id': int})


def get_items_mapping():
    items_data = pd.read_csv("data_modified/item_profile.csv",
                             delim_whitespace=True,
                             dtype={'id': int},
                             usecols=['id'])
    item_ids = set()
    for item_id in items_data['id']:
        item_ids.add(item_id)

    map = {}
    for i, item_id in enumerate(sorted(item_ids)):
        map[item_id] = i

    return map


def get_users_mapping():
    users_data = pd.read_csv("data_modified/user_profile.csv",
                             delim_whitespace=True,
                             dtype={'user_id': int},
                             usecols=['user_id'])
    users_ids = set()
    for item_id in users_data['user_id']:
        users_ids.add(item_id)

    map = {}
    for i, user_id in enumerate(sorted(users_ids)):
        map[user_id] = i

    return map


def load_interactions_map():
    print("brb reading data...")

    interactions_data = pd.read_csv(INPUT_INTERACTIONS,  # interactions.csv for production
                                    delim_whitespace=True,
                                    dtype={'user_id': int, 'item_id': int, 'interaction_type': int},
                                    usecols=['user_id', 'item_id', 'interaction_type'])

    interactions_map = {}
    for _, row in interactions_data.iterrows():
        read_val = row['interaction_type']
        user_id = row['user_id']
        item_id = row['item_id']
        read_val = {1: 1, 2: 1.2, 3: 1.4}.get(read_val)
        current_val = interactions_map.get((user_id, item_id), 0)
        new_val = max(current_val, read_val)
        interactions_map[(user_id, item_id)] = new_val

    return interactions_map


def load_urm(interactions_map, user_id_to_row, item_id_to_col):
    data = [t[1] for t in interactions_map.items()]
    coords = [t[0] for t in interactions_map.items()]
    rows = [user_id_to_row[t[0]] for t in coords]
    cols = [item_id_to_col[t[1]] for t in coords]

    print("to urm...")
    # shape=(40000, 167956) i think
    return sc.csr_matrix((data, (rows, cols)), dtype=np.float32)


active_items = load_active_during_test()

interactions_map = load_interactions_map()

user_id_to_row = get_users_mapping()
item_id_to_col = get_items_mapping()

user_row_to_id = {v: k for k, v in user_id_to_row.items()}
item_row_to_id = {v: k for k, v in item_id_to_col.items()}

urm = load_urm(interactions_map, user_id_to_row, item_id_to_col)


class FunkSVD():
    '''
    FunkSVD model
    Reference: http://sifter.org/~simon/journal/20061211.html
    Factorizes the rating matrix R into the dot product of two matrices U and V of latent factors.
    U represent the user latent factors, V the item latent factors.
    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin} \limits_{U,V}\frac{1}{2}||R - UV^T||^2_2 + \frac{\lambda}{2}(||U||^2_F + ||V||^2_F)
    Latent factors are initialized from a Normal distribution with given mean and std.

    '''

    # TODO: add global effects
    def __init__(self,
                 num_factors=2000,
                 lrate=0.04,
                 reg=0.08,
                 iters=15,
                 init_mean=0.0,
                 init_std=0.02,
                 lrate_decay=1.0,
                 rnd_seed=42):
        '''
        Initialize the model
        :param num_factors: number of latent factors
        :param lrate: initial learning rate used in SGD
        :param reg: regularization term
        :param iters: number of iterations in training the model with SGD
        :param init_mean: mean used to initialize the latent factors
        :param init_std: standard deviation used to initialize the latent factors
        :param lrate_decay: learning rate decay
        :param rnd_seed: random seed
        '''
        super(FunkSVD, self).__init__()
        self.num_factors = num_factors
        self.lrate = lrate
        self.reg = reg
        self.iters = iters
        self.init_mean = init_mean
        self.init_std = init_std
        self.lrate_decay = lrate_decay
        self.rnd_seed = rnd_seed

    def __str__(self):
        return "FunkSVD(num_factors={}, lrate={}, reg={}, iters={}, init_mean={}, " \
               "init_std={}, lrate_decay={}, rnd_seed={})".format(
            self.num_factors, self.lrate, self.reg, self.iters, self.init_mean, self.init_std, self.lrate_decay,
            self.rnd_seed
        )

    def fit(self, X):
        X = X.tocsr()
        self.dataset = X

        self.U, self.V = FunkSVD_sgd(X, self.num_factors, self.lrate, self.reg, self.iters, self.init_mean,
                                     self.init_std,
                                     self.lrate_decay, self.rnd_seed)

    def recommend(self, user_row, n=None):
        scores = np.dot(self.U[user_row], self.V.T)
        ranking = scores.argsort()[::-1]

        ranking = [item_row_to_id[r] for r in ranking if
                   (user_row_to_id[user_row], item_row_to_id[r]) not in interactions_map and item_row_to_id[
                       r] in active_items]

        return ranking[:n]


print("brb fitting...")

for iters in np.arange(start=70, stop=5000, step=10):
    svd = FunkSVD(iters=iters)
    svd.fit(urm)
    i = 0
    with open(OUTPUT, 'w') as out:
        for target_user in load_target_users()['user_id']:
            i += 1
            if i % 1000 == 0:
                print(i)
            target_user_row = user_id_to_row[target_user]
            best = svd.recommend(target_user_row, n=5)
            out.write(str(target_user) + "," + " ".join(str(i) for i in best) + "\n")

    print("evaluate for lrate: " + str(iters))
    evaluate()
