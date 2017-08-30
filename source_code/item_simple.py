import numpy as np
import pandas as pd
import scipy.sparse as sc
from sklearn.metrics.pairwise import cosine_similarity

from evaluate import evaluate
from utils import save_sparse_csr

URM_SHAPE = (2984851, 2835387)
SHRINKER = 23
K = 300
RECOMMENDATIONS_COUNT = 5
OUTPUT = "data_modified/result.csv"
INPUT_INTERACTIONS = "data_modified/interactions_train.csv"
TARGET_USERS_INPUT = "data_modified/target_users.csv"

MOST_POPULAR = [1053452, 2778525, 1244196, 1386412, 657183, 2791339, 278589, 536047, 2002097, 722433, 784737, 1092821,
                1053542, 79531, 1928254, 1133414, 2532610, 1984327, 1162250, 1443706, 830073, 1140869, 343377, 1742926,
                1201171, 1233470, 1576126, 460717, 2593483, 1237071]


def apply_shrinkage(X, dist):
    X = X.copy()
    X.data = np.ones_like(X.data)
    co_counts = X.T.dot(X)
    # compute the shrinkage factor as co_counts_ij / (co_counts_ij + shrinkage)
    # then multiply dist with it
    co_counts.data = np.divide(co_counts.data, co_counts.data + SHRINKER)
    dist = dist.multiply(co_counts)

    return dist


def load_target_users():
    return pd.read_csv(TARGET_USERS_INPUT,
                       delim_whitespace=True,
                       dtype={'user_id': int})


def load_interactions():
    interactions_data = pd.read_csv(INPUT_INTERACTIONS,  # interactions.csv for production
                                    delim_whitespace=True,
                                    dtype={'user_id': int, 'item_id': int, 'interaction_type': int, 'created_at': int},
                                    usecols=['user_id', 'item_id', 'interaction_type', 'created_at'])

    # print(interactions_data['created_at'].describe())

    interactions_weights = {1: 1.0, 2: 1.1, 3: 1.2}
    interactions_map = dict()

    for interaction in interactions_data.itertuples():
        read_val = interactions_weights[interaction.interaction_type]
        current_val = interactions_map.get((interaction.user_id, interaction.item_id), 0)
        if read_val > current_val:
            interactions_map[(interaction.user_id, interaction.item_id)] = read_val

    return interactions_map


def load_active_during_test():
    items_data = pd.read_csv("data_modified/item_profile.csv",
                             delim_whitespace=True,
                             dtype={'id': int, 'active_during_test': str},
                             usecols=['id', 'active_during_test'])

    return set(items_data[items_data.active_during_test == '1']['id'].values)


def interactions_to_urm(interactions_map):
    data, rows, cols = [], [], []
    for (user_id, item_id), interaction_val in interactions_map.items():
        data.append(interaction_val)
        rows.append(user_id)
        cols.append(item_id)

    return sc.csr_matrix((data, (rows, cols)), shape=URM_SHAPE, dtype=np.float32)


def keep_top_k(ism):
    ism = ism.tolil()

    values, rows, cols = [], [], []
    nitems = URM_SHAPE[1]
    for i in range(nitems):
        if len(ism.rows[i]) == 0:
            continue
        row_nonzeros = np.asarray(ism.rows[i])
        row_data = np.asarray(ism.data[i])
        if K > len(row_data):
            ind = range(len(row_data))
        else:
            ind = np.argpartition(row_data, -K)[-K:]
        top_vals = row_data[ind]

        values.extend(top_vals)
        cols.extend(row_nonzeros[ind])
        rows.extend(np.ones(len(ind)) * i)

    return sc.csc_matrix((values, (rows, cols)), shape=(nitems, nitems), dtype=np.float32)


active_items = load_active_during_test()
# load data from csv
interactions_map = load_interactions()

# build user-rating matrix
urm = interactions_to_urm(interactions_map)

# compute item similarity matrix
ism = cosine_similarity(urm.transpose(), dense_output=False)
ism = apply_shrinkage(urm, ism)
print("ism shape")
print(ism.shape)

ism = keep_top_k(ism)

# compute estimated user-rating matrix
print("computing estimated ratings...")
estimated_urm = urm.dot(ism.T)

#save_sparse_csr('urm_item_based_full', estimated_urm.tocsr())

# write recommendations
print("writing recommendations...")
estimated_urm = estimated_urm.tolil()
i = 0
with open(OUTPUT, 'w') as out:
    out.write("user_id,recommended_items\n")
    for target_user in load_target_users()['user_id']:
        i += 1
        if i % 100 == 0:
            print(i)
        row_nonzeros = estimated_urm.rows[target_user]
        row_data = estimated_urm.data[target_user]

        indicies_with_data = list(zip(row_nonzeros, row_data))
        indicies_with_data.sort(key=lambda t: t[1], reverse=True)

        best_items = [t[0] for t in indicies_with_data] + MOST_POPULAR
        recommendations = [int(item_id) for item_id in best_items if
                           (target_user, item_id) not in interactions_map and item_id in active_items]

        out.write(str(target_user) + "," + " ".join(str(i) for i in recommendations[:RECOMMENDATIONS_COUNT]) + "\n")

evaluate()
