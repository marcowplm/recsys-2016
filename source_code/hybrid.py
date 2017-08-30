import pandas as pd

from evaluate import evaluate
from utils import load_sparse_csr
from sklearn.preprocessing import normalize
import numpy as np

RECOMMENDATIONS_COUNT = 5
OUTPUT = "data_modified/result.csv"
INPUT_INTERACTIONS = "data_modified/interactions.csv"
TARGET_USERS_INPUT = "data_modified/target_users.csv"

MOST_POPULAR = [1053452, 2778525, 1244196, 1386412, 657183, 2791339, 278589, 536047, 2002097, 722433, 784737, 1092821,
                1053542, 79531, 1928254, 1133414, 2532610, 1984327, 1162250, 1443706, 830073, 1140869, 343377, 1742926,
                1201171, 1233470, 1576126, 460717, 2593483, 1237071]


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


active_items = load_active_during_test()
# load data from csv
interactions_map = load_interactions()

urm_user = load_sparse_csr('urm_user_based_full.npz')
urm_item = load_sparse_csr('urm_item_based_full.npz')
urm_funk = load_sparse_csr('urm_funk_full.npz')

urm_user = normalize(urm_user)
urm_item = normalize(urm_item)
urm_funk = normalize(urm_funk)

# 0.024365000000000008 0.12 0.24
alpha = 0.3
beta = 0.3

# for beta in np.arange(start=0.28, stop=10, step=0.01):
estimated_urm = urm_item * alpha + urm_user * beta + urm_funk * (1.0 - alpha - beta)

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

print("Eval for alpha: " + str(alpha) + ", beta: " + str(beta))
evaluate()
