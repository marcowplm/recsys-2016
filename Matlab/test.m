clear; close all;

importInteractions;
importUsers;
toDelete = interactions.interaction_type<3;
interactions(toDelete, :) = [];
[G, ID] = findgroups(interactions);

test2;