clear; close all;

[filename, pathname] = uigetfile('*.csv', 'Enter path of interactions.csv: ');
eval(['cd ',pathname]);
interactions = importInteractions(filename);
targetusers = importUsers('D:\Users\Marco\Documents\GitHub\recsys-2016\csv\target_users.csv');
toDelete = interactions.interaction_type<3;
interactions(toDelete, :) = [];
[G, ID] = findgroups(interactions);

test2;