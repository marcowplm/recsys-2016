

u_int_id = unique(interactions.user_id);
u_int_it = unique(interactions.item_id);
counts = hist(interactions.item_id, u_int_it);
y = max(counts);

new_mat = [u_int_it, counts'];

new_mat = sortrows(new_mat, 2);
new_mat = flipud(new_mat);

% the 20 most popular items sorted using interaction_type == 3
result = new_mat(1:20, :);
result = result(:, 1:1);

%targetusers.rec = zeros([size(targetusers) 1]);
%targetusers.rec = num2str(targetusers.rec);

 
ca = cell(height(targetusers), 1);
for i = 1:height(targetusers)
    x = result(randperm(length(result)),:);
    formatSpec = '%d %d %d %d %d';
    str = sprintf(formatSpec, x(1,:), x(2,:), x(3,:), x(4,:), x(5,:));
    ca(i, :) = {str};
end

targetusers.recommended_items = ca(:,:);
targ2 = targetusers(1:100, :);
writetable(targetusers);
    