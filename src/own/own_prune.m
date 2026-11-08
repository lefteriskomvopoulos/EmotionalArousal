function u = own_prune(u, sep)
% Keep only the largest impulse among those closer than sep samples.
idx = find(u);
[~, order] = sort(u(idx), 'descend');
keep = false(size(u));
for i = idx(order)'
    if ~any(keep(max(i - sep + 1, 1):min(i + sep - 1, numel(u))))
        keep(i) = true;
    end
end
u(~keep) = 0;
end
