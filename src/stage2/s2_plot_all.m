function s2_plot_all(root)
% s2_plot for all 50 participants, one page each, ordered by film group then id.
% Writes results/figures/stage2_states_real50.pdf (replaced if it exists).
%
%   addpath src/stage2; s2_plot_all

if nargin < 1, root = pwd; end
out = fullfile(root, 'results', 'figures', 'stage2_states_real50.pdf');
if isfile(out), delete(out); end
S = readtable(fullfile(root, 'data', 'stage2_states_005.csv'), 'TextType', 'string');
P = readtable(fullfile(root, 'data', 'stage2_params.csv'), 'TextType', 'string');
Q = P(P.min_height_uS == 0.05, :);
[~, order] = sortrows([double(categorical(Q.group, ["neutral" "negative" "positive" "mixed"])), Q.id]);
for i = order'
    s2_plot(Q.id(i), root, out, S, P);
    fprintf('P%d (%s, %s)\n', Q.id(i), Q.group(i), Q.chosen(i));
end
end
