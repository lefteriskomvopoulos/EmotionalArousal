function T = s2_subset(root, name, drop, src)
% A state table without some participants, for a variant of s2_tests / s2_slider_eval
% (docs/EXPERIMENTS.md Part 20).  Reads data/stage2_states_005.csv, or with src given
% data/stage2_states_005_<src>.csv, and writes data/stage2_states_005_<name>.csv; nothing else changes.
%
%   addpath src/stage2; s2_subset(pwd, 'mpp44', [15 106 111 115 168 178])                      % Part 20
%   addpath src/stage2; s2_subset(pwd, 'tonic14c_mpp44', [15 106 111 115 168 178], 'tonic14c')  % Part 20

if nargin < 1 || isempty(root), root = pwd; end
f = 'stage2_states_005.csv';
if nargin >= 4 && ~isempty(src), f = ['stage2_states_005_' src '.csv']; end
T = readtable(fullfile(root, 'data', f), 'TextType', 'string');
missing = setdiff(drop, unique(T.id));
assert(isempty(missing), 'not in the table: %s', mat2str(missing));
T = T(~ismember(T.id, drop), :);
writetable(T, fullfile(root, 'data', ['stage2_states_005_' name '.csv']));
fprintf('%s (from %s): %d participants, %d rows (dropped %s)\n', name, f, numel(unique(T.id)), height(T), mat2str(drop));
end
