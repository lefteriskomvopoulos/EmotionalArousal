function T = s2_slider(root)
% The affect slider of the 50 participants on the stage-2 bins (docs/EXPERIMENTS.md Part 19).
%
% Reads data/raw/Study6/S6_P<id>_All.csv (POPANE Study 6, 1000 Hz; not in the repository) and
% data/stage2_states_005.csv (the bins: t_s is the bin centre, bins are [t_s - 0.5, t_s + 0.5)).
% For each bin: the mean slider value and whether the slider moved inside the bin.  Nothing here
% looks at the state.
%
% Writes data/stage2_slider_1hz.csv: id, group, t_s, marker, affect (1-10, valence), moved (0/1).
%
%   addpath src/stage2; s2_slider

if nargin < 1, root = pwd; end
main = readtable(fullfile(root, 'data', 'stage2_states_005.csv'), 'TextType', 'string');
ids = unique(main.id, 'stable');
S = cell(numel(ids), 1);
for i = 1:numel(ids)
    f = fullfile(root, 'data', 'raw', 'Study6', sprintf('S6_P%d_All.csv', ids(i)));
    R = s2_read_raw(f, {'timestamp', 'affect'});
    mn = main(main.id == ids(i), :);
    K = height(mn);
    t0 = mn.t_s(1) - 0.5;
    kb = floor(R.timestamp - t0 + 1e-6) + 1;
    ok = kb >= 1 & kb <= K & ~isnan(R.affect);
    a = accumarray(kb(ok), R.affect(ok), [K 1], @mean, NaN);
    mv = accumarray(kb(ok), R.affect(ok), [K 1], @(v) double(numel(v) > 1 && any(diff(v) ~= 0)), 0);
    S{i} = table(mn.id, mn.group, mn.t_s, mn.marker, a, mv, ...
        'VariableNames', {'id', 'group', 't_s', 'marker', 'affect', 'moved'});
    fprintf('P%-4d bins %d, slider missing in %d bins\n', ids(i), K, sum(isnan(a)));
end
T = vertcat(S{:});
writetable(T, fullfile(root, 'data', 'stage2_slider_1hz.csv'));
end
