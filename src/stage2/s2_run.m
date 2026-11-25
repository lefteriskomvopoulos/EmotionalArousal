function P = s2_run(root, set)
% Stage 2 on the 50 real participants (docs/EXPERIMENTS.md Part 13), or with set = 'extra47' on the
% other 47 who passed selection (Part 21): same filters and rule, outputs with the suffix _extra47.
%
% For each participant and each pulse threshold (0.05 uS main, 0.02 uS secondary; rule 8):
% 1 s bins (s2_observations), the MPP filter and the binary filter, and the state chosen by the
% rule fixed before any real state was computed: MPP when it converged, binary otherwise.
%
% Writes
%   data/stage2_params.csv            one row per participant and threshold: stops, parameters
%   data/stage2_states_005.csv        one row per participant and 1 s bin (main threshold)
%   data/stage2_states_002.csv        the same for the secondary threshold
%
%   addpath src/stage2; s2_run

if nargin < 1, root = pwd; end
if nargin < 2 || isempty(set), set = 'real50'; end

sel = readtable(fullfile(root, 'data', 'selection.csv'), 'TextType', 'string');
switch set
    case 'real50'
        ids = sel.id(string(sel.selected) == "True");  sfx = '';
    case 'extra47'
        ids = sel.id(string(sel.stage2_pass) == "True" & string(sel.selected) ~= "True");  sfx = '_extra47';
    otherwise
        error('unknown set %s', set);
end

prows = {};
for th = [0.05 0.02]
    S = {};
    for i = 1:numel(ids)
        o = s2_observations(ids(i), th, root);
        mpp = s2_filter_mpp(o.n, o.r);
        bin = s2_filter_bin(o.n);
        if strcmp(mpp.stop, 'converged'), ch = mpp; chosen = "mpp"; else, ch = bin; chosen = "bin"; end

        prows(end + 1, :) = {ids(i), string(o.group), th, numel(o.n), sum(o.n), o.merged, ...
            string(mpp.stop), mpp.iterations, mpp.param_b0, mpp.param_ve, mpp.param_r0, ...
            mpp.param_r1, mpp.param_vr, string(bin.stop), bin.iterations, bin.param_ve, chosen}; %#ok<AGROW>

        K = numel(o.n);
        S{end + 1} = table(repmat(ids(i), K, 1), repmat(string(o.group), K, 1), o.t(:), o.marker(:), ...
            o.n(:), o.height(:), mpp.state(:), mpp.var(:), bin.state(:), bin.var(:), ...
            repmat(chosen, K, 1), ch.state(:), ch.state_lcl(:), ch.state_ucl(:), ch.prob(:), ...
            'VariableNames', {'id', 'group', 't_s', 'marker', 'n', 'height_uS', 'mpp_state', ...
            'mpp_var', 'bin_state', 'bin_var', 'chosen', 'state', 'state_lcl', 'state_ucl', 'prob'}); %#ok<AGROW>
        fprintf('P%-4d %.2f uS  %4d pulses  MPP %-9s  binary %-9s  -> %s\n', ids(i), th, sum(o.n), ...
            mpp.stop, bin.stop, chosen);
    end
    writetable(vertcat(S{:}), fullfile(root, 'data', sprintf('stage2_states_%03d%s.csv', round(100 * th), sfx)));
end

P = cell2table(prows, 'VariableNames', {'id', 'group', 'min_height_uS', 'bins', 'pulses', 'merged', ...
    'mpp_stop', 'mpp_iter', 'b0', 'mpp_ve', 'mpp_r0', 'mpp_r1', 'mpp_vr', ...
    'bin_stop', 'bin_iter', 'bin_ve', 'chosen'});
writetable(P, fullfile(root, 'data', ['stage2_params' sfx '.csv']));
end
