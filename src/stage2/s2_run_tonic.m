function P = s2_run_tonic(root, variant)
% Variant tonic14 on the 50 real participants (docs/EXPERIMENTS.md Part 15), 0.05 uS pulses only,
% or, with variant = 'tonic14b', the balanced floor (Part 16), or 'tonic14d', the differenced
% tonic level (Part 17), or 'tonic14c', its link calibrated from the main state (Part 18).
%
% The tonic level is read once per 14 s (bins 7, 21, ...); the input and the filter options come
% from s2_tonic_input.  MPP + tonic if it converges, else
% binary + tonic, else the participant keeps the main analysis's chosen state
% (data/stage2_states_005.csv, unchanged).
%
% Writes (<v> = tonic for tonic14, tonic14b for Part 16, tonic14d for Part 17, tonic14c for Part 18)
%   data/stage2_params_<v>.csv       one row per participant
%   data/stage2_states_005_<v>.csv   one row per participant and 1 s bin (same columns that
%                                    s2_tests reads, plus the tonic and the main state)
%
%   addpath src/stage2; s2_run_tonic                     % tonic14
%   addpath src/stage2; s2_run_tonic(pwd, 'tonic14b')    % Part 16
%   addpath src/stage2; s2_run_tonic(pwd, 'tonic14d')    % Part 17
%   addpath src/stage2; s2_run_tonic(pwd, 'tonic14c')    % Part 18

if nargin < 1, root = pwd; end
if nargin < 2 || isempty(variant), variant = 'tonic'; end
s2_tonic_input(zeros(1, 30), variant);   % stops here on an unknown variant
main = readtable(fullfile(root, 'data', 'stage2_states_005.csv'), 'TextType', 'string');
ids = unique(main.id, 'stable');

prows = {};
S = {};
for i = 1:numel(ids)
    o = s2_observations(ids(i), 0.05, root);
    mn = main(main.id == ids(i), :);
    K = numel(o.n);
    assert(height(mn) == K && all(mn.n(:)' == o.n), 'P%d: bins differ from the main run', ids(i));
    [s, fopt] = s2_tonic_input(o.tonic, variant);
    if isfield(fopt, 'calibrate') && fopt.calibrate     % Part 18: link from the main state
        vhat = ((mn.state_ucl - mn.state_lcl) / (2 * norminv(0.975))) .^ 2;
        fopt.fixed = s2_tonic_calibrate(s, mn.state, vhat);
    end

    a = s2_filter_tonic(o.n, o.r, s, fopt);
    ch = a;  chosen = "mpp_tonic";
    b = struct('stop', "", 'iterations', NaN);
    if ~strcmp(a.stop, 'converged')
        bopt = fopt;  bopt.sizes = false;
        b = s2_filter_tonic(o.n, o.r, s, bopt);
        ch = b;  chosen = "bin_tonic";
    end
    if ~strcmp(ch.stop, 'converged') || any(isnan(ch.state))
        chosen = "main";
        st = mn.state(:)';  lo = mn.state_lcl(:)';  hi = mn.state_ucl(:)';  pr = mn.prob(:)';
        s1 = NaN;  vs = NaN;  ve = NaN;  fl = false;  ir = NaN;
    else
        st = ch.state;  lo = ch.state_lcl;  hi = ch.state_ucl;  pr = ch.prob;
        s1 = ch.param_s1;  vs = ch.param_vs;  ve = ch.param_ve;  fl = ch.floored;  ir = ch.info_ratio;
    end
    r2 = corr(st(:), mn.state) ^ 2;
    prows(end + 1, :) = {ids(i), string(o.group), K, sum(o.n), sum(~isnan(s)), string(a.stop), a.iterations, ...
        a.param_r1, string(b.stop), b.iterations, chosen, ve, ch.param_s0, s1, vs, std(o.tonic, 'omitnan'), ...
        r2, fl, ir}; %#ok<AGROW>
    S{end + 1} = table(mn.id, mn.group, mn.t_s, mn.marker, mn.n, mn.height_uS, o.tonic(:), ~isnan(s(:)), ...
        repmat(chosen, K, 1), st(:), lo(:), hi(:), pr(:), mn.state, mn.bin_state, ...
        'VariableNames', {'id', 'group', 't_s', 'marker', 'n', 'height_uS', 'tonic_uS', 'tonic_observed', ...
        'chosen', 'state', 'state_lcl', 'state_ucl', 'prob', 'main_state', 'bin_state'}); %#ok<AGROW>
    fprintf('P%-4d MPP+tonic %-9s binary+tonic %-9s -> %-9s  s1 %+.2f  sd(noise) %.3f uS  floor %d  R^2 vs main %.2f\n', ...
        ids(i), a.stop, b.stop, chosen, s1, sqrt(vs), fl, r2);
end
writetable(vertcat(S{:}), fullfile(root, 'data', ['stage2_states_005_' variant '.csv']));
P = cell2table(prows, 'VariableNames', {'id', 'group', 'bins', 'pulses', 'tonic_obs', 'mpp_tonic_stop', ...
    'mpp_tonic_iter', 'mpp_tonic_r1', 'bin_tonic_stop', 'bin_tonic_iter', 'chosen', 've', 's0', 's1', 'vs', ...
    'tonic_sd', 'r2_vs_main', 'floored', 'info_ratio'});
writetable(P, fullfile(root, 'data', ['stage2_params_' variant '.csv']));
end
