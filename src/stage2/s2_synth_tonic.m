function T = s2_synth_tonic(root, variant, seed_offset)
% Synthetic check of variant tonic14 (docs/EXPERIMENTS.md Part 15; design and criteria written
% there before this ran), or, with variant = 'tonic14b', of the balanced floor (Part 16), or,
% with 'tonic14d', of the differenced tonic level (Part 17), or, with 'tonic14c', of the link
% calibrated from the pulse-only state (Part 18).
%
% seed_offset (default 0) is added to every session's seed.  Part 18 decides on fresh sessions,
% seed_offset = 900000 (-> data/stage2_synth_tonic14c_scores.csv); seed_offset = 0 gives the
% old sessions (-> data/stage2_synth_tonic14c_oldseeds_scores.csv, comparison only).
%
% Part 13's sessions at r1 = 0.5 (same seeds, so the same pulses), plus a true tonic level:
%   A   informative: s = 5 + x + N(0, 0.05^2)
%   B   drift:       s = the layout participant's own real tonic level + x  (its course is drift
%                    unrelated to the true state)
%   Ai  integrator (Part 17): s = 5 + gamma*cumsum(x) + N(0, 0.05^2), gamma = 0.01 uS/s/logit
%   Bi  drift (Part 17):      s = the real tonic level + gamma*cumsum(x)
% Parts 15-16 use A and B; Part 17 uses A, Ai and Bi.  The filter's input comes from
% s2_tonic_input.  Each session also goes through the main filters (no tonic) as the reference.
% Writes data/stage2_synth_tonic_scores.csv (tonic14) or data/stage2_synth_<variant>_scores.csv,
% prints T1-T4.
%
%   addpath src/stage2; s2_synth_tonic                     % tonic14, Part 15
%   addpath src/stage2; s2_synth_tonic(pwd, 'tonic14b')    % Part 16
%   addpath src/stage2; s2_synth_tonic(pwd, 'tonic14d')    % Part 17
%   addpath src/stage2; s2_synth_tonic(pwd, 'tonic14c', 900000)    % Part 18, fresh seeds
%   addpath src/stage2; s2_synth_tonic(pwd, 'tonic14c', 0)         % Part 18, old seeds

if nargin < 1, root = pwd; end
if nargin < 2 || isempty(variant), variant = 'tonic'; end
if nargin < 3 || isempty(seed_offset), seed_offset = 0; end
if strcmp(variant, 'tonic')
    out = fullfile(root, 'data', 'stage2_synth_tonic_scores.csv');
elseif strcmp(variant, 'tonic14c') && seed_offset == 0
    out = fullfile(root, 'data', 'stage2_synth_tonic14c_oldseeds_scores.csv');
elseif seed_offset == 0 || strcmp(variant, 'tonic14c')
    out = fullfile(root, 'data', ['stage2_synth_' variant '_scores.csv']);
else
    out = fullfile(root, 'data', sprintf('stage2_synth_%s_seeds%d_scores.csv', variant, seed_offset));
end
if any(strcmp(variant, {'tonic14d', 'tonic14c'})), scens = ["A" "Ai" "Bi"]; else, scens = ["A" "B"]; end
gamma = 0.01;                          % Part 17: uS per second per logit

deltas = [-0.5 0 0.5 1.0];
b0 = log(0.157 / (1 - 0.157));
r0 = -1.86;  vr = 0.5;  ve = 2e-4;  clip_sd = 0.2;  r1 = 0.5;

sel = readtable(fullfile(root, 'data', 'selection.csv'), 'TextType', 'string');
ids = sel.id(string(sel.selected) == "True");

rows = {};
for scen = scens
    for di = 1:numel(deltas)
        delta = deltas(di);
        cond = 4 + di;                     % s2_synth's condition number for r1 = 0.5
        for i = 1:numel(ids)
            rng(seed_offset + 1000 * cond + i);
            o = s2_observations(ids(i), 0.05, root);
            mk = o.marker;  K = numel(mk);
            base = mk == -1;  film = mk >= 100;
            d = zeros(1, K);
            clips = unique(mk(film));
            for c = clips(:)'
                d(mk == c) = delta + clip_sd * randn;
            end
            x = d + cumsum(sqrt(ve) * randn(1, K));
            n = double(rand(1, K) < 1 ./ (1 + exp(-(b0 + x))));
            r = zeros(1, K);
            r(n == 1) = r0 + r1 * x(n == 1) + sqrt(vr) * randn(1, sum(n));
            % (identical to s2_synth up to here; the tonic is drawn after)
            switch scen
                case "A",  s_true = 5 + x + 0.05 * randn(1, K);
                case "B",  s_true = fillmissing(o.tonic, 'nearest') + x;
                case "Ai", s_true = 5 + gamma * cumsum(x) + 0.05 * randn(1, K);
                case "Bi", s_true = fillmissing(o.tonic, 'nearest') + gamma * cumsum(x);
            end
            [s, fopt] = s2_tonic_input(s_true, variant);

            dtrue = mean(x(film)) - mean(x(base));
            ref = choose_main(n, r);
            if isfield(fopt, 'calibrate') && fopt.calibrate      % Part 18
                fopt.fixed = s2_tonic_calibrate(s, ref.state, ref.var);
            end
            [tn, how] = choose_tonic(n, r, s, ref, fopt);
            [dr, ~, ~, cr] = score(ref, x, film, base);
            [dt, ct, ~, crt] = score(tn, x, film, base);
            agree = corr(tn.state(:), ref.state(:)) ^ 2;
            rows(end + 1, :) = {scen, ids(i), delta, sum(n), dtrue, how, tn.stop, tn.iterations, ...
                tn.param_s1, tn.param_vs, tn.floored, dt, ct, crt, dr, cr, agree}; %#ok<AGROW>
        end
        fprintf('scenario %s  delta %+.1f  done\n', scen, delta);
    end
end

T = cell2table(rows, 'VariableNames', {'scenario', 'layout_id', 'delta', 'pulses', 'dtrue', ...
    'chosen', 'stop', 'iterations', 's1', 'vs', 'floored', 'dhat', 'corr', 'cover_ref', 'main_dhat', ...
    'main_cover_ref', 'r2_vs_main'});
writetable(T, out);

fprintf('\n');
for scen = scens
    S = T(T.scenario == scen, :);
    p = polyfit(S.dtrue, S.dhat, 1);  c1 = corr(S.dhat, S.dtrue);
    pm = polyfit(S.dtrue, S.main_dhat, 1);  cm = corr(S.main_dhat, S.dtrue);
    bias = arrayfun(@(dl) mean(S.dhat(S.delta == dl) - S.dtrue(S.delta == dl)), deltas);
    biasm = arrayfun(@(dl) mean(S.main_dhat(S.delta == dl) - S.dtrue(S.delta == dl)), deltas);
    cov = median(S.cover_ref);
    conv = mean(S.chosen ~= "main");
    fprintf('Scenario %s\n', scen);
    fprintf('  T1 r %.3f, slope %.3f               %s   (main filter: r %.3f, slope %.3f)\n', c1, p(1), ...
        pf(c1 >= 0.8 && p(1) >= 0.7 && p(1) <= 1.3), cm, pm(1));
    fprintf('  T2 bias by delta %s   %s   (main: %s)\n', mat2str(round(bias, 3)), pf(all(abs(bias) <= 0.1)), ...
        mat2str(round(biasm, 3)));
    fprintf('  T3 C3'' coverage median %.3f        %s   (main: %.3f)\n', cov, pf(cov >= 0.85 && cov <= 0.99), ...
        median(S.main_cover_ref));
    fprintf('  T4 converged %.3f (MPP+tonic %d, binary+tonic %d, main kept %d)  %s\n', conv, ...
        sum(S.chosen == "mpp_tonic"), sum(S.chosen == "bin_tonic"), sum(S.chosen == "main"), pf(conv >= 0.95));
    fprintf('  state vs main state: R^2 median %.3f; s1 median %.2f; floor active %.3f\n', ...
        median(S.r2_vs_main), median(S.s1, 'omitnan'), mean(S.floored(S.chosen ~= "main")));
end
end

function ch = choose_main(n, r)
mpp = s2_filter_mpp(n, r);
if strcmp(mpp.stop, 'converged'), ch = mpp; else, ch = s2_filter_bin(n); end
end

function [ch, how] = choose_tonic(n, r, s, main, fopt)
% Part 15: MPP + tonic if it converges, else binary + tonic, else the main analysis's state
ch = s2_filter_tonic(n, r, s, fopt);
how = "mpp_tonic";
if ~strcmp(ch.stop, 'converged')
    bopt = fopt;  bopt.sizes = false;
    ch = s2_filter_tonic(n, r, s, bopt);
    how = "bin_tonic";
end
if ~strcmp(ch.stop, 'converged') || any(isnan(ch.state))
    stop = ch.stop;  it = ch.iterations;
    ch = main;  ch.stop = stop;  ch.iterations = it;
    ch.param_s1 = NaN;  ch.param_vs = NaN;  ch.floored = false;
    how = "main";
end
end

function [dhat, c, cover, cover_ref] = score(res, x, film, base)
% as s2_synth
dhat = mean(res.state(film)) - mean(res.state(base));
c = corr(res.state(:), x(:));
cover = mean(x >= res.state_lcl & x <= res.state_ucl);
s = res.state - mean(res.state(base));
xr = x - mean(x(base));
h = (res.state_ucl - res.state_lcl) / 2;
cover_ref = mean(abs(xr - s) <= h);
end

function s = pf(ok)
if ok, s = 'PASS'; else, s = 'FAIL'; end
end
