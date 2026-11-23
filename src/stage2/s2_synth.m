function T = s2_synth(root)
% Synthetic validation of the stage-2 filters against a known arousal state
% (docs/EXPERIMENTS.md Part 13; the design and pass criteria were written there before this ran).
%
% Each session copies one real participant's layout (bin count and block codes only), draws a true
% state, then pulses and sizes from the MPP model, and runs both filters.  Writes one row per
% session to data/stage2_synth_scores.csv and prints C1-C5.
%
%   addpath src/stage2; s2_synth

if nargin < 1, root = pwd; end

% Fixed in Part 13 before the run
deltas = [-0.5 0 0.5 1.0];
r1s = [0 0.5];
b0 = log(0.157 / (1 - 0.157));
r0 = -1.86;
vr = 0.5;
ve = 2e-4;
clip_sd = 0.2;

sel = readtable(fullfile(root, 'data', 'selection.csv'), 'TextType', 'string');
ids = sel.id(string(sel.selected) == "True");

rows = {};
cond = 0;
for r1 = r1s
    for delta = deltas
        cond = cond + 1;
        for i = 1:numel(ids)
            rng(1000 * cond + i);
            o = s2_observations(ids(i), 0.05, root);     % layout only: marker and K
            mk = o.marker;
            K = numel(mk);
            base = mk == -1;
            film = mk >= 100;

            d = zeros(1, K);
            clips = unique(mk(film));
            for c = clips(:)'
                d(mk == c) = delta + clip_sd * randn;
            end
            x = d + cumsum(sqrt(ve) * randn(1, K));
            n = double(rand(1, K) < 1 ./ (1 + exp(-(b0 + x))));
            r = zeros(1, K);
            r(n == 1) = r0 + r1 * x(n == 1) + sqrt(vr) * randn(1, sum(n));

            dtrue = mean(x(film)) - mean(x(base));
            mpp = s2_filter_mpp(n, r);
            bin = s2_filter_bin(n);
            [dm, cm, vm, wm] = score(mpp, x, film, base);
            [db, cb, vb, wb] = score(bin, x, film, base);
            if strcmp(mpp.stop, 'converged')
                chosen = 'mpp';  dc = dm;  cc = cm;  vc = vm;  wc = wm;  sc = mpp.stop;
            else
                chosen = 'bin';  dc = db;  cc = cb;  vc = vb;  wc = wb;  sc = bin.stop;
            end
            rows(end + 1, :) = {ids(i), delta, r1, sum(n), dtrue, ...
                dm, cm, vm, mpp.stop, mpp.param_r1, ...
                db, cb, vb, bin.stop, ...
                chosen, dc, cc, vc, sc, wm, wb, wc}; %#ok<AGROW>
        end
        fprintf('r1 %.1f  delta %+.1f  done\n', r1, delta);
    end
end

T = cell2table(rows, 'VariableNames', {'layout_id', 'delta', 'r1_true', 'pulses', 'dtrue', ...
    'mpp_dhat', 'mpp_corr', 'mpp_cover', 'mpp_stop', 'mpp_r1', ...
    'bin_dhat', 'bin_corr', 'bin_cover', 'bin_stop', ...
    'chosen', 'dhat', 'corr', 'cover', 'stop', 'mpp_cover_ref', 'bin_cover_ref', 'cover_ref'});
writetable(T, fullfile(root, 'data', 'stage2_synth_scores.csv'));

% ---- criteria (Part 13) ----------------------------------------------------------------------
fprintf('\n');
report(T, 'chosen', T.dhat, T.corr, T.cover, T.stop);
report(T, 'MPP only', T.mpp_dhat, T.mpp_corr, T.mpp_cover, T.mpp_stop);
report(T, 'binary only', T.bin_dhat, T.bin_corr, T.bin_cover, T.bin_stop);

c1r = corr(T.dhat, T.dtrue);  p = polyfit(T.dtrue, T.dhat, 1);
bias = arrayfun(@(dl) mean(T.dhat(T.delta == dl) - T.dtrue(T.delta == dl)), deltas);
cov_med = median(T.cover);
conv = mean(strcmp(T.stop, 'converged'));
c5 = mean(strcmp(T.mpp_stop(T.r1_true == 0.5), 'converged'));
fprintf('\nC1 r %.3f, slope %.3f            %s\n', c1r, p(1), pf(c1r >= 0.8 && p(1) >= 0.7 && p(1) <= 1.3));
fprintf('C2 bias by delta %s   %s\n', mat2str(round(bias, 3)), pf(all(abs(bias) <= 0.1)));
fprintf('C3 coverage median %.3f          %s\n', cov_med, pf(cov_med >= 0.85 && cov_med <= 0.99));
cov_ref = median(T.cover_ref);
fprintf('C3'' (rescored after the first run, Part 13) baseline-referenced coverage median %.3f  %s\n', ...
    cov_ref, pf(cov_ref >= 0.85 && cov_ref <= 0.99));
for dl = deltas
    s = T.delta == dl;
    fprintf('    delta %+.1f: C3'' median %.3f (MPP %.3f, binary %.3f)\n', dl, median(T.cover_ref(s)), ...
        median(T.mpp_cover_ref(s), 'omitnan'), median(T.bin_cover_ref(s)));
end
fprintf('C4 converged %.3f                %s\n', conv, pf(conv >= 0.95));
fprintf('C5 MPP converged at r1=0.5: %.3f  %s\n', c5, pf(c5 >= 0.8));
end

function [dhat, c, cover, cover_ref] = score(res, x, film, base)
dhat = mean(res.state(film)) - mean(res.state(base));
c = corr(res.state(:), x(:));
cover = mean(x >= res.state_lcl & x <= res.state_ucl);
% C3' (Part 13, rescored after the first run): both relative to their own baseline mean
s = res.state - mean(res.state(base));
xr = x - mean(x(base));
h = (res.state_ucl - res.state_lcl) / 2;
cover_ref = mean(abs(xr - s) <= h);
end

function report(T, name, dhat, c, cover, stop)
fprintf('%-12s', name);
for r1 = unique(T.r1_true)'
    s = T.r1_true == r1;
    fprintf('| r1=%.1f: r(dhat,dtrue) %.2f, bias %+.3f, corr(x) med %.2f, cover med %.2f, converged %.2f ', ...
        r1, corr(dhat(s), T.dtrue(s)), mean(dhat(s) - T.dtrue(s)), median(c(s)), median(cover(s)), ...
        mean(strcmp(stop(s), 'converged')));
end
fprintf('\n');
end

function s = pf(ok)
if ok, s = 'PASS'; else, s = 'FAIL'; end
end
