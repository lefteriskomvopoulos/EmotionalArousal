function E = s2_slider_eval(root, varargin)
% The state against the affect slider (docs/EXPERIMENTS.md Part 19, B; set before it ran):
% the evaluation of Oliver et al. (2025), adapted, plus H6's secondary version (|a - a_rest|).
%
%   s2_slider_eval(root)                          main analysis's chosen state
%   s2_slider_eval(root, 'variant', 'tonic14c')   the tonic14c state (comparison only)
%   s2_slider_eval(root, 'selftest', 'state')     slider replaced by the state: far above chance
%   s2_slider_eval(root, 'selftest', 'random')    slider replaced by a random walk: about chance
%
% Reads data/stage2_states_005[_<variant>].csv and data/stage2_slider_1hz.csv (s2_slider).
% Writes (not in self-test) data/stage2_slider_eval[_<variant>].csv,
% data/stage2_slider_windows[_<variant>].csv, results/figures/stage2_slider_eval[_<variant>].png.

p = inputParser;
addParameter(p, 'variant', '');
addParameter(p, 'selftest', '');
addParameter(p, 'nperm', 2000);
parse(p, varargin{:});
variant = p.Results.variant;  selftest = p.Results.selftest;  nperm = p.Results.nperm;
if nargin < 1 || isempty(root), root = pwd; end
sfx = '';  if ~isempty(variant), sfx = ['_' variant]; end

X = readtable(fullfile(root, 'data', ['stage2_states_005' sfx '.csv']), 'TextType', 'string');
A = readtable(fullfile(root, 'data', 'stage2_slider_1hz.csv'), 'TextType', 'string');
ids = unique(X.id, 'stable');
wlen = 20;  lag = 5;  topq = 0.15;  minrange = 0.5;

% ---- per participant: windows ------------------------------------------------------------------
P = struct('id', {}, 'group', {}, 'win', {}, 'clipx', {}, 'clipI', {}, 'clipIr', {});
W = {};
rng(7);
for i = 1:numel(ids)
    x = X(X.id == ids(i), :);  a = A(A.id == ids(i), :);
    assert(height(x) == height(a) && max(abs(x.t_s - a.t_s)) < 1e-6, 'P%d: bins differ', ids(i));
    mk = x.marker;  st = x.state;
    sd = (x.state_ucl - x.state_lcl) / (2 * norminv(0.975));
    hai = normcdf((st - median(st)) ./ sd);
    film = mk >= 100;
    af = a.affect;
    if range(af(film)) < minrange, continue; end            % the slider barely moves
    arest = median(af(mk == -1));
    I = abs(af - arest);
    switch selftest
        case 'state',  I = st - min(st(film));                     % the state itself
        case 'random', I = abs(cumsum(0.05 * randn(height(x), 1)));  % unrelated
    end
    I = 100 * I / max(I(film));
    Ilag = nan(size(I));                                     % the slider read lag s later
    for k = 1:numel(I) - lag
        if mk(k + lag) == mk(k), Ilag(k) = I(k + lag); end
    end
    clips = unique(mk(film), 'stable');                      % viewing order
    win = [];  cx = nan(numel(clips), 1);  cI = cx;  cIr = cx;
    for c = 1:numel(clips)
        b = find(mk == clips(c));
        cx(c) = mean(st(b));  cI(c) = mean(I(b));  cIr(c) = mean(abs(af(b) - 5.5));
        for w = 1:floor(numel(b) / wlen)
            bb = b((w - 1) * wlen + (1:wlen));
            win(end + 1, :) = [clips(c) w mean(st(bb)) mean(hai(bb)) mean(I(bb)) mean(Ilag(bb), 'omitnan')]; %#ok<AGROW>
        end
    end
    P(end + 1) = struct('id', ids(i), 'group', x.group(1), 'win', win, 'clipx', cx, 'clipI', cI, 'clipIr', cIr); %#ok<AGROW>
    W{end + 1} = array2table([repmat(ids(i), size(win, 1), 1) win], 'VariableNames', ...
        {'id', 'clip', 'window', 'state', 'hai', 'slider', 'slider_lag5'}); %#ok<AGROW>
end
np = numel(P);
fprintf('%d of %d participants with a slider that moves >= %.1f across the films\n', np, numel(ids), minrange);

% ---- analyses -----------------------------------------------------------------------------------
rng(19);                                            % the circular-shift null (Part 19)
rows = {};
for L = [0 1]                                       % L = 1: the slider read 5 s later
    lagname = "no lag";  if L, lagname = "slider +5 s"; end
    col = 5 + L;
    S = arrayfun(@(q) q.win(:, col), P, 'UniformOutput', false);
    H = arrayfun(@(q) q.win(:, 4), P, 'UniformOutput', false);
    Xs = arrayfun(@(q) q.win(:, 3), P, 'UniformOutput', false);

    % analysis 1 (i): rank-matched
    [acc, sen, spe, cm] = hl_rank(H, S, topq);
    nul = zeros(nperm, 1);
    for r = 1:nperm, nul(r) = median(hl_rank(H, shift(S), topq)); end
    rows = addrow(rows, "1 high/low", "(i) rank-matched", lagname, acc, sen, spe, cm, nul, np);

    % analysis 1 (ii): leave-one-participant-out cutoff on balanced accuracy
    cut = lopo_cut(H, S, topq);
    [acc, sen, spe, cm] = hl_cut(H, S, topq, cut);
    nul = zeros(nperm, 1);
    for r = 1:nperm, [~, s1, s2] = hl_cut(H, shift(S), topq, cut); nul(r) = median((s1 + s2) / 2); end
    rows = addrow(rows, "1 high/low", "(ii) LOPO cutoff (balanced accuracy)", lagname, acc, sen, spe, cm, nul, np, ...
        median((sen + spe) / 2), median(cut));

    % analysis 1 (iii): the paper's in-sample per-person cutoff (optimistic; no null)
    [acc, sen, spe, cm, cin] = hl_insample(H, S, topq);
    rows = addrow(rows, "1 high/low", "(iii) in-sample cutoff (paper; optimistic)", lagname, acc, sen, spe, cm, [], np, ...
        NaN, median(cin));

    % analysis 2: direction of change between consecutive windows
    for lev = [2 3]
        [ok, typ] = direction(Xs, S, lev);
        nul = zeros(nperm, 1);
        for r = 1:nperm, nul(r) = mean(direction(Xs, shift(S), lev)); end
        nm = sprintf("%d levels", lev);
        rows = adddir(rows, nm, lagname, mean(ok), numel(ok), nul);
        if lev == 3
            rows = adddir(rows, "3 levels: low<->high", lagname, mean(ok(typ == 1)), sum(typ == 1), []);
            rows = adddir(rows, "3 levels: involving mid", lagname, mean(ok(typ == 2)), sum(typ == 2), []);
        end
    end
end
% analysis 2, clip level (no lag): the 5 clip-to-clip changes per person
CX = arrayfun(@(q) q.clipx, P, 'UniformOutput', false);
CI = arrayfun(@(q) q.clipI, P, 'UniformOutput', false);
ok = clipdir(CX, CI);
nul = zeros(nperm, 1);
for r = 1:nperm, nul(r) = mean(clipdir(CX, shift(CI))); end
rows = adddir(rows, "clip to clip", "no lag", mean(ok), numel(ok), nul);

% H6 secondary: Spearman across the 6 clips, intensity from the person's own rest
% (skipped in a self-test: |a - 5.5| would come from the real slider)
if ~isempty(selftest)
    E = cell2table(rows, 'VariableNames', {'analysis', 'version', 'lag', 'estimate', 'p_null', 'null_mean', ...
        'null_95', 'n', 'accuracy_median', 'sensitivity_median', 'specificity_median', 'TP', 'TN', 'FP', 'FN', ...
        'extra', 'cutoff_median'});
    disp(E(:, {'analysis', 'version', 'lag', 'estimate', 'p_null', 'null_mean', 'null_95', 'n'}));
    return;
end
rr = nan(np, 1);  r55 = rr;
for i = 1:np
    if numel(unique(P(i).clipI)) > 1, rr(i) = corr(P(i).clipx, P(i).clipI, 'Type', 'Spearman'); end
    if numel(unique(P(i).clipIr)) > 1, r55(i) = corr(P(i).clipx, P(i).clipIr, 'Type', 'Spearman'); end
end
rows = addh6(rows, "H6 secondary: |a - a_rest|", rr);
rows = addh6(rows, "H6 as Part 14 on these 41 (|a - 5.5|, 1 Hz table)", r55);

E = cell2table(rows, 'VariableNames', {'analysis', 'version', 'lag', 'estimate', 'p_null', 'null_mean', ...
    'null_95', 'n', 'accuracy_median', 'sensitivity_median', 'specificity_median', 'TP', 'TN', 'FP', 'FN', ...
    'extra', 'cutoff_median'});
disp(E(:, {'analysis', 'version', 'lag', 'estimate', 'p_null', 'null_mean', 'null_95', 'n', ...
    'accuracy_median', 'sensitivity_median', 'specificity_median', 'extra'}));
if ~isempty(selftest), return; end
writetable(E, fullfile(root, 'data', ['stage2_slider_eval' sfx '.csv']));
writetable(vertcat(W{:}), fullfile(root, 'data', ['stage2_slider_windows' sfx '.csv']));
plot_eval(E, fullfile(root, 'results', 'figures', ['stage2_slider_eval' sfx '.png']), variant);
end

% ---- analysis 1 ---------------------------------------------------------------------------------

function [acc, sen, spe, cm] = hl_rank(H, S, topq)
n = numel(H);  acc = zeros(n, 1);  sen = acc;  spe = acc;  cm = zeros(1, 4);
for i = 1:n
    k = ceil(topq * numel(H{i}));
    h = topk(H{i}, k);  s = topk(S{i}, k);
    [acc(i), sen(i), spe(i), c] = score(h, s);  cm = cm + c;
end
end

function [acc, sen, spe, cm] = hl_cut(H, S, topq, cut)
n = numel(H);  acc = zeros(n, 1);  sen = acc;  spe = acc;  cm = zeros(1, 4);
for i = 1:n
    h = topk(H{i}, ceil(topq * numel(H{i})));
    [acc(i), sen(i), spe(i), c] = score(h, S{i} >= cut(i));  cm = cm + c;
end
end

function cut = lopo_cut(H, S, topq)
n = numel(H);  cs = 0:100;  B = zeros(n, numel(cs));
for i = 1:n
    h = topk(H{i}, ceil(topq * numel(H{i})));
    for j = 1:numel(cs)
        [~, se, sp] = score(h, S{i} >= cs(j));  B(i, j) = (se + sp) / 2;
    end
end
cut = zeros(n, 1);
for i = 1:n
    m = median(B(setdiff(1:n, i), :), 1);
    [~, j] = max(m);  cut(i) = cs(j);                    % the lowest cutoff among ties
end
end

function [acc, sen, spe, cm, cin] = hl_insample(H, S, topq)
n = numel(H);  acc = zeros(n, 1);  sen = acc;  spe = acc;  cm = zeros(1, 4);  cin = acc;
for i = 1:n
    h = topk(H{i}, ceil(topq * numel(H{i})));
    best = -1;
    for c = 50:100
        s = S{i} >= c;
        if sum(s) < 5, continue; end
        a = mean(h == s);
        if a > best, best = a;  cin(i) = c; end
    end
    if best < 0, cin(i) = 50; end                       % fewer than 5 windows ever reach 50
    [acc(i), sen(i), spe(i), c] = score(h, S{i} >= cin(i));  cm = cm + c;
end
end

function t = topk(v, k)
t = false(size(v));
[~, o] = sort(v, 'descend');
o = o(~isnan(v(o)));
t(o(1:min(k, numel(o)))) = true;
end

function [acc, sen, spe, c] = score(h, s)
TP = sum(h & s);  TN = sum(~h & ~s);  FP = sum(~h & s);  FN = sum(h & ~s);
acc = (TP + TN) / numel(h);
sen = TP / max(TP + FN, 1);  spe = TN / max(TN + FP, 1);
c = [TP TN FP FN];
end

% ---- analysis 2 ---------------------------------------------------------------------------------

function [ok, typ] = direction(Xs, S, lev)
ok = [];  typ = [];
for i = 1:numel(Xs)
    s = S{i};  x = Xs{i};
    if lev == 2
        L = 1 + (s > median(s, 'omitnan'));
    else
        q = prctile(s, [20 80]);
        L = 2 * ones(size(s));  L(s < q(1)) = 1;  L(s > q(2)) = 3;
    end
    L(isnan(s)) = NaN;
    for k = 1:numel(s) - 1
        if isnan(L(k)) || isnan(L(k + 1)) || L(k) == L(k + 1), continue; end
        ok(end + 1, 1) = sign(x(k + 1) - x(k)) == sign(s(k + 1) - s(k)); %#ok<AGROW>
        typ(end + 1, 1) = 1 + (lev == 3 && (L(k) == 2 || L(k + 1) == 2)); %#ok<AGROW>
    end
end
end

function ok = clipdir(CX, CI)
ok = [];
for i = 1:numel(CX)
    dx = diff(CX{i});  dI = diff(CI{i});
    k = dI ~= 0;
    ok = [ok; sign(dx(k)) == sign(dI(k))]; %#ok<AGROW>
end
end

% ---- null and rows ------------------------------------------------------------------------------

function S = shift(S)
% each person's slider sequence circularly shifted by a random non-zero offset
for i = 1:numel(S)
    n = numel(S{i});
    S{i} = circshift(S{i}, randi(n - 1));
end
end

function rows = addrow(rows, an, ver, lagname, acc, sen, spe, cm, nul, np, extra, cut)
if nargin < 11, extra = NaN; end
if nargin < 12, cut = NaN; end
if isempty(nul)
    est = median(acc);  pn = NaN;  nm = NaN;  n95 = NaN;
elseif isnan(extra)
    est = median(acc);  pn = (1 + sum(nul >= est)) / (1 + numel(nul));  nm = mean(nul);  n95 = prctile(nul, 95);
else
    est = extra;  pn = (1 + sum(nul >= est)) / (1 + numel(nul));  nm = mean(nul);  n95 = prctile(nul, 95);
end
rows(end + 1, :) = {an, ver, lagname, est, pn, nm, n95, np, median(acc), median(sen), median(spe), ...
    cm(1), cm(2), cm(3), cm(4), extra, cut};
end

function rows = adddir(rows, ver, lagname, est, n, nul)
if isempty(nul), pn = NaN;  nm = NaN;  n95 = NaN;
else, pn = (1 + sum(nul >= est)) / (1 + numel(nul));  nm = mean(nul);  n95 = prctile(nul, 95); end
rows(end + 1, :) = {"2 direction", ver, lagname, est, pn, nm, n95, n, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN};
end

function rows = addh6(rows, ver, rr)
rr = rr(~isnan(rr));
z = atanh(max(min(rr, 0.995), -0.995));
n = numel(z);  t = mean(z) / (std(z) / sqrt(n));  pv = 2 * tcdf(-abs(t), n - 1);
ci = mean(z) + [-1 1] * tinv(0.975, n - 1) * std(z) / sqrt(n);
rows(end + 1, :) = {"H6", ver, "no lag", tanh(mean(z)), pv, NaN, NaN, n, NaN, NaN, NaN, NaN, NaN, NaN, NaN, ...
    tanh(ci(1)), tanh(ci(2))};
end

% ---- figure -------------------------------------------------------------------------------------

function plot_eval(E, out, variant)
f = figure('Visible', 'off', 'Position', [100 100 1100 420]);
t = tiledlayout(f, 1, 2, 'TileSpacing', 'compact');
nm = "main analysis's state";  if ~isempty(variant), nm = string(variant) + " state"; end
title(t, "State vs affect slider (Oliver et al. 2025 evaluation, adapted): " + nm);
sel = {E.analysis == "1 high/low" & ~contains(E.version, "(iii)"), E.analysis == "2 direction" & ~isnan(E.null_mean)};
yl = {'median accuracy (i) / balanced accuracy (ii)', 'share of transitions in the same direction'};
for p = 1:2
    ax = nexttile(t);  hold(ax, 'on');
    R = E(sel{p}, :);
    for j = 1:height(R)
        plot(ax, [j j], [R.null_mean(j) R.null_95(j)], '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 6);
        plot(ax, j, R.estimate(j), 'o', 'MarkerFaceColor', [0 0.45 0.85], 'MarkerEdgeColor', 'none', 'MarkerSize', 8);
        text(ax, j, R.estimate(j), sprintf('  p %.3f', R.p_null(j)), 'FontSize', 8);
    end
    set(ax, 'XTick', 1:height(R), 'XTickLabel', R.version + " | " + R.lag, 'XTickLabelRotation', 25);
    ylabel(ax, yl{p});  xlim(ax, [0.5 height(R) + 0.5]);  grid(ax, 'on');
    if p == 1, title(ax, 'Analysis 1: high/low agreement (grey: null mean to 95th pct)');
    else, title(ax, 'Analysis 2: direction of change (grey: null mean to 95th pct)'); end
end
exportgraphics(f, out, 'Resolution', 110);
close(f);
end
