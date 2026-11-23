function R = s2_tests(root, varargin)
% The H1-H6 tests, exactly as set in docs/EXPERIMENTS.md Part 14 before they were run.
%
% Per participant and block (baseline, each film clip): S = mean chosen state, R = log((pulses +
% 0.5) / minutes), Z = mean log pulse height.  Change scores against the person's own baseline.
% S is primary; R and Z are reported next to every test.  Two-sided alpha 0.05; equivalence by
% TOST with margin 0.5 logit units (S only).  Rank tests next to every t-test.
%
% Checks (secondary, Part 14): main (0.05 uS, chosen state), th002 (0.02 uS), binary (binary
% filter's state), noP68 (P68 left out).
%
% Writes (real run only)
%   data/stage2_block_summary.csv     one row per participant, block and threshold
%   data/stage2_tests.csv             one row per check, hypothesis, part and measure
%   results/figures/stage2_hypotheses.png
%
%   addpath src/stage2; s2_tests                     % the real run (once)
%   addpath src/stage2; s2_tests(pwd, 'synthetic', true)   % code check: made-up states with a
%                                                    % known effect on the real layouts; writes nothing
%
% H6 needs the affect slider from the raw POPANE files (data/raw/Study6/S6_P<id>_All.csv); without
% them it is reported as not tested.

ip = inputParser;
addParameter(ip, 'synthetic', false);
addParameter(ip, 'delta', 0.5);
parse(ip, varargin{:});
if nargin < 1 || isempty(root), root = pwd; end
syn = ip.Results.synthetic;
delta = ip.Results.delta;

checks = struct('name', {'main', 'th002', 'binary', 'noP68'}, 'th', {0.05, 0.02, 0.05, 0.05}, ...
    'col', {'state', 'state', 'bin_state', 'state'}, 'drop', {[], [], [], 68});

cache = struct();
rows = {};
Bout = {};
for c = checks
    key = sprintf('t%03d', round(100 * c.th));
    if ~isfield(cache, key)
        S = readtable(fullfile(root, 'data', sprintf('stage2_states_%03d.csv', round(100 * c.th))), ...
            'TextType', 'string');
        if syn, S = synth_states(S); end
        cache.(key) = S;
    end
    S = cache.(key);
    S = S(~ismember(S.id, c.drop), :);
    [B, P] = summarise(S, c.col);
    if any(strcmp(c.name, {'main', 'th002'}))
        B.min_height_uS = repmat(c.th, height(B), 1);
        Bout{end + 1} = B; %#ok<AGROW>
    end
    rows = [rows; run_tests(P, c.name, c.th, delta)]; %#ok<AGROW>
    if strcmp(c.name, 'main'), Bmain = B; Pmain = P; end
end
rows = [rows; h6(root, Bmain, syn)];

R = cell2table(rows, 'VariableNames', {'check', 'min_height_uS', 'hypothesis', 'part', 'measure', ...
    'test', 'n1', 'n2', 'estimate', 'ci_lo', 'ci_hi', 'ci90_lo', 'ci90_hi', 'p', 'p_rank', 'p_tost', ...
    'outcome', 'primary'});

print_summary(R, syn);
if ~syn
    B = vertcat(Bout{:});
    writetable(B, fullfile(root, 'data', 'stage2_block_summary.csv'));
    writetable(R, fullfile(root, 'data', 'stage2_tests.csv'));
    plot_changes(Pmain, fullfile(root, 'results', 'figures', 'stage2_hypotheses.png'));
end
end

% ---- per-block summaries and change scores -------------------------------------------------------

function [B, P] = summarise(S, col)
ids = unique(S.id, 'stable');
N = numel(ids);
emos = ["fear" "anger" "disgust" "sadness" "amusement" "tenderness"];
P.id = ids;  P.group = strings(N, 1);
P.film = nan(N, 3);  P.neg = nan(N, 3);  P.pos = nan(N, 3);  P.emo = nan(N, 6, 3);
brow = {};
for i = 1:N
    T = S(S.id == ids(i), :);
    x = T.(col);  n = T.n;  h = T.height_uS;  mk = T.marker;
    P.group(i) = T.group(1);
    base = mk == -1;
    b = measures(x, n, h, base);
    brow(end + 1, :) = {ids(i), T.group(1), -1, "baseline", "baseline", sum(base), sum(n(base)), b(1), b(2), b(3)}; %#ok<AGROW>
    clips = unique(mk(mk >= 100))';
    cs = nan(numel(clips), 3);
    ce = strings(numel(clips), 1);
    for j = 1:numel(clips)
        m = mk == clips(j);
        v = measures(x, n, h, m);
        cs(j, :) = v - b;
        [val, ce(j)] = code_info(clips(j));
        brow(end + 1, :) = {ids(i), T.group(1), clips(j), val, ce(j), sum(m), sum(n(m)), v(1), v(2), v(3)}; %#ok<AGROW>
    end
    P.film(i, :) = measures(x, n, h, mk >= 100) - b;
    if any(mk >= 200 & mk < 300), P.neg(i, :) = measures(x, n, h, mk >= 200 & mk < 300) - b; end
    if any(mk >= 300), P.pos(i, :) = measures(x, n, h, mk >= 300) - b; end
    for e = 1:6
        s = ce == emos(e);
        if any(s), P.emo(i, e, :) = reshape(mean(cs(s, :), 1, 'omitnan'), 1, 1, 3); end
    end
end
B = cell2table(brow, 'VariableNames', {'id', 'group', 'marker', 'valence', 'emotion', 'bins', ...
    'pulses', 'S', 'R', 'Z'});
end

function v = measures(x, n, h, m)
% S: mean state; R: log rate per minute with 0.5 added to the count; Z: mean log height
v = [mean(x(m)), log((sum(n(m)) + 0.5) / (sum(m) / 60)), mean(log(h(m & n == 1)))];
if ~any(m & n == 1), v(3) = NaN; end
end

function [val, emo] = code_info(c)
% POPANE stimulus list (docs/dataset_check/metadata_R1.xlsx)
if c < 200
    val = "neutral";  emo = "neutral";
elseif c < 300
    val = "negative";
    emo = ["fear" "anger" "anger" "anger" "disgust" "sadness"];  emo = emo(c - 200);
else
    val = "positive";
    emo = ["amusement" "amusement" "amusement" "amusement" "tenderness" "tenderness"];  emo = emo(c - 300);
end
end

% ---- the tests (Part 14 table) ------------------------------------------------------------------

function rows = run_tests(P, check, th, delta)
g = P.group;
mnames = ["S" "R" "Z"];
rows = {};
for m = 1:3
    mn = mnames(m);
    useq = m == 1;                       % TOST margin is in logit units: S only
    prim = strcmp(check, 'main') && m == 1;
    f = P.film(:, m);

    % H1: all 50, own films, one-sample
    r = onesample(f, delta, useq);
    rows(end + 1, :) = row(check, th, "H1", "all", mn, "one-sample t", r, dirout(r, 1), prim); %#ok<AGROW>
    for gr = ["neutral" "negative" "positive" "mixed"]
        r = onesample(f(g == gr), delta, false);
        rows(end + 1, :) = row(check, th, "H1", "group " + gr + " (descriptive)", mn, "one-sample t", r, "descriptive", false); %#ok<AGROW>
    end

    % H2: emotional groups against neutral
    r = welch(f(g ~= "neutral"), f(g == "neutral"), delta, false);
    rows(end + 1, :) = row(check, th, "H2", "emotional - neutral", mn, "Welch t", r, dirout(r, 1), prim); %#ok<AGROW>

    % H3 (a), (b): negative and positive groups against neutral
    ra = welch(f(g == "negative"), f(g == "neutral"), delta, false);
    rows(end + 1, :) = row(check, th, "H3", "(a) negative - neutral", mn, "Welch t", ra, dirout(ra, 1), prim); %#ok<AGROW>
    rb = welch(f(g == "positive"), f(g == "neutral"), delta, false);
    rows(end + 1, :) = row(check, th, "H3", "(b) positive - neutral", mn, "Welch t", rb, dirout(rb, 1), prim); %#ok<AGROW>

    % H3 (c) and H4: mixed group, within person, D = change(neg films) - change(pos films)
    mx = g == "mixed";
    D = P.neg(mx, m) - P.pos(mx, m);
    rc = onesample(D, delta, useq);
    rows(end + 1, :) = row(check, th, "H3", "(c) mixed: negative - positive films", mn, "paired TOST", rc, eqout(rc, delta, useq), prim); %#ok<AGROW>
    rows(end + 1, :) = row(check, th, "H4", "mixed: negative - positive films", mn, "paired t", rc, dirout(rc, 1), prim); %#ok<AGROW>

    % H3 secondary: negative group against positive group, between persons
    rs = welch(f(g == "negative"), f(g == "positive"), delta, useq);
    rows(end + 1, :) = row(check, th, "H3", "secondary: negative group - positive group", mn, "Welch TOST", rs, eqout(rs, delta, useq), false); %#ok<AGROW>

    % H3 and H4 verdicts (S only)
    if m == 1
        e3 = eqout(rc, delta, true);
        if dirout(ra, 1) == "supported" && dirout(rb, 1) == "supported" && e3 == "equivalent"
            v3 = "supported";
        else
            v3 = "not supported";
        end
        rows(end + 1, :) = row(check, th, "H3", "verdict: (a), (b) supported and (c) equivalent", mn, "", none(), v3, prim); %#ok<AGROW>
        d4 = dirout(rc, 1);
        if d4 == "supported" && e3 == "equivalent"
            v = "negative above positive, by less than the margin";
        elseif d4 == "supported"
            v = "H4: negative above positive";
        elseif e3 == "equivalent"
            v = "H3: negative and positive equivalent";
        elseif d4 == "opposite direction"
            v = "positive above negative";
        else
            v = "inconclusive";
        end
        rows(end + 1, :) = row(check, th, "H3/H4", "mixed group, joint reading", mn, "", none(), v, prim); %#ok<AGROW>
    end

    % H5: within valence.  Negative group: fear, anger, disgust, sadness (Friedman).
    M = squeeze(P.emo(g == "negative", 1:4, m));
    M = M(all(~isnan(M), 2), :);
    pf = friedman(M, 1, 'off');
    r5 = none();  r5.n1 = size(M, 1);  r5.p = pf;  r5.prank = pf;
    r5.est = max(mean(M)) - min(mean(M));
    o5a = sepout(pf);
    rows(end + 1, :) = row(check, th, "H5", "negative group: fear, anger, disgust, sadness (estimate: range of means)", mn, "Friedman", r5, o5a, prim); %#ok<AGROW>
    % Positive group: amusement - tenderness, paired
    gp = g == "positive";
    r6 = onesample(P.emo(gp, 5, m) - P.emo(gp, 6, m), delta, false);
    o5b = sepout(r6.p);
    rows(end + 1, :) = row(check, th, "H5", "positive group: amusement - tenderness", mn, "paired t", r6, o5b, prim); %#ok<AGROW>
    if m == 1
        if o5a == "no separation" && o5b == "no separation"
            v5 = "no evidence of separation";
        else
            v5 = "separation";
        end
        rows(end + 1, :) = row(check, th, "H5", "verdict", mn, "", none(), v5, prim); %#ok<AGROW>
    end
end
end

function r = none()
r = struct('n1', NaN, 'n2', NaN, 'est', NaN, 'ci', [NaN NaN], 'ci90', [NaN NaN], 'p', NaN, ...
    'prank', NaN, 'ptost', NaN);
end

function r = onesample(x, delta, tost)
x = x(~isnan(x));
r = none();
r.n1 = numel(x);
if numel(x) < 3, return; end
r.est = mean(x);
[~, r.p, r.ci] = ttest(x);
[~, ~, r.ci90] = ttest(x, 0, 'Alpha', 0.1);
r.prank = signrank(x);
if tost
    [~, pa] = ttest(x, -delta, 'Tail', 'right');
    [~, pb] = ttest(x, delta, 'Tail', 'left');
    r.ptost = max(pa, pb);
end
end

function r = welch(x, y, delta, tost)
x = x(~isnan(x));  y = y(~isnan(y));
r = none();
r.n1 = numel(x);  r.n2 = numel(y);
if numel(x) < 3 || numel(y) < 3, return; end
r.est = mean(x) - mean(y);
[~, r.p, r.ci] = ttest2(x, y, 'Vartype', 'unequal');
[~, ~, r.ci90] = ttest2(x, y, 'Vartype', 'unequal', 'Alpha', 0.1);
r.prank = ranksum(x, y);
if tost
    [~, pa] = ttest2(x + delta, y, 'Vartype', 'unequal', 'Tail', 'right');
    [~, pb] = ttest2(x - delta, y, 'Vartype', 'unequal', 'Tail', 'left');
    r.ptost = max(pa, pb);
end
end

function o = dirout(r, sgn)
if isnan(r.p), o = "not computed";
elseif r.p < 0.05 && sign(r.est) == sgn, o = "supported";
elseif r.p < 0.05, o = "opposite direction";
else, o = "not supported";
end
end

function o = eqout(r, delta, tost)
if ~tost, o = "reported (no margin for this measure)";
elseif r.ci90(1) > -delta && r.ci90(2) < delta, o = "equivalent";
else, o = "not shown equivalent";
end
end

function o = sepout(p)
if isnan(p), o = "not computed"; elseif p < 0.05, o = "separation"; else, o = "no separation"; end
end

function c = row(check, th, hyp, part, mn, test, r, outcome, prim)
c = {string(check), th, hyp, part, mn, string(test), r.n1, r.n2, r.est, r.ci(1), r.ci(2), ...
    r.ci90(1), r.ci90(2), r.p, r.prank, r.ptost, string(outcome), prim};
end

% ---- H6: the affect slider ------------------------------------------------------------------------

function rows = h6(root, B, syn)
rawdir = fullfile(root, 'data', 'raw', 'Study6');
base = {"main", 0.05, "H6", "", "S", "Spearman per person, Fisher z, one-sample t"};
if syn || ~isfolder(rawdir)
    rows = [base(1:3), {"state vs slider intensity |a - 5.5|", "S", "", NaN, NaN, NaN, NaN, NaN, ...
        NaN, NaN, NaN, NaN, NaN, "not tested: raw POPANE files not present (data/raw/Study6)", true}];
    return;
end
ids = unique(B.id, 'stable');
rI = nan(numel(ids), 1);  rV = rI;
for i = 1:numel(ids)
    f = fullfile(rawdir, sprintf('S6_P%d_All.csv', ids(i)));
    if ~isfile(f), continue; end
    opts = detectImportOptions(f, 'CommentStyle', '#', 'Delimiter', ',');
    opts.SelectedVariableNames = {'affect', 'marker'};
    opts = setvartype(opts, {'affect', 'marker'}, 'double');
    T = readtable(f, opts);
    Bi = B(B.id == ids(i) & B.marker >= 100, :);
    a = nan(height(Bi), 2);
    for j = 1:height(Bi)
        v = T.affect(T.marker == Bi.marker(j));
        a(j, :) = [mean(abs(v - 5.5), 'omitnan'), mean(v, 'omitnan')];
    end
    if numel(unique(a(:, 1))) > 1, rI(i) = corr(Bi.S, a(:, 1), 'Type', 'Spearman'); end
    if numel(unique(a(:, 2))) > 1, rV(i) = corr(Bi.S, a(:, 2), 'Type', 'Spearman'); end
end
rows = {};
for k = 1:2
    if k == 1, rr = rI; nm = "state vs slider intensity |a - 5.5|"; else, rr = rV; nm = "state vs raw valence (secondary)"; end
    rr = rr(~isnan(rr));
    z = atanh(max(min(rr, 0.995), -0.995));    % r = +-1 is possible with 6 clips
    r = onesample(z, 0, false);
    r.est = tanh(r.est);  r.ci = tanh(r.ci);  r.ci90 = tanh(r.ci90);
    if k == 1, o = dirout(r, 1); else, o = "reported"; end
    o = o + sprintf(" (%d of %d with a slider that varies)", numel(rr), numel(ids));
    rows(end + 1, :) = [base(1:3), {nm, "S", base{6}, r.n1, NaN, r.est, r.ci(1), r.ci(2), ...
        r.ci90(1), r.ci90(2), r.p, r.prank, NaN, o, k == 1}]; %#ok<AGROW>
end
end

% ---- output -------------------------------------------------------------------------------------

function print_summary(R, syn)
if syn, fprintf('\n=== CODE CHECK on made-up states (known effect), not the real data ===\n'); end
P = R(R.check == "main" & R.measure == "S" & ~contains(R.part, "descriptive"), :);
fprintf('\nMain check, state (S):\n');
for i = 1:height(P)
    fprintf('  %-6s %-62s n %3g/%-3g est %+.3f  95%% [%+.3f %+.3f]  90%% [%+.3f %+.3f]  p %.4f  rank p %.4f  TOST p %.4f  -> %s\n', ...
        P.hypothesis(i), P.part(i), P.n1(i), P.n2(i), P.estimate(i), P.ci_lo(i), P.ci_hi(i), ...
        P.ci90_lo(i), P.ci90_hi(i), P.p(i), P.p_rank(i), P.p_tost(i), P.outcome(i));
end
fprintf('\nVerdicts in every check (S):\n');
V = R(R.measure == "S" & (contains(R.part, "verdict") | R.part == "all" | R.part == "emotional - neutral" ...
    | R.part == "mixed: negative - positive films" | R.part == "mixed group, joint reading"), :);
for i = 1:height(V)
    fprintf('  %-7s %-6s %-50s %s\n', V.check(i), V.hypothesis(i), V.part(i), V.outcome(i));
end
end

function plot_changes(P, out)
g = P.group;
cats = {'neutral', 'negative', 'positive', 'mixed: neg films', 'mixed: pos films'};
lab = {'S: state (logit)', 'R: log pulse rate', 'Z: mean log height'};
fig = figure('Visible', 'off', 'Position', [100 100 1100 420]);
tl = tiledlayout(1, 3, 'TileSpacing', 'compact');
rng(7);
for m = 1:3
    nexttile; hold on;
    vals = {P.film(g == "neutral", m), P.film(g == "negative", m), P.film(g == "positive", m), ...
        P.neg(g == "mixed", m), P.pos(g == "mixed", m)};
    for k = 1:5
        v = vals{k}(~isnan(vals{k}));
        scatter(k + 0.12 * randn(size(v)), v, 18, [0.45 0.45 0.45], 'filled', 'MarkerFaceAlpha', 0.6);
        [~, ~, ci] = ttest(v);
        errorbar(k + 0.3, mean(v), mean(v) - ci(1), ci(2) - mean(v), 'o', 'Color', [0 0.45 0.9], ...
            'MarkerFaceColor', [0 0.45 0.9], 'LineWidth', 1.4, 'CapSize', 6);
    end
    yline(0, ':');
    xlim([0.4 5.7]); xticks(1:5); xticklabels(cats); xtickangle(30);
    ylabel(lab{m}); grid on;
    title(sprintf('%s, film - baseline', extractBefore(lab{m}, ':')));
end
title(tl, 'Stage 2: change from each person''s own baseline (pulses \geq 0.05 \muS; mean and 95% CI in blue)');
exportgraphics(fig, out, 'Resolution', 150);
close(fig);
end

% ---- code check -----------------------------------------------------------------------------------

function S = synth_states(S)
% Made-up states on the real layouts, with a known effect: +0.5 in every negative and positive
% clip, 0 in neutral clips and the baseline.  Pulses and sizes from the MPP model (Part 13).
rng(14);
b0 = log(0.157 / (1 - 0.157));
for id = unique(S.id)'
    k = find(S.id == id);
    mk = S.marker(k);
    x = cumsum(sqrt(2e-4) * randn(numel(k), 1));
    x(mk >= 200) = x(mk >= 200) + 0.5;
    n = double(rand(numel(k), 1) < 1 ./ (1 + exp(-(b0 + x))));
    h = nan(numel(k), 1);
    h(n == 1) = exp(-1.86 + 0.5 * x(n == 1) + sqrt(0.5) * randn(sum(n), 1));
    S.n(k) = n;  S.height_uS(k) = h;
    S.state(k) = x + 0.1 * randn(numel(k), 1);
    S.bin_state(k) = x + 0.1 * randn(numel(k), 1);
end
end
