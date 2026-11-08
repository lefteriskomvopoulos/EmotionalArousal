function own_run(name, algorithm, varargin)
% One session through our own deconvolution, windows and random starts included.
%
%   own_run('S4_own4')                                     amin2021, starting settings
%   own_run('S4_own4', 'faghih2014')                       the Data Challenge algorithm
%   own_run('S4_own4', 'amin2021', 'tag', 'prior01', ...
%           'prior_lambda', [0.01; 0.01], 'knot_s', 10)    any own_config setting
%
% Input   input/<name>.mat (fs, y, t, marker), written by src/eda/tbme_export.save
% Output  output/<tag>/<name>.mat, _events.csv, _windows.csv, _signals.csv, scored by
%         src/eda/tbme_score.py.  The columns are those of the black box's run script
%         (black-box branch), so the two can be compared directly.
%
% Selection and joining as in that script: lowest residual among converged starts with tau_rise
% off its limits, falling back (flagged) when no start qualifies, and half the overlap dropped
% on each side of a window.

if nargin < 2, algorithm = 'amin2021'; end
c = own_config(algorithm);
tag = algorithm;
for k = 1:2:numel(varargin)
    if strcmp(varargin{k}, 'tag')
        tag = varargin{k + 1};
    elseif isfield(c, varargin{k})
        c.(varargin{k}) = varargin{k + 1};
    else
        error('own_config has no setting %s', varargin{k});
    end
end

here = fileparts(mfilename('fullpath'));
out = fullfile(here, 'output', tag);
if ~exist(out, 'dir'), mkdir(out); end
S = load(fullfile(here, 'input', [name '.mat']));
y = S.y(:); t = S.t(:); mk = S.marker(:);
c.fsy = double(S.fs);
Fsu = c.fsy * c.u_per_y;
stride = c.window_s / 2;

bounds = zeros(0, 2); ll = -stride * c.fsy + 1; rr = 0;
while rr < numel(y)
    ll = ll + stride * c.fsy;
    rr = min(ll + c.window_s * c.fsy - 1, numel(y));
    bounds(end + 1, :) = [ll rr]; %#ok<AGROW>
end
W = size(bounds, 1);
fprintf('%s, %s: %d samples, %d windows, %d starts each\n', name, tag, numel(y), W, c.n_starts);

sel = cell(W, 1);
info = table('Size', [W 13], 'VariableTypes', ...
    {'double','double','double','double','double','double','double','double','double', ...
     'string','double','double','double'}, 'VariableNames', ...
    {'window','t_start','t_end','tau_rise','tau_decay','lambda1','lambda2','converged', ...
     'n_valid_starts','selection','r2','n_impulses','seconds'});

for w = 1:W
    yw = y(bounds(w, 1):bounds(w, 2));
    res = cell(c.n_starts, 1);
    tic
    parfor i = 1:c.n_starts
        res{i} = own_deconvolve(yw, c, (w - 1) * 1000 + i);
    end
    secs = toc;

    fit = cellfun(@(r) r.fit, res);
    conv = cellfun(@(r) r.convergenceFlag == 1, res);
    off = cellfun(@(r) abs(r.tauj(1) - c.tau_lb(1)) > 1e-4 && abs(r.tauj(1) - c.tau_ub(1)) > 1e-4, res);
    valid = conv & off;
    if any(valid),    pool = valid; how = "authors";
    elseif any(conv), pool = conv;  how = "fallback_converged";
    else,             pool = true(c.n_starts, 1); how = "fallback_unconverged";
    end
    fit(~pool) = Inf;
    [~, k] = min(fit);
    r = res{k};
    sel{w} = r;

    info(w, :) = {w, t(bounds(w, 1)), t(bounds(w, 2)), r.tauj(1), r.tauj(2), r.lambda1, ...
                  r.lambda2, r.convergenceFlag, sum(valid), how, ...
                  1 - sum((yw - r.y_rec) .^ 2) / sum((yw - mean(yw)) .^ 2), nnz(r.uj), secs};
    fprintf('window %d/%d: tau %.3f / %.3f s, %d impulses, R2 %.3f, %s, %.0f s\n', w, W, ...
            r.tauj(1), r.tauj(2), nnz(r.uj), info.r2(w), how, secs);
end

% join the windows.  u_1 of a window sits one Ty before its first y sample (own_grid).
half_y = stride * c.fsy / 2; half_u = half_y * c.u_per_y;
J = struct('y', [], 'y_rec', [], 'y_tonic', [], 'y_phasic', [], 'u', [], 'tu', []);
for w = 1:W
    r = sel{w}; n = bounds(w, 2) - bounds(w, 1) + 1; nu = numel(r.uj);
    a = 1 + (w > 1) * half_y;  b = n - (w < W) * half_y;
    au = 1 + (w > 1) * half_u; bu = nu - (w < W) * half_u;
    seg = bounds(w, 1) - 1 + (a:b);
    J.y        = [J.y;        y(seg)];
    J.y_rec    = [J.y_rec;    r.y_rec(a:b)];
    J.y_tonic  = [J.y_tonic;  r.y_tonic(a:b)];
    J.y_phasic = [J.y_phasic; r.y_phasic(a:b)];
    J.u        = [J.u;        r.uj(au:bu)];
    J.tu       = [J.tu;       t(bounds(w, 1)) - 1 / c.fsy + ((au:bu)' - 1) / Fsu];
end
assert(numel(J.y) == numel(y), 'joined signal has %d samples, expected %d', numel(J.y), numel(y));

k = J.u > 0;
writetable(table(J.tu(k), J.u(k), 'VariableNames', {'time_s', 'amplitude'}), ...
           fullfile(out, [name '_events.csv']));
writetable(info, fullfile(out, [name '_windows.csv']));
writetable(table(t, J.y, J.y_rec, J.y_tonic, J.y_phasic, mk, 'VariableNames', ...
           {'time_s', 'y', 'y_rec', 'tonic', 'phasic', 'marker'}), ...
           fullfile(out, [name '_signals.csv']));
save(fullfile(out, [name '.mat']), 'sel', 'info', 'J', 'bounds', 'c');

r2 = 1 - sum((J.y - J.y_rec) .^ 2) / sum((J.y - mean(J.y)) .^ 2);
fprintf('%s %s done: %d impulses (%.1f/min), R2 %.3f, median tau %.2f / %.2f s, %d/%d by the rule, %.0f s\n', ...
        name, tag, nnz(k), nnz(k) / (numel(y) / c.fsy / 60), r2, median(info.tau_rise), ...
        median(info.tau_decay), sum(info.selection == "authors"), W, sum(info.seconds));
end
