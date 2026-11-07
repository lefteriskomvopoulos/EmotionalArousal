function tbme_run(pid, n_starts, max_windows, win_s, tau_fix)
% Amin & Faghih (IEEE TBME 68:1726, 2021) on one POPANE participant, published code unmodified.
%
% The authors' solver (Dependencies/*.p, github.com/computational-medicine-lab/
% Tonic-Phasic-Decomposition-TBME, commit d6e5eb1) is called exactly as in their scripts:
%   - signal at 2 Hz, impulses at 4 Hz, tau bounds [0.1 1.5] / [1.5 6] s, peak distance 1 s
%   - 200 s windows every 100 s (Block_Wise_Deconvolution_Main_Code.m)
%   - in each window n_starts random starts, kept by their selection rule: lowest residual among
%     converged starts whose tau_rise is off its bounds (Auditory_Stimulation_Data_Experiment...)
%   - windows joined by dropping half the overlap on each side (scripts_for_concatenation.m;
%     their script builds y_rec/y_tonic/y_phasic from y by mistake, here each is joined itself)
% Ours, stated for the report: if no start passes their rule, the lowest-residual converged
% start is kept (or the lowest-residual start, if none converged) and the window is flagged.
%
% max_windows (optional) stops after that many windows: a quick local check of this wrapper.
% win_s (optional) is the window length in seconds, the stride being half of it as in their
% script.  Default: 400 signal samples, i.e. their 200 s at 2 Hz (100 s at 4 Hz).  Other
% lengths are tests only (400 s at 2 Hz: tau did not move, ~50x slower) and give outputs a
% _w<win_s> suffix.
% tau_fix (optional, tests only) = [tau_rise tau_decay]: the tau bounds, which the authors'
% scripts pass in (data.lb/ub), are narrowed to +-5% around it, forcing tau there.  The prior
% itself is inside their .p code and cannot be changed.  Since tau then sits on a bound, which
% their selection rule rejects, windows fall back to our flagged rule.  Output suffix
% _t<100 x tau_rise>_<100 x tau_decay>.
% The signal rate is read from the input file (fs); impulses run at twice it, as published.
% 2 Hz is the published rate; 4 Hz inputs (S<k>_f4) test whether a finer view of each
% response's rise lets the data, not the priors, set tau.
%
% pid is a participant number (99 -> P99) or a file name ('P99', or 'S99' for the synthetic
% session matched to P99, written by eda.tbme_synth).
%
% Input  input/<name>.mat  (fs, y, t, marker; written by eda.tbme_export / eda.tbme_synth)
% Output output/<name>.mat, <name>_events.csv, <name>_windows.csv, <name>_signals.csv

if nargin < 2, n_starts = 8; end
if isnumeric(pid), name = sprintf('P%d', pid); else, name = char(pid); end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'Dependencies'));
out = fullfile(here, 'output');
if ~exist(out, 'dir'), mkdir(out); end
in_name = name;
S = load(fullfile(here, 'input', [in_name '.mat']));
y = S.y(:); t = S.t(:); mk = S.marker(:);

% signal rate from the input file (2 Hz as published); impulses at twice it, as published
Fsy = double(S.fs); Fsu = 2 * Fsy;
% default window keeps the published problem size, 400 signal samples (200 s at 2 Hz)
if nargin < 4 || isempty(win_s), win_s = 400 / Fsy; end
if win_s ~= 400 / Fsy, name = sprintf('%s_w%d', name, win_s); end
win = win_s; stride = win_s / 2;
ub = [1.5 6]'; lb = [0.1 1.5]'; minimum_peak_distance = 1;
if nargin >= 5 && ~isempty(tau_fix)
    lb = 0.95 * tau_fix(:); ub = 1.05 * tau_fix(:);
    name = sprintf('%s_t%d_%d', name, round(100 * tau_fix));
end

% windows exactly as the authors' block-wise loop builds them
bounds = zeros(0, 2); ll = -stride*Fsy + 1; rr = 0;
while rr < numel(y)
    ll = ll + stride*Fsy;
    rr = min(ll + win*Fsy - 1, numel(y));
    bounds(end+1, :) = [ll rr]; %#ok<AGROW>
end
if nargin >= 3 && max_windows < size(bounds, 1)
    bounds = bounds(1:max_windows, :);
    y = y(1:bounds(end,2)); t = t(1:bounds(end,2)); mk = mk(1:bounds(end,2));
end
W = size(bounds, 1);
fprintf('%s: %d samples, %d windows, %d starts each\n', name, numel(y), W, n_starts);

sel = cell(W, 1);
info = table('Size', [W 13], 'VariableTypes', ...
    {'double','double','double','double','double','double','double','double','double', ...
     'string','double','double','double'}, 'VariableNames', ...
    {'window','t_start','t_end','tau_rise','tau_decay','lambda1','lambda2','converged', ...
     'n_valid_starts','selection','r2','n_impulses','seconds'});

for w = 1:W
    data = struct('y', y(bounds(w,1):bounds(w,2)), 'ub', ub, 'lb', lb, 'Fsu', Fsu, ...
                  'Fsy', Fsy, 'minimum_peak_distance', minimum_peak_distance);
    res = cell(n_starts, 1);
    tic
    parfor i = 1:n_starts
        res{i} = CoordinateDescentWithTonicPartRemoval_with_priors(data, i);
    end
    secs = toc;

    cost = inf(n_starts, 1); conv = false(n_starts, 1); valid = false(n_starts, 1);
    for i = 1:n_starts
        r = res{i};
        cost(i) = 0.5 * norm(data.y - r.y_rec(:), 2)^2;
        tr = round(r.tauj(1)*1e4)/1e4;
        conv(i) = r.convergenceFlag == 1;
        valid(i) = conv(i) && tr ~= lb(1) && tr ~= ub(1);
    end
    if any(valid),    pool = valid; how = "authors";
    elseif any(conv), pool = conv;  how = "fallback_converged";
    else,             pool = true(n_starts, 1); how = "fallback_unconverged";
    end
    c = cost; c(~pool) = Inf; [~, k] = min(c);
    r = res{k}; sel{w} = r;

    yw = data.y;
    info(w, :) = {w, t(bounds(w,1)), t(bounds(w,2)), r.tauj(1), r.tauj(2), r.lambda1, ...
                  r.lambda2, double(r.convergenceFlag), sum(valid), how, ...
                  1 - sum((yw - r.y_rec(:)).^2) / sum((yw - mean(yw)).^2), nnz(r.uj), secs};
    fprintf('window %d/%d: tau %.3f / %.3f s, %d impulses, R2 %.3f, %s, %.0f s\n', w, W, ...
            r.tauj(1), r.tauj(2), nnz(r.uj), info.r2(w), how, secs);
end

% join the windows, keeping the central part of each (half the overlap dropped per side)
half_y = stride*Fsy/2; half_u = stride*Fsu/2;
J = struct('y', [], 'y_rec', [], 'y_tonic', [], 'y_phasic', [], 'u', [], 'tu', []);
for w = 1:W
    r = sel{w}; n = bounds(w,2) - bounds(w,1) + 1; nu = numel(r.uj);
    assert(nu == n*Fsu/Fsy, 'window %d: %d impulse samples for %d signal samples', w, nu, n);
    a = 1 + (w > 1)*half_y;  b = n  - (w < W)*half_y;
    au = 1 + (w > 1)*half_u; bu = nu - (w < W)*half_u;
    seg = bounds(w,1) - 1 + (a:b);
    J.y        = [J.y;        y(seg)];
    J.y_rec    = [J.y_rec;    reshape(r.y_rec(a:b), [], 1)];
    J.y_tonic  = [J.y_tonic;  reshape(r.y_tonic(a:b), [], 1)];
    J.y_phasic = [J.y_phasic; reshape(r.y_phasic(a:b), [], 1)];
    J.u        = [J.u;        reshape(r.uj(au:bu), [], 1)];
    J.tu       = [J.tu;       t(bounds(w,1)) + ((au:bu)' - 1) / Fsu];
end
assert(numel(J.y) == numel(y), 'joined signal has %d samples, expected %d', numel(J.y), numel(y));

k = J.u > 0;
writetable(table(J.tu(k), J.u(k), 'VariableNames', {'time_s', 'amplitude'}), ...
           fullfile(out, [name '_events.csv']));
writetable(info, fullfile(out, [name '_windows.csv']));
writetable(table(t, J.y, J.y_rec, J.y_tonic, J.y_phasic, mk, 'VariableNames', ...
           {'time_s', 'y', 'y_rec', 'tonic', 'phasic', 'marker'}), ...
           fullfile(out, [name '_signals.csv']));
save(fullfile(out, [name '.mat']), 'sel', 'info', 'J', 'bounds', 'n_starts');

r2 = 1 - sum((J.y - J.y_rec).^2) / sum((J.y - mean(J.y)).^2);
fprintf(['%s done: %d impulses (%.1f/min), R2 %.3f, median tau %.2f / %.2f s, ' ...
         '%d/%d windows by the authors'' rule, %.0f s\n'], name, nnz(k), ...
        nnz(k) / (numel(y)/Fsy/60), r2, median(info.tau_rise), median(info.tau_decay), ...
        sum(info.selection == "authors"), W, sum(info.seconds));
end
