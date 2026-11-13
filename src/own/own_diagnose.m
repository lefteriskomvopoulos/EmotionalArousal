function own_diagnose()
% Historical (round 1): on the first input (0.5 Hz filter, 2 Hz) and own_config's starting
% values, not the current setting.  Kept for the record; no result of the pipeline depends on it.
%
% Where do the extra pulses come from?  About 5 minutes on a laptop, one start per run.
%
% Two made-up 200 s windows with 13 known pulses, a slow tonic and noise, built at 20 Hz and sent
% through the same 0.5 Hz filter and 2 Hz sampling as the real input: one with tau near the prior
% and one far from it.  Each runs under the settings listed in variants.
variants = {
    'prior as paper',      {}
    'prior / 10',          {'prior_lambda', [0.01; 0.01]}
    'prior / 100',         {'prior_lambda', [0.001; 0.001]}
    'no prior',            {'use_prior', false}
    'prior / 10, rise 3',  {'prior_lambda', [0.01; 0.01], 'tau_ub', [3; 6]}
    'prior / 100, rise 3', {'prior_lambda', [0.001; 0.001], 'tau_ub', [3; 6]}
    'no prior, rise 3',    {'use_prior', false, 'tau_ub', [3; 6]}
    };
for tau = {[0.70; 3.00], [1.20; 5.00]}
    [y, ev, g] = make_window(tau{1});
    fprintf('\ntrue tau %.2f/%.2f, %d true pulses\n', tau{1}(1), tau{1}(2), numel(ev));
    for v = 1:size(variants, 1)
        c = own_config('amin2021');
        set = variants{v, 2};
        for k = 1:2:numel(set)
            c.(set{k}) = set{k + 1};
        end
        r = own_deconvolve(y, c, 1);
        h = r.history;
        est = (find(r.uj) - 1) * g.Tu;
        fprintf('%-20s %2d found, %3d returned (end of phase 1: %3d), tau %.2f/%.2f, converged %d, %3d phase 2 steps, last move %.3f, %3.0f s\n', ...
                variants{v, 1}, hits(ev, est), numel(est), h(find(h(:, 1) == 1, 1, 'last'), 5), ...
                r.tauj(1), r.tauj(2), r.convergenceFlag, sum(h(:, 1) == 2), h(end, 9), r.seconds);
    end
end
end

function [y, ev, g] = make_window(tau)
c = own_config('amin2021');
g = own_grid(400, c);
rng(7);
Fs = 20;
t = (0:1 / Fs:200)';
ev = (8:15:188)' + 3 * rand(13, 1);
w = 8 * exp(0.7 * randn(13, 1));
phasic = zeros(size(t));
for k = 1:13
    s = max(t - ev(k), 0);
    phasic = phasic + w(k) * g.Tu * (exp(-s / tau(2)) - exp(-s / tau(1))) / (tau(2) - tau(1));
end
x = phasic + 2 + 0.5 * sin(2 * pi * t / 250) + 0.02 * randn(size(t));

% the real chain's filter: 6.4 s Hamming FIR at 0.5 Hz, no delay
N = round(6.4 * Fs);
n = (-N / 2:N / 2)';
b = sin(2 * pi * 0.5 * n / Fs) ./ (pi * n);
b(N / 2 + 1) = 2 * 0.5 / Fs;
b = b .* (0.54 - 0.46 * cos(2 * pi * (0:N)' / N));
b = b / sum(b);
low = conv([x(1) * ones(N / 2, 1); x; x(end) * ones(N / 2, 1)], b, 'valid');
y = low(round(g.k * g.Ty * Fs) + 1);
end

function n = hits(ev, est)
% one-to-one matches within 1 s, closest first
n = 0;
used = false(size(est));
for k = 1:numel(ev)
    d = abs(est - ev(k));
    d(used) = inf;
    [m, i] = min(d);
    if ~isempty(m) && m <= 1
        used(i) = true;
        n = n + 1;
    end
end
end
