function res = s2_filter_tonic(n, r, s, opt)
% Stage 2, variant `tonic14` (docs/EXPERIMENTS.md Part 15): the main filter plus one continuous
% observation, the tonic skin-conductance level, observed only in some bins.
%
%     x_k = x_{k-1} + e_k,                 e_k ~ N(0, ve)
%     P(n_k = 1 | x_k) = 1/(1+exp(-(b0 + x_k)))          b0 pinned to the pulse log-odds, b1 = 1
%     r_k | n_k = 1 ~ N(r0 + r1*x_k, vr)                  sizes (opt.sizes = true: the MPP filter)
%     s_k ~ N(s0 + s1*x_k, vs)  where s_k is not NaN      the continuous observation
%
% Without the continuous observation this is s2_filter_mpp (sizes on) line for line.  The
% continuous terms (Newton update, posterior variance, M step) are those of the Data Challenge 2
% filters filter_one_mpp_one_cont.m and filter_one_bin_one_cont.m (Piyush More's submission,
% NPSP 2026) with b1 = 1, restricted to the bins where s is observed.  With sizes off, s in every
% bin and opt.init = 'dc', it reproduces filter_one_bin_one_cont exactly (s2_test_dc, Problem 3C).
%
% Not taken over: the Data Challenge's variance floor (line 174) and balance rule (line 165).  The
% weight of s is set by how often it is observed (Part 15, option 3: once per 14 s).  No sign
% constraint on s1.  Stops: 'converged', 'r1<=0' (sizes on, as s2_filter_mpp), 'vs<=0', 'max_iter'.
%
% opt.floor = true is variant tonic14b (Part 16, the balanced floor): one observation of s may
% carry at most the information the pulses carry over opt.gap bins,
%     s1^2 / vs <= I_pulse = gap * pbar * (1 - pbar),    pbar = the pulse fraction per bin,
% applied as vs <- max(vs, s1^2 / I_pulse) to the starting value and after every M step (after
% the vs <= 0 check).  With opt.floor = false (default) the code is tonic14 unchanged.
%
% opt.fixed = [s0 s1 vs] is variant tonic14c (Part 18): the link is not estimated; it starts at
% these values (vs then floored if opt.floor) and the M step keeps them.  EM still estimates ve
% and, with sizes, r0, r1, vr.  With opt.fixed empty (default) the code is unchanged.
%
% n    1 x K, 0/1 pulse in bin k
% r    1 x K, log pulse size (ignored where n == 0, or when opt.sizes is false; may be [])
% s    1 x K, continuous observation, NaN where not observed
% opt  sizes (true), init ('data' | 'dc'), M (5e4), tol (1e-4), floor (false), gap (14), fixed ([])

if nargin < 4, opt = struct(); end
if ~isfield(opt, 'sizes'), opt.sizes = true;   end
if ~isfield(opt, 'init'),  opt.init = 'data';  end
if ~isfield(opt, 'M'),     opt.M = 5e4;        end
if ~isfield(opt, 'tol'),   opt.tol = 1e-4;     end
if ~isfield(opt, 'floor'), opt.floor = false;  end
if ~isfield(opt, 'gap'),   opt.gap = 14;       end
if ~isfield(opt, 'fixed'), opt.fixed = [];     end
fixed = ~isempty(opt.fixed);

n = double(n(:)');
K = length(n);
if isempty(r), r = zeros(1, K); end
r = double(r(:)');
r(n == 0) = 0;
s = double(s(:)');
so = ~isnan(s);                      % bins where s is observed
ps = find(so);
pt = find(n == 1);
sz = opt.sizes;

M = opt.M;
ve = zeros(1, M);
r0 = zeros(1, M);  r1 = zeros(1, M);  vr = zeros(1, M);
s0 = zeros(1, M);  s1 = zeros(1, M);  vs = zeros(1, M);

x_pred = zeros(1, K);  v_pred = zeros(1, K);
x_updt = zeros(1, K);  v_updt = zeros(1, K);
x_smth = zeros(1, K);  v_smth = zeros(1, K);
p_updt = zeros(1, K);
A = zeros(1, K);  W = zeros(1, K);  CW = zeros(1, K);

base_prob = sum(n) / length(n);
b0 = log(base_prob / (1 - base_prob));

switch opt.init
    case 'dc'      % filter_one_bin_one_cont.m (continuous only)
        ve(1) = 0.05;  s0(1) = 0.1;  s1(1) = s(ps(1));  vs(1) = 0.002;
        r0(1) = 0.003;  r1(1) = 0.001;  vr(1) = 0.002;
    case 'data'    % Data Challenge Problem 3F rules, for the sizes and for s
        ve(1) = 5e-3;
        if sz, r0(1) = mean(r(pt));  r1(1) = std(r(pt));  vr(1) = var(r(pt)) / 2; end
        s0(1) = mean(s(ps));  s1(1) = std(s(ps));  vs(1) = var(s(ps)) / 2;
    otherwise
        error('unknown init %s', opt.init);
end
if fixed
    s0(1) = opt.fixed(1);  s1(1) = opt.fixed(2);  vs(1) = opt.fixed(3);
end
I_pulse = opt.gap * base_prob * (1 - base_prob);
floored = false;
if opt.floor
    [vs(1), floored] = balanced_floor(vs(1), s1(1), I_pulse);
end
x_smth(1) = 0;

stop = 'max_iter';
for m = 1:M

    % ---- E step: forward filter --------------------------------------------------------------
    for k = 1:K
        if k == 1
            x_pred(k) = x_smth(1);
            v_pred(k) = ve(m) + ve(m);
        else
            x_pred(k) = x_updt(k - 1);
            v_pred(k) = v_updt(k - 1) + ve(m);
        end
        usz = sz && n(k) == 1;
        x_updt(k) = posterior_mode(x_pred(k), v_pred(k), r(k), r0(m), r1(m), vr(m), b0, n(k), ...
            usz, so(k), s(k), s0(m), s1(m), vs(m));
        p_updt(k) = 1 / (1 + exp(-(b0 + x_updt(k))));
        a = 1 / v_pred(k);
        if usz,   a = a + ((r1(m) ^ 2) / vr(m)); end
        if so(k), a = a + ((s1(m) ^ 2) / vs(m)); end
        v_updt(k) = 1 / (a + p_updt(k) * (1 - p_updt(k)));
    end

    % ---- E step: backward smoother -----------------------------------------------------------
    x_smth(K) = x_updt(K);
    v_smth(K) = v_updt(K);
    W(K) = v_smth(K) + x_smth(K) ^ 2;
    A(1:end-1) = v_updt(1:end-1) ./ v_pred(2:end);
    for k = (K - 1):-1:1
        x_smth(k) = x_updt(k) + A(k) * (x_smth(k + 1) - x_pred(k + 1));
        v_smth(k) = v_updt(k) + (A(k) ^ 2) * (v_smth(k + 1) - v_pred(k + 1));
        CW(k) = A(k) * v_smth(k + 1) + x_smth(k) * x_smth(k + 1);
        W(k) = v_smth(k) + x_smth(k) ^ 2;
    end

    if m == M, break; end

    % ---- M step ------------------------------------------------------------------------------
    if sz
        R = linear_parameters(x_smth, W, r, pt);
        if R(2) > 0
            r0(m + 1) = R(1);
            r1(m + 1) = R(2);
            vr(m + 1) = maximum_variance(r, r0(m + 1), r1(m + 1), W, x_smth, pt);
        else
            stop = 'r1<=0';
            break;
        end
    end
    if fixed                       % Part 18: the calibrated link is kept
        s0(m + 1) = s0(m);  s1(m + 1) = s1(m);  vs(m + 1) = vs(m);
    else
        S = linear_parameters(x_smth, W, s, ps);
        s0(m + 1) = S(1);
        s1(m + 1) = S(2);
        vs(m + 1) = maximum_variance(s, s0(m + 1), s1(m + 1), W, x_smth, ps);
        if ~(vs(m + 1) > 0)
            stop = 'vs<=0';
            break;
        end
        if opt.floor
            [vs(m + 1), floored] = balanced_floor(vs(m + 1), s1(m + 1), I_pulse);
        end
    end
    ve(m + 1) = (sum(W(2:end)) + sum(W(1:end-1)) - 2 * sum(CW)) / K;

    if sz
        mean_dev = mean(abs([ve(m + 1) r0(m + 1) r1(m + 1) vr(m + 1) s0(m + 1) s1(m + 1) vs(m + 1)] - ...
                            [ve(m) r0(m) r1(m) vr(m) s0(m) s1(m) vs(m)]));
    else           % the order of filter_one_bin_one_cont.m
        mean_dev = mean(abs([ve(m + 1) s0(m + 1) s1(m + 1) vs(m + 1)] - [ve(m) s0(m) s1(m) vs(m)]));
    end
    if mean_dev < opt.tol
        stop = 'converged';
        break;
    end

    x_pred(:) = 0;  v_pred(:) = 0;  x_updt(:) = 0;  v_updt(:) = 0;
    x_smth(2:end) = 0;  v_smth(:) = 0;  p_updt(:) = 0;
    A(:) = 0;  W(:) = 0;  CW(:) = 0;
end

res = struct();
res.n = n;
res.r = r;
res.s = s;
res.state = x_smth;
res.var = v_smth;
res.prob = 1 ./ (1 + exp(-(b0 + x_smth)));
res.cont_est = s0(m) + s1(m) * x_smth;
res.state_lcl = norminv(0.025, x_smth, sqrt(v_smth));
res.state_ucl = norminv(0.975, x_smth, sqrt(v_smth));
res.param_b0 = b0;
res.param_ve = ve(m);
res.param_r0 = r0(m);  res.param_r1 = r1(m);  res.param_vr = vr(m);
res.param_s0 = s0(m);  res.param_s1 = s1(m);  res.param_vs = vs(m);
res.sizes = sz;
res.floor = opt.floor;
res.I_pulse = I_pulse;
res.info_ratio = (s1(m) ^ 2) / vs(m) / I_pulse;   % tonic information per observation / I_pulse
res.floored = floored;                             % the floor was active at the last M step
res.n_cont = numel(ps);
res.iterations = m;
res.stop = stop;
end

function y = posterior_mode(x_pred, v_pred, r, r0, r1, vr, b0, n, usz, so, s, s0, s1, vs)
% Newton-Raphson for the posterior mode; each case is the Data Challenge's formula
M = 200;
it = zeros(1, M);
it(1) = x_pred;
for i = 1:(M - 1)
    if ~so && ~usz          % filter_one_mpp.m, n == 0 (also the binary case)
        C = v_pred;
        f = it(i) - x_pred - C * (n - exp(b0 + it(i)) / (1 + exp(b0 + it(i))));
        df = 1 + C * exp(b0 + it(i)) / (1 + exp(b0 + it(i))) ^ 2;
    elseif ~so && usz       % filter_one_mpp.m, n == 1
        C = v_pred / ((r1 ^ 2) * v_pred + vr);
        f = it(i) - x_pred - C * (r1 * (r - r0 - r1 * x_pred) + vr * (n - 1 / (1 + exp(-(b0 + it(i))))));
        df = 1 + C * vr * exp(b0 + it(i)) / ((1 + exp(b0 + it(i))) ^ 2);
    elseif so && ~usz       % filter_one_bin_one_cont.m (= filter_one_mpp_one_cont.m, n == 0, b1 = 1)
        C = v_pred / ((s1 ^ 2) * v_pred + vs);
        f = it(i) - x_pred - C * (s1 * (s - s0 - s1 * x_pred) + vs * (n - (1 / (1 + exp((-1) * (b0 + it(i)))))));
        df = 1 + C * vs * exp(b0 + it(i)) / ((1 + exp(b0 + it(i))) ^ 2);
    else                    % filter_one_mpp_one_cont.m, n == 1, b1 = 1
        C = v_pred / (vr * vs + v_pred * ((r1 ^ 2) * vs + (s1 ^ 2) * vr));
        f = it(i) - x_pred - C * (r1 * vs * (r - r0 - r1 * x_pred) + s1 * vr * (s - s0 - s1 * x_pred) + ...
            vr * vs * (n - (1 / (1 + exp((-1) * (b0 + it(i)))))));
        df = 1 + C * vr * vs * exp(b0 + it(i)) / ((1 + exp(b0 + it(i))) ^ 2);
    end
    it(i + 1) = it(i) - f / df;
    if abs(it(i + 1) - it(i)) < 1e-14
        y = it(i + 1);
        return
    end
end
error('Newton-Raphson failed to converge.');
end

function [vs, hit] = balanced_floor(vs, s1, I_pulse)
% Part 16: s1^2 / vs <= I_pulse
lo = (s1 ^ 2) / I_pulse;
hit = vs < lo;
if hit, vs = lo; end
end

function y = maximum_variance(z, a0, a1, W, x_smth, idx)
x_smth = x_smth(idx);  W = W(idx);  z = z(idx);  K = length(idx);
y = (z * z' + K * (a0 ^ 2) + (a1 ^ 2) * sum(W) ...
    - 2 * a0 * sum(z) - 2 * a1 * dot(x_smth, z) + 2 * a0 * a1 * sum(x_smth)) / K;
end

function y = linear_parameters(x_smth, W, z, idx)
x_smth = x_smth(idx);  W = W(idx);  z = z(idx);  K = length(idx);
y = [K sum(x_smth); sum(x_smth) sum(W)] \ [sum(z); sum(z .* x_smth)];
end
