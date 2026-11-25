function res = s2_filter_mpp(n, r, opt)
% Stage 2, marked point process (MPP) filter: pulses + their sizes -> hidden arousal state.
%
% Taken from filter_one_mpp.m, the course's Data Challenge 2 filter code as used in Piyush More's
% submission (NPSP 2026); it implements Wickramasuriya & Faghih, IEEE Access 8:68499, 2020.  The
% model, the Newton update, the smoother and the EM updates are the Data Challenge's, line for
% line (s2_test_dc reproduces its saved results exactly):
%
%     x_k = x_{k-1} + e_k,                e_k ~ N(0, ve)       hidden arousal state
%     P(n_k = 1 | x_k) = 1/(1+exp(-(b0 + x_k)))                did a pulse occur in bin k?
%     r_k | n_k = 1  ~ N(r0 + r1*x_k, vr)                      how big was it?
%
%   b0 is fixed at the overall pulse log-odds (only b0 + x_k enters, so the level of x is pinned
%   to it).  EM learns ve, r0, r1, vr; the first state x_1 is carried over between iterations.
%
% What differs from filter_one_mpp.m (interface only, the arithmetic is unchanged):
%   1. n and r are passed separately.  In the Data Challenge a pulse was any u > 0 and r = u; here
%      r is the log pulse height, which is negative for heights below 1 uS, so u > 0 cannot mark
%      the pulses.
%   2. The starting values are an option (opt.init).  'dc' = the Data Challenge's values (used by
%      s2_test_dc to reproduce its results); 'data' = from the data, the rules of Data Challenge
%      Problem 3F: r0 = mean(r), r1 = std(r), vr = var(r)/2 over the pulse bins, ve = 5e-3.
%   3. Silent, and it reports why it stopped (res.stop): 'converged', 'r1<=0' (the Data
%      Challenge's stop when the size slope turns non-positive: arousal cannot shrink pulses) or
%      'max_iter'.
%   4. The confidence limits of p_k (a 1e4-point integral per bin) are left out; they are not
%      used in the project.  The state's 95% limits are kept.
%
% n    1 x K, 0/1 pulse in bin k
% r    1 x K, pulse size in bin k (ignored where n == 0)
% opt  struct, optional: init ('data' | 'dc'), M (max EM iterations, 5e4), tol (1e-4)

if nargin < 3, opt = struct(); end
if ~isfield(opt, 'init'), opt.init = 'data'; end
if ~isfield(opt, 'M'),    opt.M = 5e4;       end
if ~isfield(opt, 'tol'),  opt.tol = 1e-4;    end

n = double(n(:)');
r = double(r(:)');
r(n == 0) = 0;
K = length(n);
pt = find(n == 1);

M = opt.M;
ve = zeros(1, M);
r0 = zeros(1, M);
r1 = zeros(1, M);
vr = zeros(1, M);

x_pred = zeros(1, K);  v_pred = zeros(1, K);
x_updt = zeros(1, K);  v_updt = zeros(1, K);
x_smth = zeros(1, K);  v_smth = zeros(1, K);
p_updt = zeros(1, K);
A = zeros(1, K);  W = zeros(1, K);  CW = zeros(1, K);

base_prob = sum(n) / length(n);
b0 = log(base_prob / (1 - base_prob));

switch opt.init
    case 'dc'      % filter_one_mpp.m
        ve(1) = 5e-3;  r0(1) = 0.003;  r1(1) = 0.001;  vr(1) = 0.002;
    case 'data'    % Data Challenge Problem 3F rules
        ve(1) = 5e-3;  r0(1) = mean(r(pt));  r1(1) = std(r(pt));  vr(1) = var(r(pt)) / 2;
    otherwise
        error('unknown init %s', opt.init);
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
        x_updt(k) = posterior_mode(x_pred(k), v_pred(k), r(k), r0(m), r1(m), vr(m), b0, n(k));
        p_updt(k) = 1 / (1 + exp(-(b0 + x_updt(k))));
        if n(k) == 0
            v_updt(k) = 1 / ((1 / v_pred(k)) + p_updt(k) * (1 - p_updt(k)));
        else
            v_updt(k) = 1 / ((1 / v_pred(k)) + ((r1(m) ^ 2) / vr(m)) + p_updt(k) * (1 - p_updt(k)));
        end
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
    R = linear_parameters(x_smth, W, r, pt);
    if R(2) > 0
        r0(m + 1) = R(1);
        r1(m + 1) = R(2);
        vr(m + 1) = maximum_variance(r, r0(m + 1), r1(m + 1), W, x_smth, pt);
    else
        stop = 'r1<=0';
        break;
    end
    ve(m + 1) = (sum(W(2:end)) + sum(W(1:end-1)) - 2 * sum(CW)) / K;

    mean_dev = mean(abs([ve(m + 1) r0(m + 1) r1(m + 1) vr(m + 1)] - [ve(m) r0(m) r1(m) vr(m)]));
    if mean_dev < opt.tol
        stop = 'converged';
        break;
    end

    % reset for the next iteration (x_smth(1) is kept: it starts the next forward pass)
    x_pred(:) = 0;  v_pred(:) = 0;  x_updt(:) = 0;  v_updt(:) = 0;
    x_smth(2:end) = 0;  v_smth(:) = 0;  p_updt(:) = 0;
    A(:) = 0;  W(:) = 0;  CW(:) = 0;
end

res = struct();
res.n = n;
res.r = r;
res.state = x_smth;
res.var = v_smth;
res.prob = 1 ./ (1 + exp(-(b0 + x_smth)));
res.mpp_est = r0(m) + r1(m) * x_smth;
res.state_lcl = norminv(0.025, x_smth, sqrt(v_smth));
res.state_ucl = norminv(0.975, x_smth, sqrt(v_smth));
res.certainty = 1 - normcdf(prctile(x_smth, 50) * ones(1, K), x_smth, sqrt(v_smth));
res.param_b0 = b0;
res.param_ve = ve(m);
res.param_r0 = r0(m);
res.param_r1 = r1(m);
res.param_vr = vr(m);
res.iterations = m;
res.stop = stop;
end

function y = posterior_mode(x_pred, v_pred, z, r0, r1, vr, b0, n)
% Newton-Raphson for the mode of the posterior of x_k (filter_one_mpp.m, get_posterior_mode)
M = 100;
it = zeros(1, M);
it(1) = x_pred;
for i = 1:(M - 1)
    if n == 0
        C = v_pred;
        f = it(i) - x_pred - C * (n - exp(b0 + it(i)) / (1 + exp(b0 + it(i))));
        df = 1 + C * exp(b0 + it(i)) / (1 + exp(b0 + it(i))) ^ 2;
    else
        C = v_pred / ((r1 ^ 2) * v_pred + vr);
        f = it(i) - x_pred - C * (r1 * (z - r0 - r1 * x_pred) + vr * (n - 1 / (1 + exp(-(b0 + it(i))))));
        df = 1 + C * vr * exp(b0 + it(i)) / ((1 + exp(b0 + it(i))) ^ 2);
    end
    it(i + 1) = it(i) - f / df;
    if abs(it(i + 1) - it(i)) < 1e-14
        y = it(i + 1);
        return
    end
end
error('Newton-Raphson failed to converge.');
end

function y = maximum_variance(z, r0, r1, W, x_smth, pt)
x_smth = x_smth(pt);  W = W(pt);  z = z(pt);  K = length(pt);
y = (z * z' + K * (r0 ^ 2) + (r1 ^ 2) * sum(W) ...
    - 2 * r0 * sum(z) - 2 * r1 * dot(x_smth, z) + 2 * r0 * r1 * sum(x_smth)) / K;
end

function y = linear_parameters(x_smth, W, z, pt)
x_smth = x_smth(pt);  W = W(pt);  z = z(pt);  K = length(pt);
y = [K sum(x_smth); sum(x_smth) sum(W)] \ [sum(z); sum(z .* x_smth)];
end
