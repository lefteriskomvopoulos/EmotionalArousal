function res = s2_filter_bin(n, opt)
% Stage 2, binary filter: pulse timing only -> hidden arousal state.
%
% Taken from filter_one_bin.m, the course's Data Challenge 2 filter code as used in Piyush More's
% submission (NPSP 2026); after Smith et al., J Neurosci 24:447, 2004 and Wickramasuriya et al.,
% Front Neurosci 13:780, 2019.  s2_test_dc reproduces its saved results exactly.  Same state
% model as s2_filter_mpp without the size observation:
%
%     x_k = x_{k-1} + e_k,  e_k ~ N(0, ve);     P(n_k = 1 | x_k) = 1/(1+exp(-(b0 + x_k)))
%
% EM learns ve and the starting state x0.  In the Data Challenge this filter converged where the
% MPP filter stopped because pulse sizes did not follow the state (Subjects 2 and 8), so in
% stage 2 it is the fallback and the timing-only check.
%
% What differs from filter_one_bin.m (interface only, the arithmetic is unchanged): silent;
% reports why it stopped (res.stop: 'converged' or 'max_iter'); the confidence limits of p_k
% (a 1e6-point integral per bin) are left out.
%
% n    1 x K, 0/1 pulse in bin k
% opt  struct, optional: M (max EM iterations, 2e4), tol (1e-4), ve0 (0.05)

if nargin < 2, opt = struct(); end
if ~isfield(opt, 'M'),   opt.M = 2e4;    end
if ~isfield(opt, 'tol'), opt.tol = 1e-4; end
if ~isfield(opt, 've0'), opt.ve0 = 0.05; end

n = double(n(:)');
K = length(n);
M = opt.M;
ve = zeros(1, M);

x_pred = zeros(1, K);  v_pred = zeros(1, K);
x_updt = zeros(1, K);  v_updt = zeros(1, K);
x_smth = zeros(1, K);  v_smth = zeros(1, K);
p_updt = zeros(1, K);
A = zeros(1, K);  W = zeros(1, K);  CW = zeros(1, K);

base_prob = sum(n) / length(n);
b0 = log(base_prob / (1 - base_prob));
ve(1) = opt.ve0;
x_smth(1) = 0;

stop = 'max_iter';
for m = 1:M
    for k = 1:K
        if k == 1
            x_pred(k) = x_smth(1);
            v_pred(k) = ve(m) + ve(m);
        else
            x_pred(k) = x_updt(k - 1);
            v_pred(k) = v_updt(k - 1) + ve(m);
        end
        x_updt(k) = state_update(x_pred(k), v_pred(k), b0, n(k));
        p_updt(k) = 1 / (1 + exp(-(b0 + x_updt(k))));
        v_updt(k) = 1 / ((1 / v_pred(k)) + p_updt(k) * (1 - p_updt(k)));
    end

    x_smth(K) = x_updt(K);
    v_smth(K) = v_updt(K);
    W(K) = v_smth(K) + x_smth(K) ^ 2;
    A(1:end-1) = v_updt(1:end-1) ./ v_pred(2:end);
    x0_prev = x_smth(1);
    for k = (K - 1):-1:1
        x_smth(k) = x_updt(k) + A(k) * (x_smth(k + 1) - x_pred(k + 1));
        v_smth(k) = v_updt(k) + (A(k) ^ 2) * (v_smth(k + 1) - v_pred(k + 1));
        CW(k) = A(k) * v_smth(k + 1) + x_smth(k) * x_smth(k + 1);
        W(k) = v_smth(k) + x_smth(k) ^ 2;
    end

    if m == M, break; end

    ve(m + 1) = (sum(W(2:end)) + sum(W(1:end-1)) - 2 * sum(CW) + 0.5 * W(1)) / (K + 1);
    x0 = x_smth(1) / 2;
    if abs(ve(m + 1) - ve(m)) < opt.tol && abs(x0 - x0_prev) < opt.tol
        stop = 'converged';
        break;
    end

    x_pred(:) = 0;  v_pred(:) = 0;  x_updt(:) = 0;  v_updt(:) = 0;
    x_smth(2:end) = 0;  x_smth(1) = x0;  v_smth(:) = 0;  p_updt(:) = 0;
    A(:) = 0;  W(:) = 0;  CW(:) = 0;
end

res = struct();
res.n = n;
res.state = x_smth;
res.var = v_smth;
res.prob = 1 ./ (1 + exp(-(b0 + x_smth)));
res.state_lcl = norminv(0.025, x_smth, sqrt(v_smth));
res.state_ucl = norminv(0.975, x_smth, sqrt(v_smth));
res.certainty = 1 - normcdf(prctile(x_smth, 50) * ones(1, K), x_smth, sqrt(v_smth));
res.param_b0 = b0;
res.param_ve = ve(m);
res.iterations = m;
res.stop = stop;
end

function y = state_update(x_pred, v_pred, b0, n)
% Newton-Raphson for the posterior mode (filter_one_bin.m, get_state_update)
M = 50;
it = zeros(1, M);
it(1) = x_pred;
for i = 1:(M - 1)
    f = it(i) - x_pred - v_pred * (n - exp(b0 + it(i)) / (1 + exp(b0 + it(i))));
    df = 1 + v_pred * exp(b0 + it(i)) / ((1 + exp(b0 + it(i))) ^ 2);
    it(i + 1) = it(i) - f / df;
    if abs(it(i + 1) - it(i)) < 1e-14
        y = it(i + 1);
        return
    end
end
error('Newton-Raphson failed to converge.');
end
