function [u, lam, ok] = own_gcv_focuss(u, yt, B, c)
% GCV-FOCUSS+ (supplement, section G): lambda2 is chosen by GCV at every iteration, so the
% number of impulses is no longer fixed.
sep = round(c.sep_s * c.fsy * c.u_per_y);
M = numel(yt);
hi = log10(c.lambda2_max_gcv);
opts = optimset('TolX', c.lambda_log_tol, 'Display', 'off');
lam = c.lambda2_init;
prev = inf;
ok = false;
for it = 1:c.gcv_max_iters
    u = own_irls_step(u, yt, B, lam, c);
    u(u < c.amp_threshold_uS / max(B(:))) = 0;
    if c.prune_in_gcv
        u = own_prune(u, sep);
    end
    S = find(u);
    if isempty(S)
        return
    end
    [R, Sig, ~] = svd(B(:, S) .* (abs(u(S))' .^ ((2 - c.p) / 2)), 'econ');
    gam = R' * yt;
    sig2 = diag(Sig) .^ 2;
    rest = max(sum(yt .^ 2) - sum(gam .^ 2), 0);     % the part with singular value 0
    lam = 10 ^ fminbnd(@(x) gcv(10 ^ x, gam, sig2, rest, M), hi - c.lambda_log_span, hi, opts);
    if it >= c.gcv_min_iters && abs(lam - prev) <= c.gcv_tol * lam
        ok = true;
        return
    end
    prev = lam;
end
end

function G = gcv(lam, gam, sig2, rest, M)
% Eq. 3 of the supplement, in the standard form: trace(I - H) squared below.
w = lam ./ (sig2 + lam);
G = M * (sum(gam .^ 2 .* w .^ 2) + rest) / (sum(w) + M - numel(sig2)) ^ 2;
end
