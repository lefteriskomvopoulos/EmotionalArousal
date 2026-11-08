function lam1 = own_gcv_tonic(target, C, c)
% lambda1 by GCV, eq. 11.  The paper prints H = C (C'C + lam I) C'; the inverse is missing.
%
% With the penalty lam1 ||q||^2 the influence matrix uses 2 lam1.
M = numel(target);
[U, S, ~] = svd(C, 'econ');
s2 = diag(S) .^ 2;
g = U' * target;
rest = max(sum(target .^ 2) - sum(g .^ 2), 0);
hi = log10(c.lambda1_max);
opts = optimset('TolX', c.lambda_log_tol, 'Display', 'off');
lam1 = 10 ^ fminbnd(@(x) gcv(10 ^ x), hi - c.lambda_log_span, hi, opts);

    function G = gcv(lam)
        w = 2 * lam ./ (s2 + 2 * lam);
        G = M * (sum(g .^ 2 .* w .^ 2) + rest) / (M - sum(1 - w)) ^ 2;
    end
end
