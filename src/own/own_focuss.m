function [u, lam] = own_focuss(u, yt, B, nu, c)
% FOCUSS+ (supplement, section F): sparsity is imposed by keeping at most nu impulses.
sep = round(c.sep_s * c.fsy * c.u_per_y);
ynorm = norm(yt);
lam = 0;
for r = 1:c.focuss_iters
    lam = (1 - min(norm(yt - B * u) / max(ynorm, realmin), 1)) * c.lambda2_max_focuss;
    u = own_irls_step(u, yt, B, lam, c);
    if r > c.focuss_iters / 2                 % after half the iterations
        u = own_prune(u, sep);
        if nnz(u) > nu
            [~, order] = sort(u, 'descend');
            u(order(nu + 1:end)) = 0;
        end
    end
end
end
