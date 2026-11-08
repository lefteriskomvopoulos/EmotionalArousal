function q = own_tonic_step(target, y, C, lam1, q, c)
% q by an interior-point QP: 1/2 ||target - C q||^2 + lam1 ||q||^2 subject to C q <= y.
%
% target is y - phasic.  If the solver fails, the previous q is kept.
P = size(C, 2);
H = C' * C + (2 * lam1 + c.jitter) * eye(P);
opts = optimoptions('quadprog', 'Algorithm', 'interior-point-convex', 'Display', 'off');
[new, ~, flag] = quadprog((H + H') / 2, -C' * target, C, y, [], [], [], [], [], opts);
if flag > 0 && all(isfinite(new))
    q = new;
end
end
