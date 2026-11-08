function [tau, yp0] = own_shape_step(tau, yp0, target, u, g, c)
% tau and yp0 by the interior-point method, with u and the tonic held fixed.
%
% target is y - C q.  Only the impulses that exist are used, so each evaluation is fast.
% yp0 is limited to the phasic room at the first sample, target(1); the paper uses y(1).
S = find(u);
uS = u(S);
top = max(target(1), 1e-6);
lb = [c.tau_lb; 0];
ub = [c.tau_ub; top];
Aineq = [c.min_tau_ratio, -1, 0];
weight = c.prior_lambda(:) ./ (2 * c.prior_sigma(:) .^ 2);
opts = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', 'off', ...
                    'MaxIterations', c.shape_iters);
x0 = [tau; min(yp0, top)];
x = fmincon(@cost, x0, Aineq, 0, [], [], lb, ub, [], opts);
tau = x(1:2);
yp0 = x(3);

    function J = cost(x)
        [a, BS] = own_matrices(x(1:2), g, c, S);
        r = target - a * x(3) - BS * uS;
        J = 0.5 * (r' * r);
        if c.use_prior
            J = J + sum(weight .* (x(1:2) - c.prior_mu(:)) .^ 2);
        end
    end
end
