function r = own_deconvolve(y, c, seed)
% One window, one random start: block coordinate descent (supplement, section D).
%
%   phase 1, c.phase1_iters times:  u by FOCUSS+ from all ones, then tau and yp0, then q
%   phase 2, until nothing moves:   u by GCV-FOCUSS+ from the last u, then tau and yp0,
%                                   then lambda1 by GCV and q
%
% 'faghih2014' is the same loop without the tonic and without the prior (own_config).
% r.history has one row per outer iteration, named in r.history_names.
warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:singularMatrix');
warning('off', 'MATLAB:rankDeficientMatrix');
rng(seed, 'twister');
started = tic;
y = y(:);
g = own_grid(numel(y), c);

tau = c.tau_lb + rand(2, 1) .* (c.tau_ub - c.tau_lb);
tau(1) = min(tau(1), tau(2) / c.min_tau_ratio);
yp0 = rand * max(y(1), 0);
if c.use_tonic
    C = own_tonic_basis(g, c);
    q = start_tonic(C, y, c);
else
    C = zeros(g.M, 0);
    q = zeros(0, 1);
end
nu = sum(y(2:end - 1) > y(1:end - 2) & y(2:end - 1) >= y(3:end)) + c.nu_extra;
lam1 = c.lambda1_init;
u = ones(g.N, 1);
lam2 = 0;

names = {'phase', 'tau_rise', 'tau_decay', 'yp0', 'impulses', 'lambda1', 'lambda2', 'fit', 'max_du', 'seconds'};
history = nan(c.phase1_iters + c.phase2_max, numel(names));
row = 0;

for j = 1:c.phase1_iters
    block_u_tau(ones(g.N, 1), false);
    update_tonic(false);
    log_row(1, nan);
end

converged = false;
for i = 1:c.phase2_max
    u_old = u;
    tau_old = tau;
    block_u_tau(u, true);
    update_tonic(true);
    log_row(2, max(abs(u - u_old)));
    if ~any(u)
        break
    end
    if all(abs(u - u_old) <= c.tol_u) && all(abs(tau - tau_old) <= c.tol_tau)
        converged = true;
        break
    end
end

[a, B] = own_matrices(tau, g, c);
r.tauj = tau;
r.y0 = yp0;
r.uj = u;
r.qj = q;
r.y = y;
r.y_phasic = a * yp0 + B * u;
r.y_tonic = C * q;
r.y_rec = r.y_phasic + r.y_tonic;
r.lambda1 = lam1;
r.lambda2 = lam2;
r.convergenceFlag = double(converged);
r.fit = 0.5 * sum((y - r.y_rec) .^ 2);
r.history = history(1:row, :);
r.history_names = names;
r.seconds = toc(started);

    % The nested functions below change u, lam1, lam2, tau, yp0 and q of the main function.
    function block_u_tau(from, use_gcv)
        target = y - C * q;
        [a, B] = own_matrices(tau, g, c);
        yt = target - a * yp0;
        if use_gcv
            [u, lam2] = own_gcv_focuss(from, yt, B, c);
        else
            [u, lam2] = own_focuss(from, yt, B, nu, c);
        end
        if any(u)
            [tau, yp0] = own_shape_step(tau, yp0, target, u, g, c);
        end
    end

    function update_tonic(use_gcv)
        if ~c.use_tonic
            return
        end
        [a, B] = own_matrices(tau, g, c);
        target = y - a * yp0 - B * u;
        if use_gcv
            lam1 = own_gcv_tonic(target, C, c);
        end
        q = own_tonic_step(target, y, C, lam1, q, c);
    end

    function log_row(phase, max_du)
        row = row + 1;
        [a, B] = own_matrices(tau, g, c);
        fit = 0.5 * sum((y - a * yp0 - B * u - C * q) .^ 2);
        history(row, :) = [phase, tau', yp0, nnz(u), lam1, lam2, fit, max_du, toc(started)];
    end
end

function q = start_tonic(C, y, c)
% Starting q.  'flat': a level at min(y).  'floor': a spline through the signal's rolling minimum.
switch c.tonic_init
    case 'flat'
        q = min(y) * ones(size(C, 2), 1) / c.tonic_scale;
    case 'floor'
        w = round(c.floor_window_s * c.fsy);
        q = C \ movmean(movmin(y, w), w);
    otherwise
        error('unknown tonic_init %s', c.tonic_init);
end
q = q + c.q_init_sd * randn(size(C, 2), 1) / c.tonic_scale;
end
