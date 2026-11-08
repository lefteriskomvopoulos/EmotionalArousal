function own_test()
% Tests of our own code on made-up data, about 2-5 minutes on a laptop.  No real data is used.
%
%   addpath src/own; own_test
fails = 0;
c = own_config('amin2021');
c.n_starts = 1;
g = own_grid(400, c);
tau = [0.7; 3.0];
[a, B] = own_matrices(tau, g, c);

% 1. a constant input settles at its own value: gain 1 at DC
fails = fails + check('DC gain of B', abs(sum(B(end, :)) - 1) < 1e-6);

% 2. one held impulse, against integrating the differential equation
j = 41;
f = @(t, x) [x(2); ((abs(t - (j - 0.5) * g.Tu) <= g.Tu / 2) - (tau(1) + tau(2)) * x(2) - x(1)) / (tau(1) * tau(2))];
T = g.k * g.Ty;
[~, x] = ode45(f, [0; T], [0; 0], odeset('RelTol', 1e-9, 'AbsTol', 1e-10, 'MaxStep', 0.05));
fails = fails + check('B against the ODE', max(abs(x(2:end, 1) - B(:, j))) < 1e-5);

% 3. the initial value decays with tau_decay
fails = fails + check('a', abs(a(2) / a(1) - exp(-g.Ty / tau(2))) < 1e-12);

% 4. tonic basis
C = own_tonic_basis(g, c);
fails = fails + check('tonic rows sum to 1', max(abs(sum(C, 2) - 1)) < 1e-12);
fprintf('      tonic columns: %d (paper 39 before dropping empty ones)\n', size(C, 2));

% 5. pruning keeps the larger of two close impulses
u = zeros(20, 1); u([5 6 12]) = [1 2 3];
fails = fails + check('prune', isequal(find(own_prune(u, 4)), [6; 12]));

% 6. known tau, known impulses, no tonic: u alone
rng(3);
truth = zeros(g.N, 1);
truth(40:60:760) = 2 + 6 * rand(13, 1);
y = B * truth + 0.003 * randn(g.M, 1);
peaks = sum(y(2:end - 1) > y(1:end - 2) & y(2:end - 1) >= y(3:end));
[u, ~] = own_focuss(ones(g.N, 1), y, B, peaks + c.nu_extra, c);
[u, lam, ok] = own_gcv_focuss(u, y, B, c);
hits = score(truth, u, g);
fprintf('      u alone: %d of %d found, %d returned, lambda2 %.2e, converged %d\n', hits, nnz(truth), nnz(u), lam, ok);
fails = fails + check('u alone', hits >= 12 && nnz(u) <= 18);

% 7. the whole loop on a made-up window, for both algorithms
tonic = 2 + 0.5 * sin(g.k * g.Ty / 40);
data = B * truth + tonic + 0.003 * randn(g.M, 1);
r = own_deconvolve(data, c, 1);
report('amin2021', r, truth, g, tau);
fails = fails + check('amin2021 fit', 1 - sum((data - r.y_rec) .^ 2) / sum((data - mean(data)) .^ 2) > 0.95);

c = own_config('faghih2014');
phasic = B * truth + 0.003 * randn(g.M, 1);
r = own_deconvolve(phasic, c, 1);
report('faghih2014', r, truth, g, tau);
fails = fails + check('faghih2014 fit', 1 - sum((phasic - r.y_rec) .^ 2) / sum((phasic - mean(phasic)) .^ 2) > 0.95);

fprintf('\n%s\n', ternary(fails == 0, 'ALL PASSED', sprintf('%d FAILED', fails)));
end

function bad = check(name, ok)
fprintf('%-6s %s\n', ternary(ok, 'PASS', 'FAIL'), name);
bad = ~ok;
end

function hits = score(truth, u, g)
% true impulses with a returned one within 1 s
true_j = find(truth);
est_j = find(u);
hits = 0;
for j = true_j'
    hits = hits + any(abs(est_j - j) * g.Tu <= 1);
end
end

function report(name, r, truth, g, tau)
fprintf('      %s: tau %.2f/%.2f (true %.2f/%.2f), %d of %d found, %d returned, converged %d, %.0f s\n', ...
        name, r.tauj(1), r.tauj(2), tau(1), tau(2), score(truth, r.uj, g), nnz(truth), nnz(r.uj), ...
        r.convergenceFlag, r.seconds);
end

function s = ternary(ok, a, b)
if ok, s = a; else, s = b; end
end
