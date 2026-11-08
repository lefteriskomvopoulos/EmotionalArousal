function g = own_grid(M, c)
% Which u sample reaches which y sample, for a window of M signal samples.
%
% y_k is taken at k Ty and u_j at (j - 1) Tu, so u_1 sits one Ty before the first y sample
% (Amin & Faghih eq. 7).  lag(k, j) = k L - j + 1 counts Tu steps from u_j to y_k; 0 = not yet.
% The black box indexes one Ty differently (its u_1 sits at the first y sample): rebuilt from its
% saved tau and u with these matrices, its phasic part matched exactly with u moved by +2 samples,
% gain 1.000 (docs/EXPERIMENTS.md).  The physics is the same; own_run.m turns our indices into
% times, so events land at the same moments.
g.M = M;
g.L = c.u_per_y;
g.N = M * g.L;
g.Ty = 1 / c.fsy;
g.Tu = g.Ty / g.L;
[j, k] = meshgrid(1:g.N, 1:M);
g.lag = max(k * g.L - j + 1, 0);
g.k = (1:M)';
end
