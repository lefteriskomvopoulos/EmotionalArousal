function link = s2_tonic_calibrate(s, xhat, vhat)
% Variant tonic14c (docs/EXPERIMENTS.md Part 18): the tonic link calibrated from the pulse-only
% state (regression calibration).
%
%   s      1 x K continuous observation, NaN where not observed (from s2_tonic_input)
%   xhat   1 x K pulse-only state (the main analysis's chosen state)
%   vhat   1 x K its variance
%   link   [s0 s1 vs] for s2_filter_tonic's opt.fixed (the balanced floor is applied there)
%
% Least squares of s on xhat over the observed bins gives s0, s1.  vs is the mean squared
% residual minus s1^2 * mean(vhat), the part of the spread that is only the pulse state's own
% uncertainty; if that leaves vs <= 0, vs = 1e-12 and the floor takes over.

s = s(:);  xhat = xhat(:);  vhat = vhat(:);
ob = ~isnan(s);
X = [ones(sum(ob), 1) xhat(ob)];
b = X \ s(ob);
res = s(ob) - X * b;
vs = mean(res .^ 2) - b(2) ^ 2 * mean(vhat(ob));
if ~(vs > 0), vs = 1e-12; end
link = [b(1) b(2) vs];
end
