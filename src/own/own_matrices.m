function [a, B] = own_matrices(tau, g, c, support)
% a (M x 1) and B (M x N) of  y = a yp0 + B u + C q + noise  for tau = [tau_rise; tau_decay].
%
% The response obeys  tr td y'' + (tr + td) y' + y = u  (eq. 2), so h(t) has gain 1 at DC.
% support (optional) returns only those columns of B, for speed.
tr = tau(1);
td = tau(2);
a = exp(-g.k * g.Ty / td);
t = (0:g.N)' * g.Tu;
switch c.input_model
    case 'zoh'      % exact response to u held for one Tu: differences of the step response
        s = 1 - (td * exp(-t / td) - tr * exp(-t / tr)) / (td - tr);
        v = [0; diff(s)];
    case 'impulse'  % the paper's h(t) sampled at the lags
        v = (exp(-t / td) - exp(-t / tr)) / (td - tr);
        v(1) = 0;
    otherwise
        error('unknown input_model %s', c.input_model);
end
if nargin < 4
    B = v(g.lag + 1);
else
    B = v(g.lag(:, support) + 1);
end
end
