function C = own_tonic_basis(g, c)
% C (M x P) of the tonic  C q: cubic B-splines with knots every c.knot_s (supplement, section B).
%
% Rows sum to c.tonic_scale inside the window, so q reads as a level in uS when the scale is 1.
% P = ceil(T / knot) + 5 as in the paper (39 for 200 s); basis functions that never reach the
% window are dropped.
T = g.M * g.Ty;
P = ceil(T / c.knot_s) + c.extra_basis;
centres = ((1:P) - 3) * c.knot_s;
s = abs((g.k * g.Ty - centres) / c.knot_s);
C = zeros(size(s));
inner = s < 1;
outer = s >= 1 & s < 2;
C(inner) = 2 / 3 - s(inner) .^ 2 + s(inner) .^ 3 / 2;
C(outer) = (2 - s(outer)) .^ 3 / 6;
C = c.tonic_scale * C(:, any(C, 1));
end
