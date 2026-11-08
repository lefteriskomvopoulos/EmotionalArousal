function u = own_irls_step(u, yt, B, lam, c)
% One FOCUSS+ update: u = P B' (B P B' + lam I)^-1 yt, P = diag(|u|^(2-p)), negatives set to 0.
%
% P is zero where u is zero, so those columns are skipped.  Zeros therefore stay zero, which is
% how the support shrinks.
S = find(u);
d = abs(u(S)) .^ (2 - c.p);
BS = B(:, S);
G = (BS .* d') * BS';
G(1:size(G, 1) + 1:end) = G(1:size(G, 1) + 1:end) + lam + c.jitter;
uS = d .* (BS' * (G \ yt));
uS(uS < 0) = 0;
u(:) = 0;
u(S) = uS;
end
