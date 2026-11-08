function own_check_blackbox(name, folder)
% Bug check against the black box's saved output.  It runs no solver.
%
%   own_check_blackbox('P53_pp', 'hpc/real/output_pp')
%
% For each window it rebuilds the black box's phasic part from its own tau and u, with our
% matrices, under both input models and small time shifts of u, and fits its tonic with our
% basis.  R2 near 1 means our model is the same as theirs.  The fitted "gain" gives the scale
% between the two u, and "best shift" any offset in time (samples of u).
c = own_config('amin2021');
S = load(fullfile(folder, [name '.mat']));
fprintf('%s: fields%s\n', name, sprintf(' %s', string(fieldnames(S.sel{1}))));
first = S.sel{1};
fprintf('sizes: y %d, uj %d, qj %d, y0_e %d, tauj %d\n', numel(first.y), numel(first.uj), ...
        numel(first.qj), numel(first.y0_e), numel(first.tauj));
models = {'zoh', 'impulse'};
for w = 1:numel(S.sel)
    r = S.sel{w};
    M = numel(r.y);
    if M ~= 400, continue, end
    g = own_grid(M, c);
    u = r.uj(:);
    p = r.y_phasic(:);
    fprintf('window %2d, tau %.3f/%.3f:', w, r.tauj(1), r.tauj(2));
    for m = 1:2
        c.input_model = models{m};
        [a, B] = own_matrices(r.tauj(1:2), g, c);
        best = -inf;
        for s = -4:4
            b = B * circshift(u, s);
            beta = [a, b] \ p;
            fit = 1 - sum((p - [a, b] * beta) .^ 2) / sum((p - mean(p)) .^ 2);
            if fit > best
                best = fit; shift = s; gain = beta(2);
            end
        end
        fprintf('  %s R2 %.5f gain %.3f best shift %+d', models{m}, best, gain, shift);
    end
    c.input_model = 'zoh';
    C = own_tonic_basis(g, c);
    t = r.y_tonic(:);
    fprintf('  tonic R2 %.5f\n', 1 - sum((t - C * (C \ t)) .^ 2) / sum((t - mean(t)) .^ 2));
end
end
