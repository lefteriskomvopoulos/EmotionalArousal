function s2_plot(id, root, out, S, P)
% One participant through stage 2, in the layout of the Data Challenge plots: phasic signal,
% pulses, the chosen state with its 95% band, pulse probability.  Grey = baseline; dotted lines =
% film clip boundaries.  Writes results/figures/stage2_state_P<id>.png, or appends a page to `out`
% when it is a .pdf (s2_plot_all).  S and P (the state and parameter tables) may be passed in so
% that a loop reads them once.
%
%   addpath src/stage2; s2_plot(99)

if nargin < 2 || isempty(root), root = pwd; end
if nargin < 3 || isempty(out)
    out = fullfile(root, 'results', 'figures', sprintf('stage2_state_P%d.png', id));
end
sig = readtable(fullfile(root, 'hpc', 'own', 'final', 'output', 'own4_p1000_k14_f3_s8', ...
    sprintf('P%d_own4_signals.csv', id)));
if nargin < 4
    S = readtable(fullfile(root, 'data', 'stage2_states_005.csv'), 'TextType', 'string');
end
S = S(S.id == id, :);
if nargin < 5
    P = readtable(fullfile(root, 'data', 'stage2_params.csv'), 'TextType', 'string');
end
P = P(P.id == id & P.min_height_uS == 0.05, :);

t = S.t_s;
mk = S.marker;
base = mk == -1;
edges = t(find(diff(mk) ~= 0 & ~(isnan(mk(1:end-1)) & isnan(mk(2:end)))) + 1);

fig = figure('Visible', 'off', 'Position', [100 100 1000 900]);
tl = tiledlayout(4, 1, 'TileSpacing', 'compact');

ax(1) = nexttile;
plot(sig.time_s, sig.phasic, 'Color', [0.1 0.4 0.8]);
ylabel({'(a) phasic', '\muS'});

ax(2) = nexttile;
p = S.n == 1;
stem(t(p), S.height_uS(p), 'k', 'filled', 'MarkerSize', 2);
ylabel({'(b) pulses', 'height \muS'});

ax(3) = nexttile;
fill([t; flipud(t)], [S.state_lcl; flipud(S.state_ucl)], [0.5 1 1], 'EdgeColor', 'none'); hold on;
plot(t, S.state, 'Color', [0 0.45 0.9], 'LineWidth', 1.2);
ylabel({'(c) state x_k', sprintf('(%s filter)', upper(P.chosen))});

ax(4) = nexttile;
plot(t, S.prob, 'Color', [0.85 0 0.2], 'LineWidth', 1.2);
ylabel({'(d) probability', 'p_k'});
xlabel('time (s)');

for a = ax
    yl = ylim(a);
    hold(a, 'on');
    patch(a, [t(find(base, 1)) t(find(base, 1, 'last')) t(find(base, 1, 'last')) t(find(base, 1))], ...
        [yl(1) yl(1) yl(2) yl(2)], [0.6 0.6 0.6], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    for e = edges', xline(a, e, ':', 'Color', [0.4 0.4 0.4]); end
    ylim(a, yl); xlim(a, [t(1) t(end)]); grid(a, 'on');
end
linkaxes(ax, 'x');
if P.chosen == "mpp"
    how = sprintf('MPP converged, r_1 %.2f, v_e %.1e', P.mpp_r1, P.mpp_ve);
else
    how = sprintf('MPP stopped (%s), binary state shown, v_e %.1e', P.mpp_stop, P.bin_ve);
end
title(tl, {sprintf('P%d (%s group): stage 2, %d pulses \\geq 0.05 \\muS in 1 s bins', ...
    id, P.group, P.pulses), how});
if endsWith(out, '.pdf')
    exportgraphics(fig, out, 'ContentType', 'image', 'Resolution', 110, 'Append', true);
else
    exportgraphics(fig, out, 'Resolution', 150);
end
close(fig);
end
