% Smoke test of the published Amin & Faghih (IEEE TBME 68:1726, 2021) code on its own data.
%
% Reproduces Auditory_Stimulation_Data_Experiment_with_Regularization.m for participant 11 only:
% same data, same window (200-400 s), same rates, bounds, peak distance, 8 random starts and
% the same selection rule (lowest residual among converged runs with tau_rise off its bounds).
% The authors' files are not edited; only paths are made portable and the result is summarised.
%
% Needs ss_8_14_18.mat (Google Drive link in the repository README) in the repository folder.
% Run from MATLAB:  run('<project folder>/src/tbme/tbme_smoke.m')

here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..', '..');
repo = fullfile(root, 'external', 'Tonic-Phasic-Decomposition-TBME');
out  = fullfile(root, 'results', 'tbme_smoke');
if ~exist(out, 'dir'), mkdir(out); end
addpath(fullfile(repo, 'Dependencies'));
% the README calls it ss_8_14_18.mat; the Google Drive download is named differently
mat = fullfile(repo, 'ss_8_14_18.mat');
if ~isfile(mat), mat = fullfile(repo, 'ss_8_14_18_auditory_stimulation.mat'); end
load(mat, 'ss');

subject = 11;
Fss = 100; Fsy = 2; Fsu = 4;
ub = [1.5 6]'; lb = [0.1 1.5]';
minimum_peak_distance = 1;
parallel_operations = 8;

y = downsample(ss(subject).sc(1).y, Fss/Fsy);
y = y(200*Fsy:400*Fsy-1); y = y(:);

data.y = y; data.ub = ub; data.lb = lb; data.Fsu = Fsu; data.Fsy = Fsy;
data.minimum_peak_distance = minimum_peak_distance;

tic
parfor i = 1:parallel_operations
    fprintf('start %d\n', i);
    results(i) = CoordinateDescentWithTonicPartRemoval_with_priors(data, i);
end
elapsed = toc;

% the authors' selection rule (their cost1)
best = []; cost_prev = Inf;
for i = 1:parallel_operations
    r = results(i);
    cost = 0.5 * norm(y - r.y_rec(:), 2)^2;
    tr = round(r.tauj(1)*1e4)/1e4;
    fprintf('start %d: tau %.3f / %.3f s, converged %d, %d impulses, cost %.4f\n', ...
            i, r.tauj(1), r.tauj(2), r.convergenceFlag, nnz(r.uj), cost);
    if cost < cost_prev && r.convergenceFlag == 1 && tr ~= lb(1) && tr ~= ub(1)
        best = r; cost_prev = cost;
    end
end

if isempty(best)
    fprintf('\nno start passed the selection rule\n');
else
    r2 = 1 - sum((y - best.y_rec(:)).^2) / sum((y - mean(y)).^2);
    fprintf(['\nselected: tau_rise %.3f s, tau_decay %.3f s | %d impulses in 200 s | ' ...
             'R2 %.3f | lambda1 %.3g, lambda2 %.3g\n'], best.tauj(1), best.tauj(2), ...
            nnz(best.uj), r2, best.lambda1, best.lambda2);
    fprintf('result fields: %s\n', strjoin(fieldnames(best)', ', '));

    ty = (0:numel(y)-1) / Fsy; tu = (0:numel(best.uj)-1) / Fsu;
    f = figure('Visible', 'off', 'Position', [0 0 1100 600]);
    subplot(2,1,1); plot(ty, y, ty, best.y_rec(:), '--', ty, best.y_tonic(:));
    legend('SC', 'reconstruction', 'tonic'); ylabel('\muS');
    title(sprintf('Participant %d, 200-400 s (published code)', subject));
    subplot(2,1,2); stem(tu(best.uj > 0), best.uj(best.uj > 0), 'Marker', 'none');
    ylabel('impulse'); xlabel('time (s)');
    exportgraphics(f, fullfile(out, 'tbme_smoke_s11.png'));
end
fprintf('total time for %d starts: %.0f s\n', parallel_operations, elapsed);
save(fullfile(out, 'result_s11.mat'), 'results', 'best', 'y', 'elapsed');
