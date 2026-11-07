function tbme_filter_probe()
% Does the authors' solver low-pass the signal it is given?  Their folder ships LowPassFilter.p
% but their scripts never call it, so one window of P53 is run with a logging stand-in ahead of
% it on the path, and the solver's stored signal is compared with its input.
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'Dependencies'));
addpath(fullfile(here, 'shadow'));                 % last added = first on the path
which('LowPassFilter')
S = load(fullfile(here, 'input', 'P53.mat'));
data = struct('y', S.y(1:400), 'ub', [1.5 6]', 'lb', [0.1 1.5]', 'Fsu', 4, 'Fsy', 2, ...
              'minimum_peak_distance', 1);
r = CoordinateDescentWithTonicPartRemoval_with_priors(data, 1);
fprintf('\nsolver output fields: %s\n', strjoin(fieldnames(r)', ' '));
if isfield(r, 'y')
    fprintf('largest |r.y - input| = %g uS\n', max(abs(r.y(:) - data.y(:))));
end
fprintf('probe finished: tau %.3f / %.3f, %d impulses\n', r.tauj(1), r.tauj(2), nnz(r.uj));
end
