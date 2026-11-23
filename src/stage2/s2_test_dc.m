function s2_test_dc(dc_dir)
% Check that s2_filter_mpp / s2_filter_bin compute exactly what the Data Challenge 2 filters
% computed: same input, the Data Challenge's starting values, against its saved results.
%
% dc_dir  the Data Challenge's State_Estimation_Code folder (dataset in ../State_Estimation_Dataset
%         and saved results in results/).  The data are course material and not in this repository.
%
%   addpath src/stage2; s2_test_dc('C:/.../Data Challenge 2/Final_Submission/State_Estimation_Code')
%
% Cases: Problem 1 (Subject 1, MPP, converged), Problem 2A (Subject 2, MPP, stopped on r1 <= 0),
% Problem 2B (Subject 2, binary), Problem 6 (Subject 8, binary; Subject 10, MPP).

data_dir = fullfile(dc_dir, '..', '..', 'State_Estimation_Dataset');
if ~exist(data_dir, 'dir'), data_dir = fullfile(dc_dir, '..', 'State_Estimation_Dataset'); end
cases = {
    'Problem1_results.mat',          1,  'mpp'
    'Problem2A_results.mat',         2,  'mpp'
    'Problem2B_results.mat',         2,  'bin'
    'Problem6_Subject8_results.mat', 8,  'bin'
    'Problem6_Subject10_results.mat', 10, 'mpp'};

worst = 0;
for c = 1:size(cases, 1)
    saved = load(fullfile(dc_dir, 'results', cases{c, 1}));
    saved = saved.results;
    d = load(fullfile(data_dir, sprintf('Subject%d.mat', cases{c, 2})));
    u = d.mpp(:)';
    n = double(u > 0);
    if strcmp(cases{c, 3}, 'mpp')
        res = s2_filter_mpp(n, u, struct('init', 'dc'));
        dp = max(abs([res.param_ve res.param_r0 res.param_r1 res.param_vr] - ...
                      [saved.param_ve saved.param_r0 saved.param_r1 saved.param_vr]));
    else
        res = s2_filter_bin(n);
        dp = abs(res.param_ve - saved.param_ve);
    end
    dx = max(abs(res.state(:) - saved.state(:)));
    dv = max(abs(res.var(:) - saved.var(:)));
    worst = max([worst dx dv dp]);
    fprintf('%-32s Subject %2d %s  stop %-9s  max|dstate| %.1e  max|dvar| %.1e  max|dparam| %.1e\n', ...
        cases{c, 1}, cases{c, 2}, cases{c, 3}, res.stop, dx, dv, dp);
end
if worst < 1e-9
    fprintf('PASS: identical to the Data Challenge results (worst difference %.1e)\n', worst);
else
    error('FAIL: differs from the Data Challenge results (worst difference %.1e)', worst);
end
end
