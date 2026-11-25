function [s, fopt] = s2_tonic_input(level, variant)
% The continuous observation and filter options of each tonic variant (docs/EXPERIMENTS.md
% Parts 15-17), from a tonic level in every 1 s bin.
%
%   tonic     (tonic14,  Part 15)  the level at bins 7, 21, 35, ...; no floor
%   tonic14b  (Part 16)            the same input; balanced floor
%   tonic14d  (Part 17)            the change between consecutive readings, at the midpoint bins
%                                  14, 28, 42, ...; balanced floor
%   tonic14c  (Part 18)            the tonic14d input and floor; fopt.calibrate = true: the
%                                  caller sets fopt.fixed from s2_tonic_calibrate
%
% level    1 x K tonic level (uS)
% s        1 x K continuous observation, NaN where not observed
% fopt     options for s2_filter_tonic

level = level(:)';
K = numel(level);
obs = 7:14:K;
s = nan(1, K);
switch variant
    case 'tonic'
        s(obs) = level(obs);
        fopt = struct('floor', false);
    case 'tonic14b'
        s(obs) = level(obs);
        fopt = struct('floor', true, 'gap', 14);
    case 'tonic14d'
        s(obs(1:end-1) + 7) = diff(level(obs));
        fopt = struct('floor', true, 'gap', 14);
    case 'tonic14c'
        s(obs(1:end-1) + 7) = diff(level(obs));
        fopt = struct('floor', true, 'gap', 14, 'calibrate', true);
    otherwise
        error('unknown variant %s', variant);
end
end
