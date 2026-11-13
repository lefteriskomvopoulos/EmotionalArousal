function c = own_config(algorithm)
% Every setting of our own deconvolution, in one place.
%
% algorithm  'amin2021'    Amin & Faghih, IEEE TBME 68:1726, 2021: tonic spline + priors on tau
%            'faghih2014'  Faghih et al., PLOS ONE 9:e85204, 2014 (the Data Challenge algorithm):
%                          no tonic and no priors, so its input must already be phasic
%
% The values are starting points, not targets: they are tuned on synthetic tuning sessions, and
% every run passes the tuned ones as arguments.  They are not changed here, so earlier rounds stay
% reproducible.  The frozen setting (after round 5) is in src/own/make_own_final_bundle.sh: own4
% input (4 Hz, u_per_y 1), knots every 14 s, prior 2e-4, lambda2 limits 2e-3 / 2e-4, lambda1 2e-4;
% round 6 (make_own_bundle.sh) tested the scale (y_gain) and froze the setting again with 8 starts:
% own4_p1000_k14_f3_s8, i.e. the above plus 'n_starts',8, in uS (y_gain 1), lambda2 limit 2e-4.
% "paper" = stated in the paper.  "DC" = Data Challenge code.  "black box" = read from its output.
% "ours" = the paper is silent or the value is our choice.

if nargin < 1, algorithm = 'amin2021'; end
c.algorithm = algorithm;

% Rates and windows
c.fsy = 2;                  % Hz, signal rate (paper); own_run reads it from the input file
c.u_per_y = 2;              % impulses per signal sample: 4 Hz u on a 2 Hz y (paper)
c.window_s = 200;           % s (authors' scripts)
c.n_starts = 4;             % random starts per window (ours; the paper uses 8)
c.y_gain = 1;               % the solver sees y x y_gain; settings in uS and every output are
                            % scaled back, so only the penalties see the change (own_run).
                            % 1 = plain uS.  Round 6: s_ref / the session's typical SCR size

% Response model
c.input_model = 'zoh';      % 'zoh': u held over one u sample, as in DC.  Rebuilds the black box's
                            % phasic part exactly (R2 1.00000, gain 1.000 on P53_pp).
                            % 'impulse': the paper's printed h(t), gain 0.25, R2 0.99
c.tau_lb = [0.1; 1.5];      % s, tau_rise, tau_decay (paper)
c.tau_ub = [2.0; 14.0];     % s (paper: 1.5, 6).  Typical SCR rise 1-3 s and half-recovery 2-10 s;
                            % 6 s would allow a half-recovery of only about 4.4 s
c.min_tau_ratio = 1.1;      % tau_decay >= this x tau_rise.  Keeps h(t) defined (ours).  DC uses 4

% Prior on tau: lambda / (2 sigma^2) x (tau - mu)^2
c.use_prior = true;
c.prior_mu = [0.650571; 2.77325];      % s (paper)
c.prior_sigma = [0.212443; 0.521739];  % s (paper)
c.prior_lambda = [0.1; 0.1];           % lambda3, lambda4 (paper).  First thing to tune

% Tonic: a cubic B-spline with knots every knot_s
c.use_tonic = true;
c.knot_s = 6;               % s (paper: the longest decay it allows).  Tested at 6, 10, 14 and
                            % 20 s; 14 kept (rounds 1 and 5)
c.extra_basis = 5;          % P = ceil(T / knot_s) + 5 (paper); empty columns are then dropped
c.tonic_scale = 1;          % 1 keeps q in uS (ours).  The paper uses 100 on its own scaling
c.tonic_init = 'floor';     % starting q: 'floor' under the signal, or 'flat' at min(y) (ours).
                            % With 'flat', pulses fill in for the tonic and stay: 40 returned for
                            % 13 true, unconverged.  With 'floor', 10-11 returned, converged
c.floor_window_s = 30;      % s, rolling minimum for 'floor' (ours)
c.q_init_sd = 0.02;         % uS, random spread of the starting q (ours, paper: 0.02)
c.lambda1_init = 0;         % tonic penalty during phase 1, before GCV (ours)
c.lambda1_max = 1e-4;       % upper limit of the GCV search (paper)

% Sparsity
c.p = 0.5;                  % ||u||_p^p (DC; the paper leaves it open, 0 < p <= 2)
c.lambda2_max_focuss = 0.01;% FOCUSS+ lambda_max (not in the paper).  DC used 1 on cortisol, whose
                            % matrix is far larger than ours in uS; this keeps the same proportion
c.lambda2_init = 1e-6;      % starting lambda of GCV-FOCUSS+ (DC)
c.lambda2_max_gcv = 1e-4;   % upper limit of the GCV search (paper)
c.lambda_log_span = 8;      % GCV searches log10(lambda) over this many decades below the limit
c.lambda_log_tol = 0.01;    % decades.  fminbnd's default tolerance (1e-4) is as wide as the
                            % limit, so a plain search on lambda would stop at once (ours)
c.nu_extra = 20;            % FOCUSS+ keeps at most (peaks in y) + 20 impulses (paper)
c.sep_s = 1;                % impulses closer than this are pruned to the largest (paper)
c.amp_threshold_uS = 0.02;  % pulses whose response would peak below this are zeroed in GCV-FOCUSS+.
                            % The project's SCR minimum, as a height, so it means the same at any
                            % tau.  About 0.35 in u at the prior's tau (black box 0.25, paper 3)
c.prune_in_gcv = false;     % also apply sep_s in GCV-FOCUSS+ (ours, a tuning option)
c.jitter = 1e-10;           % added to matrix diagonals for numerical safety (DC)

% Iterations
c.focuss_iters = 30;        % FOCUSS+ iterations per call (DC)
c.phase1_iters = 30;        % outer iterations with FOCUSS+ (paper)
c.phase2_max = 300;         % outer iterations with GCV-FOCUSS+, at most (ours)
c.gcv_min_iters = 10;       % GCV-FOCUSS+ iterations, at least (DC)
c.gcv_max_iters = 100;      % and at most (ours; DC 500)
c.gcv_tol = 0.05;           % relative change of lambda2 that ends GCV-FOCUSS+ (ours); must not
                            % be finer than lambda_log_tol, or it can never be met
c.tol_u = 1e-2;             % phase 2 ends when no u moves more than this (DC: 2 decimals)
c.tol_tau = 1e-4;           % and no tau moves more than this, s (DC: 4 decimals)
c.shape_iters = 50;         % interior-point iterations per shape step (ours)

switch algorithm
    case 'amin2021'
    case 'faghih2014'
        c.use_tonic = false;
        c.use_prior = false;
        c.prior_lambda = [0; 0];
    otherwise
        error('unknown algorithm %s', algorithm);
end
end
