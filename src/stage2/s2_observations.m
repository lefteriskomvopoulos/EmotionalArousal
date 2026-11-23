function obs = s2_observations(id, min_height, root)
% Stage 2's input for one real participant: one bin per second over the whole analysed segment.
%
% id          participant number (P<id>)
% min_height  pulse threshold in uS: 0.05 for the main analysis, 0.02 for the secondary check
%             (AGENTS.md rule 8)
% root        repository root (default: the current folder)
%
% Bins are 1 s wide and start at the segment's first sample, so they cover the baseline, the
% transition and the films, including stretches with no pulses.  The solver prunes pulses closer
% than 1 s (sep_s = 1, docs/EXPERIMENTS.md Part 12), so a bin almost never holds two.  (At 5 s
% bins, as in the proposal, 37% of occupied bins held more than one and only the largest was
% kept.)  Exceptions: two pairs 0.25 s apart, one 4 Hz sample, in the same window (P60 at
% 1042.1 s, P80 at 376.0 s; Part 12's "none within 1 s" missed them).  Adjacent samples are one
% burst (proposal, Methods), so a bin with two pulses gets one pulse whose height is the sum;
% obs.merged counts them.
%
% obs.t        bin centres on the recording clock (s)
% obs.n        0/1 pulse in the bin
% obs.height   pulse height in uS (NaN without a pulse)
% obs.r        log(height), the size the MPP filter models (proposal: Gaussian on log size)
% obs.marker   block code at the bin centre: -1 baseline, 1xx/2xx/3xx neutral/negative/positive
%              film clip, NaN in the transition
% obs.group    film group

if nargin < 2, min_height = 0.05; end
if nargin < 3, root = pwd; end

sig = readtable(fullfile(root, 'hpc', 'own', 'final', 'output', 'own4_p1000_k14_f3_s8', ...
    sprintf('P%d_own4_signals.csv', id)));
pul = readtable(fullfile(root, 'data', 'real50_pulses.csv'), 'TextType', 'string');
pul = pul(pul.id == id, :);

t0 = sig.time_s(1);
K = floor(sig.time_s(end) - t0);
t = t0 + (0:K-1) + 0.5;

n = zeros(1, K);
height = nan(1, K);
keep = pul.height_uS >= min_height & pul.time_s >= t0 & pul.time_s < t0 + K;
% 1e-6 s guards pulses exactly 1 s apart against rounding into the same bin
k = floor(pul.time_s(keep) - t0 + 1e-6) + 1;
h = accumarray(k, pul.height_uS(keep), [K 1])';
n(h > 0) = 1;
height(h > 0) = h(h > 0);
merged = numel(k) - numel(unique(k));

obs.id = id;
obs.group = '';
if any(keep), obs.group = char(pul.group(find(keep, 1))); end
obs.min_height = min_height;
obs.t = t;
obs.n = n;
obs.height = height;
obs.r = log(height);
obs.marker = interp1(sig.time_s, sig.marker, t, 'nearest', 'extrap');
obs.merged = merged;
end
