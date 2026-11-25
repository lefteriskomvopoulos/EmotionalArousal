# Tracking sympathetic arousal from skin conductance during emotional film viewing

Komvopoulos Eleftherios (ek4538@nyu.edu) and Piyush More (pm4081@nyu.edu)
BE-GY 9603 / ECE-GY 9153 A, Neural and Physiological Signal Processing. September 30, 2026.

## Dataset

We use Study 6 of the POPANE database (Behnke et al., Scientific Data 9:10, 2022), available on OSF
under DOI 10.17605/OSF.IO/94BPX. It contains 187 healthy adults aged 18 to 38, 88 men and 99 women.
Skin conductance was recorded from finger electrodes at 1 kHz with an ADInstruments GSR amplifier.
ECG, blood pressure, and a continuous affect slider were recorded at the same time. Each session
began with a five-minute resting baseline, followed by six two-minute film clips played back to back.
Film order varied between participants. Each participant watched one of four sets: neutral films, a
negative set (fear, anger, disgust, sadness), a positive set (amusement, tenderness), or a mixed set
of three negative and three positive clips.

Seventeen participants had already been flagged for poor electrodermal data by the dataset authors.
Beyond those, we applied four criteria of our own, all automatic. Twenty-four we rejected for
electrode noise, meaning that more than 3% of their samples jumped by over 0.05 µS within a tenth of
a second. A skin conductance response normally rises over one to four seconds (Boucsein et al.,
Psychophysiology 49:1017, 2012), so a jump that fast cannot be sweat; it comes from the electrode
losing contact or from movement. Six participants had more than 5% missing samples, and four showed
almost no detectable responses at all. Finally, twenty-five were electrodermal non-responders, whose
median burst fell below 0.02 µS. We chose that threshold because the same guidelines treat anything
under 0.01 µS as no response, and 0.01 µS is also the smallest step these recordings can represent.
A participant whose typical burst is that small has no usable signal to deconvolve. Around one person
in ten is electrodermally non-responsive, and excluding them is accepted practice. We fixed all four criteria before
running any analysis. Only the amplitude threshold is derived from published guidance; the noise and
missing-data cut points are our own. The categories overlap, and 68 participants were
removed in total, leaving 119: neutral 29, negative 29, positive 32, mixed 29.

## Methods

Skin conductance can be described as a sparse train of neural impulses, convolved with the slow
response of the sweat gland, sitting on top of a drifting baseline and measurement noise. Our
analysis inverts that description in two stages: first we recover the impulses, then we estimate the
arousal that produced them.

**Preprocessing and deconvolution.** We downsample from 1 kHz to 10 Hz with an anti-aliasing filter,
which is ample for a response that takes about a second to rise. The drift is estimated as the third
percentile of the signal over a moving 30-second window, smoothed, then subtracted; bursts only push
conductance upward, so the lower edge of the signal follows the drift beneath them. A high-pass
filter would also strip the slow tail of every genuine burst and push parts of the signal below zero,
which the next stage cannot handle. The sweat-gland kernel is modelled as the difference of two
exponentials, one governing the rise and one the decay, with both constants fitted per participant.
We recover the impulse train by minimising squared reconstruction error plus an L1 penalty under a
non-negativity constraint, using FISTA: nerves can only fire, so impulses cannot be negative, and the
penalty keeps the solution sparse. The objective is that of cvxEDA (Greco et al., 2016), and the
alternation between recovering impulses and fitting system parameters follows Faghih et al. (2014).
Adjacent non-zero samples from the same burst are merged into one event, leaving a list of impulse
times and amplitudes: a marked point process.

Two hyperparameters required a decision. We swept the sparsity weight on the weakest and strongest
responders in each group and kept the value at which both fell within the physiological range of five
to fifteen impulses per minute. It is a single global value, applied identically to everyone with no
per-subject scaling, so amplitudes stay comparable across participants. The tonic window we set to 30
seconds; at 60 seconds the fitted decay constant came out at roughly twice published values, because
slow content left in the signal is absorbed into an artificially stretched kernel.

**State estimation.** We then apply the marked point process filter of Wickramasuriya and Faghih
(IEEE Access 8:68499, 2020). In this model a hidden arousal state drifts as a random walk, and it
drives two observations in each five-second bin: whether an impulse occurred, through a logistic
model, and how large that impulse was, through a Gaussian model on its log amplitude. The state is
estimated by a forward Bayesian filter, in which the update has no closed form and is therefore found
by Newton iteration, and then refined by a backward smoother so that every bin benefits from the
entire recording. The five model parameters are learned by expectation-maximization.

## Hypotheses

H1 is a sanity check that must hold: films raise arousal above the resting baseline. H2 is that
emotional films raise it more than neutral ones. Our primary hypothesis, H3, is that arousal reflects
intensity alone, so negative and positive films both exceed neutral but do not differ from each
other. H4 is the competing alternative: negative exceeds positive, tested within the same person in
the mixed group. The two predictions are mutually exclusive, so the mixed group can distinguish
between them. H5 is that emotions within a valence do not separate from one another. H6 is that the decoded state
tracks the participant's own affect slider.

To test these, each participant is reduced to a small set of per-block summaries: mean estimated
arousal, impulse rate, and mean impulse amplitude. We then compare film-minus-baseline change scores
between groups, which removes each person's own baseline level from the comparison. The mixed group
additionally allows a paired within-person test of valence. All three measures are reported together,
since an effect that appears in the estimated state but not in the raw impulse statistics would tell
us something about our estimator rather than about physiology.

![Figure 1. One two-minute film clip from participant 99 through the deconvolution. From the top: raw skin conductance with the estimated tonic level; the phasic signal after subtracting it; the recovered impulses, each with a time and an amplitude; and the reconstruction obtained by convolving those impulses back with the fitted kernel (R-squared 0.979 for this window).](../results/figures/study6_demo_P99.png)

![Figure 2. The full session for the same participant. From the top: the phasic signal; the recovered impulses; and the estimated arousal state with a one standard deviation band, with the affect slider overlaid in orange. Vertical lines mark block boundaries. The band widens wherever impulses become sparse, so the model reports its own uncertainty.](../results/figures/study6_state_P99.png)

## Preliminary work and what remains

The deconvolution already runs on the full cohort. Impulse rates across all 119 participants fall in
the physiological range, and the median decay constant is 6.6 seconds. The state estimator runs end
to end as well, producing a smooth trajectory whose uncertainty band widens wherever impulses become
sparse. For the participant shown here the estimated arousal is higher during the films than during
the resting baseline, which is the direction H1 predicts, though a single participant proves
nothing.

The estimator is not yet reliable. Expectation-maximization proved unstable on a single recording:
it first fitted a negative slope between arousal and burst size, which is physiologically impossible,
and then drove the amplitude noise to zero, which made the state follow every individual burst. We
constrained two parameters to stabilise it, and as a result the state currently tracks how often
bursts occur rather than how large they are. The binning is a second problem. At five seconds, about
a quarter of the bins contain more than one impulse, and only the largest is kept.

We will validate the estimator on synthetic data with a known arousal curve, and then revisit both
the constraints and the bin width.
Without such a test we cannot distinguish a failing estimator from a genuine null result. We will
also cross-check the deconvolution against cvxEDA, and replace the hand-chosen sparsity weight with
generalized cross-validation, as in Faghih et al. (2014). The full cohort will then be analysed with
mixed models, using participant as a random effect.
