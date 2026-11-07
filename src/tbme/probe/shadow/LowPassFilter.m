function y_filtered = LowPassFilter(y, Fs, wc, filter_order)
% Stand-in for the authors' encrypted Dependencies/LowPassFilter.p, placed ahead of it on the
% path: logs every call, then does what their open LowPassFilter.m does (Skin-Conductance-
% Deconvolution, Deconvolution_Code/LowPassFilter.m).  Used only by tbme_filter_probe.m.
fprintf('LowPassFilter called: %d samples, Fs %g, wc %g, order %g\n', numel(y), Fs, wc, filter_order);
y = y(:);
Fs = Fs/2;
rng default;
Fnorm = wc/(Fs/2);
df = designfilt('lowpassfir','FilterOrder',filter_order,'CutoffFrequency',Fnorm);
D = mean(grpdelay(df));
temp = filter(df,[y(1) * ones(D,1); y ; y(end) * ones(D,1)]);
temp = temp(2*D+1:end);
y_filtered = temp(:);
end
