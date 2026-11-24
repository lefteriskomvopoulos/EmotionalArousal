function R = s2_read_raw(f, vars)
% Selected columns of a raw POPANE Study 6 file (S6_P<id>_All.csv), all as double (an empty
% marker becomes NaN).  The header is the first line not starting with '#'; automatic detection
% fails on these files (docs/EXPERIMENTS.md Part 19).
%
%   R = s2_read_raw('data/raw/Study6/S6_P99_All.csv', {'timestamp', 'affect', 'marker'})

fid = fopen(f);
if fid < 0, error('cannot open %s', f); end
c = onCleanup(@() fclose(fid));
h = 0;  line = fgetl(fid);
while startsWith(line, '#'), h = h + 1;  line = fgetl(fid); end
names = strsplit(strtrim(line), ',');
o = delimitedTextImportOptions('NumVariables', numel(names), 'VariableNames', names, ...
    'DataLines', [h + 2 Inf], 'Delimiter', ',');
o = setvartype(o, names, 'double');
o.SelectedVariableNames = vars;
R = readtable(f, o);
end
