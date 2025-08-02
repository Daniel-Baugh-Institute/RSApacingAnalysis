function [ hrv_frag, acceleration_segment_boundaries_3plus, alternation_segment_boundaries_4plus ] = hrv_fragmentation( nni, varargin )
%Computes HRV fragmentation indices [1]_ of a NN interval time series.
%
%:param nni: Vector of NN-interval dirations (in seconds)
%
%:returns: Table containing the following fragmentation metrics:
%
%    - PIP: Percentage of inflection points.
%    - IALS: Inverse average length of segments.
%    - PSS: Percentage of NN intervals that are in short segments.
%    - PAS: Percentage of NN intervals that are in alternation segments of at least 4 intervals.

%   acceleration_segment_boundaries_3plus: NN interval indices with
%   acceleration/deceleration segments that are three or more beats long
%
%   alternation_segment_boundaries_2plus: NN interval indices with
%   acceleration/deceleration segments that are two or more beats long
%
%.. [1] Costa, M. D., Davis, R. B., & Goldberger, A. L. (2017). Heart Rate
%   Fragmentation: A New Approach to the Analysis of Cardiac Interbeat Interval
%   Dynamics. Frontiers in Physiology, 8(May), 1–13.
%
%  Script modified to allow for calculation of cardiac efficiency for
%  alternation segments versus acceleration/deceleration segments

import mhrv.defaults.*;

%% Input

% Define input
p = inputParser;
p.addRequired('nni', @(x) ~isempty(x) && isvector(x));

% Get input
p.parse(nni, varargin{:});

%% Calculate fragmentation indices

% Number of NN intervals
N = length(nni);

% Reshape input into a row vector
nni = reshape(nni, [1, N]);

% delta NNi: differences of consecutinve NN intervals
dnni = diff(nni);

% Product of consecutive NN interval differences
ddnni = dnni(1:end-1) .* dnni(2:end);

% Logical vector of inflection point locations (zero crossings). Add a fake inflection points at the
% beginning and end so that we can count the first and last segments (i.e. we want these segments
% to be surrounded by inflection points like regular segments are).
ip = [-1, ddnni, -1] < 0;

% Number of inflection points (where detla NNi changes sign). Subtract 2 for the fake points we
% added.
nip = nnz(ip) - 2;

% Percentage of inflection points (PIP)
pip = nip / N;

% Indices of inflection points
ip_idx = find(ip);

% Length of acceleration/deceleration segments: the difference between inflection point indices
% is the length of the segments. This includes the first and last segment because of the fake points
% we added.
segment_lengths = diff(ip_idx);
disp('ip_idx')
ip_idx(1:3)

% Inverse Average Length of Segments (IALS)
ials = 1 / mean(segment_lengths);

% Number of NN intervals in segments with less than three intervals
short_segment_lengths = segment_lengths(segment_lengths < 3);
nss = sum(short_segment_lengths);

% Percentage of NN intervals that are in short segments (PSS)
pss = nss / N;

% An alternation segment is a segment of length 1
alternation_segment_boundaries = [1, segment_lengths > 1, 1];
alternation_segment_lengths = diff(find(alternation_segment_boundaries));

% Percentage of NN intervals in alternation segments length > 3 (PAS)
nas = sum(alternation_segment_lengths(alternation_segment_lengths > 3));
pas = nas / N;

% Added by MG
% acceleration_segment_boundaries_3plus  = NaN;
% alternation_segment_boundaries_4plus = NaN;
% Acceleration/deceleration segment boundaries (for segments with at least
% three intervals
acceleration_segment_boundaries_ip_idx = find(diff(ip_idx) > 1);
% gives the index of the first point in the difference
% Ex: if ip_idx = [1 5 9 10 15 16]; find(diff(ip_idx)>3) = [1     2     4]
acceleration_segment_boundaries_start = ip_idx(acceleration_segment_boundaries_ip_idx);
acceleration_segment_boundaries_end = ip_idx(acceleration_segment_boundaries_ip_idx + 1);
acceleration_segment_boundaries_3plus = [acceleration_segment_boundaries_start(:), acceleration_segment_boundaries_end(:)];


% Alternation segment boundaries (for segments with at least three
% consecutive alternations
d = diff(ip_idx);

% Identify where the difference is not 1
breaks = [0 find(d ~= 1) length(ip_idx)];

% Initialize result
alternation_segment_boundaries_4plus = [];

% This section was written using ChatGPT then modified
% Calculate RR differences and their sign
dRR = diff(nni);  % RR is your vector of RR intervals
sign_dRR = sign(dRR);  % +1 for increase, -1 for decrease, 0 for no change



i = 1;
nas_check = 0;
while i <= length(sign_dRR) - 2
    count = 1;  % start with one valid alternation
    while (i + count <= length(sign_dRR)) && ...
            (sign_dRR(i + count) ~= sign_dRR(i + count - 1))
        count = count + 1;
    end

    idx = 1:1:length(sign_dRR);

    if count >= 3
        % Map back to RR indices: use i to i+count
        start_idx = idx(i);  % index in RR of start of pattern
        try
            end_idx = idx(i + count);  % corresponds to RR(end_idx + 1)
        catch
            disp('End idx exceeds the number of array elements')
            i + count
            end_idx = idx(end);
        end
        alternation_segment_boundaries_4plus = [alternation_segment_boundaries_4plus; start_idx, end_idx];
        i = i + count + 1;  % skip this entire segment to avoid overlap
        nas_check = nas_check + count - 1;
    else
        i = i + 1;
    end

end

pas_check = nas_check/N;

if pas == pas_check
    disp('Passed PAS check')
else
    disp('Failed PAS check')
    pas_check
    pas
end



%% Create metrics table
hrv_frag = table;
hrv_frag.Properties.Description = 'Fragmentation HRV metrics';

hrv_frag.PIP = pip * 100;
hrv_frag.Properties.VariableUnits{'PIP'} = '%';
hrv_frag.Properties.VariableDescriptions{'PIP'} = 'Percentage of inflection points in the NN interval time series';

hrv_frag.IALS = ials;
hrv_frag.Properties.VariableUnits{'IALS'} = 'n.u.';
hrv_frag.Properties.VariableDescriptions{'IALS'} = 'Inverse average length of the acceleration/deceleration segments';

hrv_frag.PSS = pss * 100;
hrv_frag.Properties.VariableUnits{'PSS'} = '%';
hrv_frag.Properties.VariableDescriptions{'PSS'} = 'Percentage of short segments';

hrv_frag.PAS = pas * 100;
hrv_frag.Properties.VariableUnits{'PAS'} = '%';
hrv_frag.Properties.VariableDescriptions{'PAS'} = 'The percentage of NN intervals in alternation segments';
end

