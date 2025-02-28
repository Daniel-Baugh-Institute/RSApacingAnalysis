function s = saddle(M)
% Function from https://www.mathworks.com/matlabcentral/answers/545807-write-a-function-called-saddle-in-the-input-matrix-m-the-function-should-return-with-exactly-two-c
% Find saddlepoints
% Create logical vector that are true for each saddle condition separately
[TF1, P1] = islocalmin(M,1); % only finds maxima along one dimension
[TF2, P2] = islocalmax(M,2); % only finds maxima along one dimension
TFA = TF1 & TF2;
% Find locations for flipped rows/cols
[TF1, P1] = islocalmax(M,1); % only finds maxima along one dimension
[TF2, P2] = islocalmin(M,2); % only finds maxima along one dimension
TFB = TF1 & TF2;
% Return indices of saddle points
s = TFA | TFB;
end
