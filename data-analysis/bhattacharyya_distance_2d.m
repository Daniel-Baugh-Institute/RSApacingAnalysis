function D_B = bhattacharyya_distance_2d(PDF)
% Computes the Bhattacharyya distance between two 2D probability distributions
%
% Inputs:
%   PES - MxN matrix where the row is the time sample and the column is
%   the animal number
%
% Output:
%   D_B - Bhattacharyya distance

% Add path
addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))
[num_slices, num_subjects] = size(PDF);
D_B = zeros(num_slices-1,num_subjects);
for i = 1:num_subjects
    for j = 1:num_slices-1
        P = PDF(j,i).F_grid;
        Q = PDF(j+1,i).F_grid;
        % Ensure P and Q are valid probability distributions
        if any(P(:) < 0) || any(Q(:) < 0)
            error('Probability densities must be non-negative.');
        end

        % Normalize P and Q to sum to 1 (if they are not already)
        P = P / sum(P(:));
        Q = Q / sum(Q(:));

        % Compute the Bhattacharyya coefficient
        BC = sum(sqrt(P(:) .* Q(:)));

        % Compute the Bhattacharyya distance
        D_B(j,i) = -log(BC);
    end
end
end
