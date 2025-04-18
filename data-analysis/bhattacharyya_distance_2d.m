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

% Plot HF and control Bhattacharya distance
HFidx = [15 19 ];%[1:4 10 13 15 17 19 21 23 25 27 29 31 34];
RSApacedIdx = [11 14 16 18 20 22];
MonoPacedIdx = [24 26 28 30 32 35];
ctrlIdx = 5:6;%5:9;
timevec = 0:30:(num_slices - 2)*30;

hold on
legend_entries = [];
legend_labels = {};

for i = 1:num_subjects
    if ismember(i, HFidx)
        color = 'r'; % HF
        label = 'Heart failure';
    % elseif ismember(i, RSApacedIdx)
    %     color = 'c'; % RSA paced
    %     label = 'RSA paced';
    % elseif ismember(i, MonoPacedIdx)
    %     color = 'm'; % Mono Paced
    %     label = 'Mono paced';
    elseif ismember(i, ctrlIdx)
        color = 'b'; % Control
        label = 'Control';
    % elseif i == 34
    %     color = 'g';
    %     label = 'Mono off pace';
    % elseif i == 12
        % color = 'k';
        % label = 'RSA off pace';
    else
        disp('Error index mismatch with HF/control label')
        continue
    end

    % Plot line
    if ismember(i,HFidx) || ismember(i,ctrlIdx)
        h = plot(timevec, D_B(:, i), 'o-','Color', color);
    end

    % Save handle and label only once per condition
    if ~any(strcmp(legend_labels, label))
        legend_entries(end+1) = h;
        legend_labels{end+1} = label;
    end
end

xlabel('Time (min)')
ylabel('Bhattacharyya distance')
legend(legend_entries, legend_labels)
saveas(gcf, 'plot_Bhattacharyya.png')

end
