function [meanP,minP,medianP,stdP,meanTE] = calcTE(PES)
% Calculate transition energy
addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))
[num_slices,num_subjects] = size(PES);

% Preallocate
meanP = zeros(num_slices,num_subjects);
minP = zeros(num_slices,num_subjects);
medianP = zeros(num_slices,num_subjects);
stdP = zeros(num_slices,num_subjects);
meanTE = zeros(num_slices,num_subjects);

for i = 1:num_subjects
    for j = 1:num_slices
    disp('subject number')
    disp(i)
    F_grid = PES(j,i).F_grid;

    % Check for inf of -inf vlaues from taking -ln(0) to get PES from PDF
    F_grid(isinf(F_grid)) = 0;

    % Find location and height of local maxima
    [TF1, P1] = islocalmin(F_grid,1); % only finds maxima along one dimension
    [TF2, P2] = islocalmin(F_grid,2);
    TF = TF1 & TF2;
    heights = F_grid(TF);

    %% Find energy barrier to move between minima using peak prominence
    P1_filt = -P1.*TF;
    P2_filt = -P2.*TF;
    P = min(P1_filt,P2_filt,'omitnan');
    % TODO: split HF and control
    meanP(j,i) = mean(nonzeros(P));
    minP(j,i) = min(min(P));
    medianP(j,i) = median(nonzeros(P));
    stdP(j,i) = std(nonzeros(P));

    % Find saddlepoints
    s = saddle(F_grid);
    saddleHeights = F_grid(s);

    % Calculate transition energy (difference between local min and saddle
    % point)
    n_min = sum(TF,'all');
    numCombos = n_min*(n_min-1)/2;

    if sum(s,'all') == 0 % no saddle points
        disp('No saddle points found for patient')
        disp(num2str(i))
        properties(j,i).TE = min(min(P));
        meanTE(j,i) = min(min(P));
    else
        temp = [];
        for jj = 1:length(heights)
            % Differnece between each minimum and each saddle point?
            for ii = 1:length(saddleHeights)
                temp = [temp heights(jj) - saddleHeights(ii)];
            end
        end
        properties(j,i).TE = temp;
        meanTE(j,i) = mean(temp);
    end
    end
end

% stats to compare HF and control
HF_idx = 1:5;
ctrl_idx = 6:9;

% compare HR, BP, CO, CoBF

disp('Mean prominence')
meanP_HF = meanP(:,HF_idx);
meanP_ctrl = meanP(:,ctrl_idx);
disp('Mean and std HF:')
mean(meanP_HF,'omitmissing')
meanP_HF = rmmissing(meanP_HF);
std(meanP_HF(:))
disp('Mean and std control:')
mean(meanP_ctrl,'omitmissing')
meanP_ctrl = rmmissing(meanP_ctrl);
std(meanP_ctrl(:))
[h,p,ci,stats] = ttest2(meanP_HF(:),meanP_ctrl(:))

disp('Median prominence')
medianP_HF = medianP(:,HF_idx);
medianP_ctrl = medianP(:,ctrl_idx);
disp('Mean and std HF:')
mean(medianP_HF(:),'omitmissing')
medianP_HF = rmmissing(medianP_HF);
std(medianP_HF(:))
disp('Mean and std control:')
mean(medianP_ctrl(:),'omitmissing')
medianP_ctrl = rmmissing(medianP_ctrl);
std(medianP_ctrl(:))
[h,p,ci,stats] = ttest2(medianP_HF(:),medianP_ctrl(:))

disp('Min prominence')
minP_HF = minP(:,HF_idx);
minP_ctrl = minP(:,ctrl_idx);
disp('Mean and std HF:')
mean(minP_HF(:),'omitmissing')
minP_HF = rmmissing(minP_HF);
std(minP_HF(:))
disp('Mean and std control:')
minP_ctrl = rmmissing(minP_ctrl);
mean(minP_ctrl(:))
std(minP_ctrl(:))
[h,p,ci,stats] = ttest2(minP_HF(:),minP_ctrl(:))

disp('Standard deviation of prominence')
stdP_HF = stdP(:,HF_idx);
stdP_ctrl = stdP(:,ctrl_idx);
disp('Mean and std HF:')
stdP_HF = rmmissing(stdP_HF);
mean(stdP_HF(:))
std(stdP_HF(:))
disp('Mean and std control:')
stdP_ctrl = rmmissing(stdP_ctrl);
mean(stdP_ctrl(:))
std(stdP_ctrl(:))
[~,p,ci,stats] = ttest2(stdP_HF(:),stdP_ctrl(:))

disp('Mean transition energy')
meanTE_HF = meanTE(:,HF_idx);
meanTE_ctrl = meanTE(:,ctrl_idx);
disp('Mean and std HF:')
meanTE_HF = rmmissing(meanTE_HF);
mean(meanTE_HF(:))
std(meanTE_HF(:))
disp('Mean and std control:')
meanTE_ctrl = rmmissing(meanTE_ctrl);
mean(meanTE_ctrl(:))
std(meanTE_ctrl(:))
[h,p,ci,stats] = ttest2(meanTE_HF(:),meanTE_ctrl(:))
end