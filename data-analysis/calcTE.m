function [meanP,maxP,medianP,stdP,meanTE] = calcTE(PES)
% Calculate transition energy
addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))
[num_slices,num_subjects] = size(PES);

% Preallocate
meanP = zeros(num_slices,num_subjects);
maxP = zeros(num_slices,num_subjects);
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
    [TF1, P1] = islocalmax(F_grid,1); % only finds maxima along one dimension
    [TF2, P2] = islocalmax(F_grid,2);
    TF = TF1 & TF2;
    heights = F_grid(TF);

    %% Find energy barrier to move between minima using peak prominence
    P1_filt = P1.*TF;
    P2_filt = P2.*TF;
    P = max(P1_filt,P2_filt,'omitnan');
    % TODO: split HF and control
    meanP(j,i) = mean(nonzeros(P));
    maxP(j,i) = max(max(P));
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

% Plot
ydata = [meanP_ctrl(:);meanP_HF(:)];
xgroupdata = [categorical(repmat({'Control'}, 1, length(meanP_ctrl(:)))), ....
    categorical(repmat({'HF'}, 1, length(meanP_HF(:))))];
figure;
boxchart(ydata, 'GroupByColor', xgroupdata);
hold on;

% Plot raw data points directly above their corresponding box plots
xHF = 1.25*ones(size(meanP_HF(:)));
xCtrl = 0.75 * ones(size(meanP_ctrl(:)));

scatter(xHF, meanP_HF(:), 'r', 'filled', 'MarkerFaceAlpha', 0.5);
scatter(xCtrl, meanP_ctrl(:), 'b', 'filled', 'MarkerFaceAlpha', 0.5);

ax = gca;
set(ax,'xticklabel',[])
ylabel('Mean Prominence');
legend({'Control','HF'}, 'Location', 'best');
hold off;
set(gca,'FontSize',16)
saveas(gcf,'Mean_prominence_12min.png')

% Median prominence
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

% Plot
ydata = [medianP_ctrl(:);medianP_HF(:)];
figure;
boxchart(ydata, 'GroupByColor', xgroupdata);
hold on;

% Plot raw data points directly above their corresponding box plots
scatter(xHF, medianP_HF(:), 'r', 'filled', 'MarkerFaceAlpha', 0.5);
scatter(xCtrl, medianP_ctrl(:), 'b', 'filled', 'MarkerFaceAlpha', 0.5);

ax = gca;
set(ax,'xticklabel',[])
ylabel('Median prominence');
legend({'Control','HF'}, 'Location', 'best');
hold off;
set(gca,'FontSize',16)
saveas(gcf,'Median_prominence_12min.png')

% Max prominence
disp('Max prominence')
maxP_HF = maxP(:,HF_idx);
maxP_ctrl = maxP(:,ctrl_idx);
disp('Mean and std HF:')
mean(maxP_HF(:),'omitmissing')
maxP_HF = rmmissing(maxP_HF);
std(maxP_HF(:))
disp('Mean and std control:')
maxP_ctrl = rmmissing(maxP_ctrl);
mean(maxP_ctrl(:))
std(maxP_ctrl(:))
[h,p,ci,stats] = ttest2(maxP_HF(:),maxP_ctrl(:))

% Plot
ydata = [maxP_ctrl(:);maxP_HF(:)];
xgroupdata = [categorical(repmat({'Control'}, 1, length(maxP_ctrl(:)))), ....
    categorical(repmat({'HF'}, 1, length(maxP_HF(:))))];

figure;
boxchart(ydata, 'GroupByColor', xgroupdata);
hold on;

% Plot raw data points directly above their corresponding box plots
xHF = 1.25*ones(size(maxP_HF(:)));
xCtrl = 0.75 * ones(size(maxP_ctrl(:)));
scatter(xHF, maxP_HF(:), 'r', 'filled', 'MarkerFaceAlpha', 0.5);
scatter(xCtrl, maxP_ctrl(:), 'b', 'filled', 'MarkerFaceAlpha', 0.5);

ax = gca;
set(ax,'xticklabel',[])
ylabel('Max Prominence');
legend({'Control','HF'}, 'Location', 'best');
hold off;
set(gca,'FontSize',16)
saveas(gcf,'Max_prominence_12min.png')

% Std of prominence
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
[h,p,ci,stats] = ttest2(stdP_HF(:),stdP_ctrl(:))

% Plot
ydata = [stdP_ctrl(:);stdP_HF(:)];
xgroupdata = [categorical(repmat({'Control'}, 1, length(stdP_ctrl(:)))), ....
    categorical(repmat({'HF'}, 1, length(stdP_HF(:))))];

figure;
boxchart(ydata, 'GroupByColor', xgroupdata);
hold on;

% Plot raw data points directly above their corresponding box plots
xHF = 1.25*ones(size(stdP_HF(:)));
xCtrl = 0.75 * ones(size(stdP_ctrl(:)));
scatter(xHF, stdP_HF(:), 'r', 'filled', 'MarkerFaceAlpha', 0.5);
scatter(xCtrl, stdP_ctrl(:), 'b', 'filled', 'MarkerFaceAlpha', 0.5);

ax = gca;
set(ax,'xticklabel',[])
ylabel('Standard deviation of prominence');
legend({'Control','HF'}, 'Location', 'best');
hold off;
set(gca,'FontSize',16)
saveas(gcf,'Std_prominence_12min.png')


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

% Plot
ydata = [meanTE_ctrl(:);meanTE_HF(:)];
xgroupdata = [categorical(repmat({'Control'}, 1, length(meanTE_ctrl(:)))), ....
    categorical(repmat({'HF'}, 1, length(meanTE_HF(:))))];

figure;
boxchart(ydata, 'GroupByColor', xgroupdata);
hold on;

% Plot raw data points directly above their corresponding box plots
xHF = 1.25*ones(size(meanTE_HF(:)));
xCtrl = 0.75 * ones(size(meanTE_ctrl(:)));
scatter(xHF, meanTE_HF(:), 'r', 'filled', 'MarkerFaceAlpha', 0.5);
scatter(xCtrl, meanTE_ctrl(:), 'b', 'filled', 'MarkerFaceAlpha', 0.5);

ax = gca;
set(ax,'xticklabel',[])
ylabel('Mean transition energy');
legend({'Control','HF'}, 'Location', 'best');
hold off;
set(gca,'FontSize',16)
saveas(gcf,'Mean_TE_12min.png')
end