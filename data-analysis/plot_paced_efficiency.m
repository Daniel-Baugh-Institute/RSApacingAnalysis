function plot_paced_efficiency(work_per_beat, efficiency_per_beat, work_mean, efficiency_mean, work_std, efficiency_std)
% Function to plot results from calcEfficiency to compare before and after
% pacing 
% Inputs: work_per_beat, efficiency_per_beat, work_mean,
% efficiency_mean, work_std, efficiency_std,filename
addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))
rng default

%% CHANGE HERE FOR DIFFERENT GROUP COMPARISONS %%


% Compare unpaced HF and RSA paced HF
pacedRSAidx = [12 15 17 19 21 23] - 1;  % paced
baseRSAidx = [11 14 16 18 20 22] - 1;  % unpaced
offPaceRSAidx = 13 - 1;

% Compare unpaced HF and mono paced HF
baseMonoIdx = [23 25 27 29 31 34] - 1;  % unpaced
pacedMonoIdx = [24 26 28 30 32 35] - 1;  % paced
offPaceMono = 33 - 1;
% AIDs = [1:10 11 16 18 20 22 24 26 28 30 32 35]; 

x = [1 2 3];

%% Efficiency
% RSA paced
yRSA(1,:) = efficiency_mean(baseRSAidx(1):baseRSAidx(1)+2);
stdRSA(1,:) = efficiency_std(baseRSAidx(1):baseRSAidx(1)+2);
for i = 2:length(pacedRSAidx) 
    yRSA(i,:) = [efficiency_mean(baseRSAidx(i):baseRSAidx(i)+1), NaN];
    stdRSA(i,:) = [efficiency_std(baseRSAidx(i):baseRSAidx(i)+1), NaN];
end

% Monotonic paced
for i = 1:length(pacedRSAidx) 
    if i == 5
        yMono(i,:) = efficiency_mean(baseMonoIdx(i):baseMonoIdx(i)+2);
        stdMono(i,:) = efficiency_std(baseMonoIdx(i):baseMonoIdx(i)+2);
    else
        yMono(i,:) = [efficiency_mean(baseMonoIdx(i):baseMonoIdx(i)+1), NaN];
        stdMono(i,:) = [efficiency_std(baseMonoIdx(i):baseMonoIdx(i)+1), NaN];
    end   
end

figure;
for i = 1:length(pacedRSAidx)
    rng(i)
    jitter = 0.05*randn(1,3);
    errorbar(x+jitter,yRSA(i,:),stdRSA(i,:),'co-','MarkerFaceColor','c')
    hold on
    errorbar(x-jitter,yMono(i,:),stdMono(i,:),'mo-','MarkerFaceColor','m')
end

ax = gca;
xticks([1 2 3])
set(ax,'xticklabel',{'unpaced','paced','off-pace'})
ylabel('Efficiency (mm Hg)');
legend({'RSA','Mono'}, 'Location', 'best');
hold off;
set(gca,'FontSize',16)
saveas(gcf,'efficiency_paced.png')

% Calculate mean of delta from unpaced to paced for mono and RSA
disp('Comparing efficiency change from unpaced to paced between mono and RSA paced groups...')
diffMono = yMono(:,1) - yMono(:,2);
diffRSA = yRSA(:,1) - yRSA(:,2);
[h,p,ci,stats] = ttest2(diffMono,diffRSA)

% Plot
ydata = [diffMono; diffRSA];
xgroupdata = [categorical(repmat({'Mono'}, 1, length(diffMono))), ....
    categorical(repmat({'RSA'}, 1, length(diffRSA)))];

figure;
b = boxchart(ydata, 'GroupByColor', xgroupdata,'MarkerStyle','none');
hold on;
b(1).BoxFaceColor = 'magenta';
b(2).BoxFaceColor = 'cyan';
b(1).BoxFaceAlpha = 0.1;
b(2).BoxFaceAlpha = 0.1;

% Plot raw data points directly above their corresponding box plots
a = -0.1;
b = 0.1;
n = numel(diffRSA);
jitter = a + (b-a).*rand(n,1);
xMono = 0.75*ones(size(diffMono)) + jitter;
xRSA = 1.25 * ones(size(diffRSA)) + jitter;
scatter(xRSA, diffRSA, 'c', 'filled', 'MarkerFaceAlpha', 0.5,'MarkerEdgeColor','b');
scatter(xMono, diffMono, 'm', 'filled', 'MarkerFaceAlpha', 0.5,'MarkerEdgeColor','r');

ax = gca;
set(ax,'xticklabel',[])
ylabel({'Unpaced to paced efficiency'; 'change (mm Hg)'});
legend({'Mono','RSA'}, 'Location', 'best');
hold off;
set(gca,'FontSize',16)
saveas(gcf,'efficiency_delta_box.png')

%% Work

% RSA paced
yRSA(1,:) = work_mean(baseRSAidx(1):baseRSAidx(1)+2);
stdRSA(1,:) = work_std(baseRSAidx(1):baseRSAidx(1)+2);
for i = 2:length(pacedRSAidx) 
    yRSA(i,:) = [work_mean(baseRSAidx(i):baseRSAidx(i)+1), NaN];
    stdRSA(i,:) = [work_std(baseRSAidx(i):baseRSAidx(i)+1), NaN];
end

% Monotonic paced
for i = 1:length(pacedRSAidx) 
    if i == 5
        yMono(i,:) = work_mean(baseMonoIdx(i):baseMonoIdx(i)+2);
        stdMono(i,:) = work_std(baseMonoIdx(i):baseMonoIdx(i)+2);
    else
        yMono(i,:) = [work_mean(baseMonoIdx(i):baseMonoIdx(i)+1), NaN];
        stdMono(i,:) = [work_std(baseMonoIdx(i):baseMonoIdx(i)+1), NaN];
    end   
end

figure;
for i = 1:length(pacedRSAidx)
    jitter = 0.05*randn(1,3);
    errorbar(x+jitter,yRSA(i,:),stdRSA(i,:),'co-','MarkerFaceColor','c')
    hold on
    errorbar(x-jitter,yMono(i,:),stdMono(i,:),'mo-','MarkerFaceColor','m')
end

ax = gca;
xticks([1 2 3])
set(ax,'xticklabel',{'unpaced','paced','off-pace'})
ylabel('Work (mL*mm Hg)');
legend({'RSA','Mono'}, 'Location', 'best');
hold off;
set(gca,'FontSize',16)
saveas(gcf,'work_paced.png')

% Calculate mean of delta from unpaced to paced for mono and RSA
disp('Comparing efficiency change from unpaced to paced between mono and RSA paced groups...')
diffMono = yMono(:,1) - yMono(:,2);
diffRSA = yRSA(:,1) - yRSA(:,2);
[h,p,ci,stats] = ttest2(diffMono,diffRSA)

% Plot
ydata = [diffMono; diffRSA];
xgroupdata = [categorical(repmat({'Mono'}, 1, length(diffMono))), ....
    categorical(repmat({'RSA'}, 1, length(diffRSA)))];

figure;
b = boxchart(ydata, 'GroupByColor', xgroupdata,'MarkerStyle','none');
hold on;
b(1).BoxFaceColor = 'magenta';
b(2).BoxFaceColor = 'cyan';
b(1).BoxFaceAlpha = 0.1;
b(2).BoxFaceAlpha = 0.1;

% Plot raw data points directly above their corresponding box plots
a = -0.1;
b = 0.1;
n = numel(diffRSA);
jitter = a + (b-a).*rand(n,1);
xMono = 0.75*ones(size(diffMono)) + jitter;
xRSA = 1.25 * ones(size(diffRSA)) + jitter;
scatter(xRSA, diffRSA, 'c', 'filled', 'MarkerFaceAlpha', 0.5,'MarkerEdgeColor','b');
scatter(xMono, diffMono, 'm', 'filled', 'MarkerFaceAlpha', 0.5,'MarkerEdgeColor','r');

ax = gca;
set(ax,'xticklabel',[])
ylabel({'Unpaced to paced'; 'work change (mL*mm Hg)'});
legend({'Mono','RSA'}, 'Location', 'best');
hold off;
set(gca,'FontSize',16)
saveas(gcf,'work_delta_box.png')



end