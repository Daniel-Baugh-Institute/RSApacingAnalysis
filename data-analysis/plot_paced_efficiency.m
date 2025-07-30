function plot_paced_efficiency(work_mean, efficiency_mean, work_std, efficiency_std,healthy_mean, healthy_std,control_flag)
% Function to plot results from calcEfficiency to compare before and after
% pacing 
% Inputs: work_per_beat, efficiency_per_beat, work_mean,
% efficiency_mean, work_std, efficiency_std,filename, control_flag (1 if
% plotting control/healthy animals, 0 if not)
addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))
rng default

%% CHANGE HERE FOR DIFFERENT GROUP COMPARISONS %%


% Compare unpaced HF and RSA paced HF
pacedRSAidx = [11 14 16 18 20 22];  % paced
baseRSAidx = [10 13 15 17 19 21];  % unpaced
offPaceRSAidx = 13 - 1;

% Compare unpaced HF and mono paced HF
baseMonoIdx = [23 25 27 29 31 34];  % unpaced
pacedMonoIdx = [24 26 28 30 32 35];  % paced
offPaceMono = 33;
% AIDs = [1:10 11 16 18 20 22 24 26 28 30 32 35]; 

x = [2 3 4];

%% Efficiency / CoBF / MAP
varName = 'Efficiency (mm Hg^{-1})'; %'PIP (%)';
filePrefix = 'efficiency';%'pip';
% RSA paced
yRSA(1,:) = efficiency_mean(baseRSAidx(1):baseRSAidx(1)+2);
stdRSA(1,:) = efficiency_std(baseRSAidx(1):baseRSAidx(1)+2);
for i = 2:length(pacedRSAidx) 
    yRSA(i,:) = [efficiency_mean(baseRSAidx(i):baseRSAidx(i)+1), NaN];
    stdRSA(i,:) = [efficiency_std(baseRSAidx(i):baseRSAidx(i)+1), NaN];
end

% Monotonic paced
for i = 1:length(pacedMonoIdx) 
    if i == 5
        yMono(i,:) = efficiency_mean(baseMonoIdx(i):baseMonoIdx(i)+2);
        stdMono(i,:) = efficiency_std(baseMonoIdx(i):baseMonoIdx(i)+2);
    else
        yMono(i,:) = [efficiency_mean(baseMonoIdx(i):baseMonoIdx(i)+1), NaN];
        stdMono(i,:) = [efficiency_std(baseMonoIdx(i):baseMonoIdx(i)+1), NaN];
    end   
end
disp('yMono in plot_paced_efficiency')
yMono
yRSA

figure;
hold on;
ax = gca;
if control_flag == 1
    xticks([1 2 3 4])
    set(ax,'xticklabel',{'healthy','unpaced','paced','off-pace'})
    jitter = 0.05*randn(1,length(healthy_mean));
    errorbar(ones(1,length(healthy_mean))+jitter, healthy_mean, healthy_std,'bo','MarkerFaceColor','b','HandleVisibility','off')
else
    xticks([2 3 4])
    set(ax,'xticklabel',{'unpaced','paced','off-pace'})
end

for i = 1:length(pacedMonoIdx)
    rng(i)
    jitter = 0.05*randn(1,3);
    if i == 1
        errorbar(x+jitter,yRSA(i,:),stdRSA(i,:),'co-','MarkerFaceColor','c','LineWidth',2,'MarkerEdgeColor','b','HandleVisibility','on')
        hold on
    elseif i == 5
        errorbar(x-jitter,yMono(i,:),stdMono(i,:),'mo-','MarkerFaceColor','m','LineWidth',2,'MarkerEdgeColor','r','HandleVisibility','on')
    else
        errorbar(x+jitter,yRSA(i,:),stdRSA(i,:),'co-','MarkerFaceColor','c','HandleVisibility','off')
        hold on
        errorbar(x-jitter,yMono(i,:),stdMono(i,:),'mo-','MarkerFaceColor','m','HandleVisibility','off')
    end
end



ylabel(varName)
% ylabel('Efficiency (mm Hg)');
legend({'RSA','Mono'}, 'Location', 'best');
hold off;
set(gca,'FontSize',16)
% saveas(gcf,'efficiency_paced.png')
saveas(gcf,[filePrefix '_paced.png'])

% Calculate mean of delta from unpaced to paced for mono and RSA
disp('Comparing efficiency change from unpaced to paced between mono and RSA paced groups...')
diffMono = yMono(:,2) - yMono(:,1);
diffRSA = yRSA(:,2) - yRSA(:,1);
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
% ylabel({'Unpaced to paced efficiency'; 'change (mm Hg)'});
varName = 'efficiency';
ylabel({['Unpaced to paced ' varName]; 'change (mm Hg^{-1})'});
legend({'Mono','RSA'}, 'Location', 'best');
hold off;
set(gca,'FontSize',16)
% saveas(gcf,'efficiency_delta_box.png')
saveas(gcf,[filePrefix '_delta_box.png'])

%% Work / CO / RR
varName = 'Work (mL*mm Hg)';
filePrefix = 'work';
% RSA paced
yRSA(1,:) = work_mean(baseRSAidx(1):baseRSAidx(1)+2);
stdRSA(1,:) = work_std(baseRSAidx(1):baseRSAidx(1)+2);
for i = 2:length(pacedRSAidx) 
    yRSA(i,:) = [work_mean(baseRSAidx(i):baseRSAidx(i)+1), NaN];
    stdRSA(i,:) = [work_std(baseRSAidx(i):baseRSAidx(i)+1), NaN];
end

% Monotonic paced
for i = 1:length(pacedMonoIdx) 
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
    try
        errorbar(x-jitter,yMono(i,:),stdMono(i,:),'mo-','MarkerFaceColor','m')
    catch
        disp('Missing data')
    end
end

ax = gca;
xticks([1 2 3])
set(ax,'xticklabel',{'unpaced','paced','off-pace'})
% ylabel('Work (mL*mm Hg)');
ylabel(varName)
legend({'RSA','Mono'}, 'Location', 'best');
hold off;
set(gca,'FontSize',16)
% saveas(gcf,'work_paced.png')
saveas(gcf,[filePrefix '_paced.png'])

% Calculate mean of delta from unpaced to paced for mono and RSA
disp('Comparing efficiency change from unpaced to paced between mono and RSA paced groups...')
diffMono = yMono(:,2) - yMono(:,1);
diffRSA = yRSA(:,2) - yRSA(:,1);
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
% ylabel({'Unpaced to paced'; 'work change (mL*mm Hg)'});
ylabel({'Unpaced to paced'; 'work change (mL * mm Hg)'});
legend({'Mono','RSA'}, 'Location', 'best');
hold off;
set(gca,'FontSize',16)
saveas(gcf,[filePrefix '_delta_box.png'])



end