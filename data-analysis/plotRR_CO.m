function plotRR_CO(nni, nn_times, data)
% Function to plot alternation and acceleration/deceleration segments
% Input: 
% nni, RR or NN intervals 

% nn_times: RR interval timestamps

% data: matrix with CO data and times

addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))

tiledlayout(2,1,'TileSpacing','tight')
% RR intervals
nexttile;
plot(nn_times,nni,'ko-')
% xlim([1545 1575])
ylabel('RR interval (s)')
set(gca,'FontSize',16)

% CO

save('plotRR_CO.mat','nn_times','nni','data')
nexttile;
COidx = find(data.timeHRs_CO*3600 > min(nn_times) & data.timeHRs_CO*3600 < max(nn_times));
times_CO = data.timeHRs_CO(COidx)*60*60; %s
CO = data.CO(COidx);
plot(times_CO,CO,'b-')

hold on
xlabel('Time (s)')
ylabel('CO (L/min)')
% xlim([1545 1575])
set(gca,'FontSize',16)
filename = 'plot_CO_RR.png';
saveas(gcf,filename)

end