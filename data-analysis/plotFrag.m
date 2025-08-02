function plotFrag(nni, nn_times, acceleration_segment_boundaries_3plus, alternation_segment_boundaries_4plus,filename)
% Function to plot alternation and acceleration/deceleration segments
% Input: 
% nni, RR or NN intervals 

% nn_times: RR interval timestamps

% acceleration_segment_boundaries_3plus: nx2 matrix with
% acceleration/deceleration segments with at least three heart beats. First
% column is start index of segment, second column is stop index.

% alternation_segment_boundaries_4plus: alternation segments with at least
% four heart beats. First column is start index of segment, second column
% is stop index.



mydir = pwd
addpath(genpath(mydir))

% Separate acceleration versus deceleration segments for plotting
[num_samples,~] = size(acceleration_segment_boundaries_3plus);
acceleration_segment_boundaries = [];
deceleration_segment_boundaries = [];
for j = 1:num_samples
    start_idx = acceleration_segment_boundaries_3plus(j,1);
    end_idx = acceleration_segment_boundaries_3plus(j,2);

    % acceleration (nni decreasing)
    if nni(start_idx) > nni(end_idx)
        acceleration_segment_boundaries = [acceleration_segment_boundaries; start_idx end_idx];
    % deceleration (nni increasing)
    elseif nni(start_idx) < nni(end_idx)
        deceleration_segment_boundaries = [deceleration_segment_boundaries; start_idx end_idx];
    else
        disp('Error: not an acceleration or deceleration segment')
    end
end

figure;
hold on;




offset = 0.001; % offset for visualization
% Acceleration
[num_samples,~] = size(acceleration_segment_boundaries);
for i = 1:num_samples
    times = nn_times(acceleration_segment_boundaries(i,1):acceleration_segment_boundaries(i,2));
    intervals = nni(acceleration_segment_boundaries(i,1):acceleration_segment_boundaries(i,2));
    if i == 1
       plot(times,intervals+offset,'c-','LineWidth',3,'DisplayName', 'Acceleration')
    else
        plot(times,intervals+offset,'c-','LineWidth',3,'HandleVisibility','off')
    end
    hold on
end

% Deceleration
[num_samples,~] = size(deceleration_segment_boundaries);
for i = 1:num_samples
    times = nn_times(deceleration_segment_boundaries(i,1):deceleration_segment_boundaries(i,2));
    intervals = nni(deceleration_segment_boundaries(i,1):deceleration_segment_boundaries(i,2));
    if i == 1
       plot(times,intervals+offset,'Color',[0.9608    0.4667    0.1608],'LineWidth',3,'DisplayName', 'Decleration')
    else
        plot(times,intervals+offset,'Color',[0.9608    0.4667    0.1608],'LineWidth',3,'HandleVisibility','off')
    end
    hold on
end


% Alternation
[num_samples,~] = size(alternation_segment_boundaries_4plus);
for i = 1:num_samples
    times = nn_times(alternation_segment_boundaries_4plus(i,1):alternation_segment_boundaries_4plus(i,2));
    intervals = nni(alternation_segment_boundaries_4plus(i,1):alternation_segment_boundaries_4plus(i,2));
    if i == 1
        plot(times,intervals-offset,'m-','LineWidth',3,'DisplayName','Alternation')
    else
        plot(times,intervals-offset,'m-','LineWidth',3,'HandleVisibility','off')
    end
    hold on
end

plot(nn_times,nni,'ko-','HandleVisibility','off','LineWidth',1)
xlabel('Time (s)')
ylabel('RR interval (s)')

legend('Location','northoutside','Orientation','horizontal')
set(gcf,'Position',[0 0 800 500])
set(gca,'FontSize',24)
xlim([nn_times(1)-0.001 nn_times(end)+0.001])
filename = ['./plots/' filename];
saveas(gcf,filename)


end