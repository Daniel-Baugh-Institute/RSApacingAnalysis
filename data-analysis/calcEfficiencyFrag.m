function [work_segment_accel, work_segment_alt, efficiency_segment_accel, efficiency_segment_alt] = calcEfficiencyFrag(data)
% , work_mean, efficiency_mean, work_std, efficiency_std
% Function to calculate beat to beat cardiac work from cardiac output and
% arterial pressure
% Input: data, MxN struct where the row is the time sample and the column is
%   the animal number
addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))
rng default
HFidx = [1:4 10 13 15 17 19 21 23 25 27 29 31 34];
ctrlIdx = 5:9;
RSApaced = [11 14 16 18 20 22];
MonoPaced = [24 26 28 30 32 35];


[num_slices, num_subjects] = size(data)
num_slices = 1;

% Find indices for acceleration and alternation RR intervals
for jj = 3:num_subjects%1:num_subjects 
    for i = 1:num_slices
        if ~isnan(data(i,jj).RRint)
            nni = data(i,jj).RRint;
        else
            continue
        end
        [ ~, acceleration_segment_boundaries_3plus, alternation_segment_boundaries_2plus ] = mhrv.hrv.hrv_fragmentation( nni );
        
        acceleration_segment_boundaries_3plus(1:10,:)
        alternation_segment_boundaries_2plus(1:10,:)
        % Acceleration/deceleration windows
        work_segment_accel = [];
        efficiency_segment_accel = [];
        CoBF_per_beat_accel = [];

        for m = 1:length(acceleration_segment_boundaries_3plus)
            % Convert to time stamps
            start = data(i,jj).RRtime(acceleration_segment_boundaries_3plus(m,1));
            stop = data(i,jj).RRtime(acceleration_segment_boundaries_3plus(m,2));
            disp('calcEfficiencyFrag')
            num_beats_check = acceleration_segment_boundaries_3plus(m,2) - acceleration_segment_boundaries_3plus(m,1);
            % disp('Negative CoBF')
            % sum(data(i,jj).CoBF < 0)
            % 
            % disp('Negative CO')
            % sum(data(i,jj).CO < 0)
            % 
            % disp('Negative BP')
            % sum(data(i,jj).BP < 0)

            % Calculate work and efficiency for acceleration windows
            [work_temp, efficiency_temp, CoBF_per_beat_temp, ~, bp_max] = effCalc(data,jj,i,start,stop,num_beats_check);

            if work_temp > 0
                work_segment_accel(m) = work_temp;
            else
                work_segment_accel(m) = NaN;
            end
            if efficiency_temp > 0
                efficiency_segment_accel(m) = efficiency_temp;
            else
                efficiency_segment_accel(m) = NaN;
            end
            if CoBF_per_beat_temp > 0
                CoBF_per_beat_accel(m) = CoBF_per_beat_temp;
            else
                CoBF_per_beat_accel(m) = NaN;
            end
            % vol_max_accel(m) = vol_max;
            
            try
                bp_max_accel(m) = bp_max;
            catch
                disp('bp_max empty?')
                size(bp_max)
                bp_max_accel(m) = NaN;
            end
        end

        figure;
        histogram(bp_max_accel > 0)
        xlabel('BP')
        saveas(gcf,['hist_bp_accel' num2str(jj) '.png'])
        


        % Alternation windows
        work_segment_alt = [];
        efficiency_segment_alt = [];
        CoBF_per_beat_alt = [];
        for m = 1:length(alternation_segment_boundaries_2plus)
            % Convert to time stamps
            start = data(i,jj).RRtime(alternation_segment_boundaries_2plus(m,1));
            stop = data(i,jj).RRtime(alternation_segment_boundaries_2plus(m,2));
            num_beats_check = alternation_segment_boundaries_2plus(m,2) - alternation_segment_boundaries_2plus(m,1);

            % Calculate work and efficiency for acceleration windows
            [work_temp, efficiency_temp, CoBF_per_beat_temp, ~, bp_max] = effCalc(data,jj,i,start,stop,num_beats_check);

            if work_temp > 0
                work_segment_alt(m) = work_temp;
            else
                work_segment_alt(m) = NaN;
            end
            if efficiency_temp > 0
                efficiency_segment_alt(m) = efficiency_temp;
            else
                efficiency_segment_alt(m) = NaN;
            end

            if CoBF_per_beat_temp > 0
                CoBF_per_beat_alt(m) = CoBF_per_beat_temp;
            else
                CoBF_per_beat_alt(m) = NaN;
            end

            % vol_max_alt(m) = vol_max;

            try
                bp_max_alt(m) = bp_max;
            catch
                disp('bp_max_alt empty')
                bp_max_alt(m) = NaN;
            end

        end
        disp('here')
        length(CoBF_per_beat_alt)
        length(efficiency_segment_alt)
        length(work_segment_alt)


        figure;
        histogram(bp_max_alt > 0)
        xlabel('BP')
        saveas(gcf,['hist_bp_alt' num2str(jj) '.png'])

    end

    if ismember(jj,HFidx)
        tit = 'HF';
    elseif ismember(jj,ctrlIdx)
        tit = 'C';
    elseif ismember(jj,MonoPaced)
        tit = 'Mono paced';
    elseif ismember(jj,RSApaced)
        tit = 'RSA paced';
    else
        tit = 'off-pace';
    end

    % t-test within subject
    % Work
    disp('Comparing work between alternating and acceleration/deceleration groups')
    % remove nan
    work_segment_alt_clean = work_segment_alt(~isnan(work_segment_alt));
    work_segment_accel_clean = work_segment_accel(~isnan(work_segment_accel));
    [h,p,ci,stats] = ttest2(work_segment_alt_clean,work_segment_accel_clean)

    % Plot
    ydata = [work_segment_alt_clean(:); work_segment_accel_clean(:)];
    xgroupdata = [categorical(repmat({'Alternation'}, 1, length(work_segment_alt_clean(:)))), ....
    categorical(repmat({'Acceleration/deceleration'}, 1, length(work_segment_accel_clean(:))))];

    figure;
    boxchart(ydata, 'GroupByColor', xgroupdata,'MarkerStyle','none');
    hold on;

    % Plot raw data points directly above their corresponding box plots
    a = -0.2;
    b = 0.2;
    n = numel(work_segment_alt_clean);
    jitteralt = a + (b-a).*rand(n,1);
    n = numel(work_segment_accel_clean);
    jitteraccel = a + (b-a).*rand(n,1);
    xAccel = 1.25*ones(size(work_segment_accel_clean(:))) + jitteraccel;
    xAlt = 0.75 * ones(size(work_segment_alt_clean(:))) + jitteralt;
    scatter(xAlt, work_segment_alt_clean(:), 'b', 'filled', 'MarkerFaceAlpha', 0.5);
    scatter(xAccel, work_segment_accel_clean(:), 'r', 'filled', 'MarkerFaceAlpha', 0.5);

    ax = gca;
    set(ax,'xticklabel',[])
    ylabel('Segment work (mL*mm Hg)');
    legend({'Alternation','Acceleration/deceleration'}, 'Location', 'best');
    title(['Sheep ' num2str(jj) ' | ' tit])
    hold off;
    set(gca,'FontSize',16)
    saveas(gcf,['work_accel-alt_box_' num2str(jj) '.png'])
    struct(i,jj).work = ydata;

    % Efficiency
    disp('Comparing work between alternating and acceleration/deceleration groups')
    efficiency_segment_alt_clean = efficiency_segment_alt(~isnan(efficiency_segment_alt));
    efficiency_segment_accel_clean = efficiency_segment_accel(~isnan(efficiency_segment_accel));
    [h,p,ci,stats] = ttest2(efficiency_segment_alt_clean,efficiency_segment_accel_clean)

    % Plot
    ydata = [efficiency_segment_alt_clean(:); efficiency_segment_accel_clean(:)];
    xgroupdata = [categorical(repmat({'Alternation'}, 1, length(efficiency_segment_alt_clean(:)))), ....
    categorical(repmat({'Acceleration/deceleration'}, 1, length(efficiency_segment_accel_clean(:))))];

    figure;
    boxchart(ydata, 'GroupByColor', xgroupdata,'MarkerStyle','none');
    hold on;

    % Plot raw data points directly above their corresponding box plots
    a = -0.1;
    b = 0.1;
    n = numel(efficiency_segment_alt_clean);
    jitteralt = a + (b-a).*rand(n,1);
    n = numel(efficiency_segment_accel_clean);
    jitteraccel = a + (b-a).*rand(n,1);
    xAccel = 1.25*ones(size(efficiency_segment_accel_clean(:))) + jitteraccel;
    xAlt = 0.75 * ones(size(efficiency_segment_alt_clean(:))) + jitteralt;
    scatter(xAlt, efficiency_segment_alt_clean(:), 'b', 'filled', 'MarkerFaceAlpha', 0.5);
    scatter(xAccel, efficiency_segment_accel_clean(:), 'r', 'filled', 'MarkerFaceAlpha', 0.5);

    ax = gca;
    set(ax,'xticklabel',[])
    ylabel('Segment efficiency (mm Hg)');
    legend({'Alternation','Acceleration/deceleration'}, 'Location', 'best');
    title(['Sheep ' num2str(jj)  ' | ' tit])
    hold off;
    set(gca,'FontSize',16)
    saveas(gcf,['efficiency_accel-alt_box_' num2str(jj) '.png'])

    struct(i,jj).efficiency = ydata;

    % CoBF
    disp('Comparing CoBF between alternating and acceleration/deceleration groups')
    % remove nan
    CoBF_per_beat_alt_clean = CoBF_per_beat_alt(~isnan(CoBF_per_beat_alt));
    CoBF_per_beat_accel_clean = CoBF_per_beat_accel(~isnan(CoBF_per_beat_accel));
    [h,p,ci,stats] = ttest2(CoBF_per_beat_alt_clean,CoBF_per_beat_accel_clean)

    % Plot
    ydata = [CoBF_per_beat_alt_clean(:); CoBF_per_beat_accel_clean(:)];
    xgroupdata = [categorical(repmat({'Alternation'}, 1, length(CoBF_per_beat_alt_clean(:)))), ....
    categorical(repmat({'Acceleration/deceleration'}, 1, length(CoBF_per_beat_accel_clean(:))))];

    figure;
    boxchart(ydata, 'GroupByColor', xgroupdata,'MarkerStyle','none');
    hold on;

    % Plot raw data points directly above their corresponding box plots
    a = -0.2;
    b = 0.2;
    n = numel(CoBF_per_beat_alt_clean);
    jitteralt = a + (b-a).*rand(n,1);
    n = numel(CoBF_per_beat_accel_clean);
    jitteraccel = a + (b-a).*rand(n,1);
    xAccel = 1.25*ones(size(CoBF_per_beat_accel_clean(:))) + jitteraccel;
    xAlt = 0.75 * ones(size(CoBF_per_beat_alt_clean(:))) + jitteralt;
    scatter(xAlt, CoBF_per_beat_alt_clean(:), 'b', 'filled', 'MarkerFaceAlpha', 0.5);
    scatter(xAccel, CoBF_per_beat_accel_clean(:), 'r', 'filled', 'MarkerFaceAlpha', 0.5);

    ax = gca;
    set(ax,'xticklabel',[])
    ylabel('Segment CoBF per beat (mL)');
    legend({'Alternation','Acceleration/deceleration'}, 'Location', 'best');
    title(['Sheep ' num2str(jj) ' | ' tit])
    hold off;
    set(gca,'FontSize',16)
    saveas(gcf,['CoBF_accel-alt_box_' num2str(jj) '.png'])
    struct(i,jj).CoBF_per_beat = ydata;

    % 3D plot of CoBF, Efficiency, work
    length(CoBF_per_beat_alt)
    length(efficiency_segment_alt)
    length(work_segment_alt)
    figure; 
    plot3(CoBF_per_beat_alt,efficiency_segment_alt,work_segment_alt,'bo','MarkerFaceColor','b')
    hold on
    plot3(CoBF_per_beat_accel,efficiency_segment_accel,work_segment_accel,'ro','MarkerFaceColor','r')
    legend({'Alternation','Acceleration/deceleration'}, 'Location', 'best');
    xlabel('CoBF (mL)')
    ylabel('Efficiency (mm Hg)')
    zlabel('Work (mL*mm Hg)')
    title(['Sheep ' num2str(jj) ' | ' tit])
    set(gca,'FontSize',16)
    saveas(gcf,['plot3D_' num2str(jj) '.png'])
end





save('efficiencyFrag.mat','struct')
end