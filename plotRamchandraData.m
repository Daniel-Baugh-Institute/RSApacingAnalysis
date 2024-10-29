function plotRamchandraData(channelNum,prefix,plotName)

    n = length(channelNum); % 4 data channels
    tiledlayout(n,1)
    
    for i = 1:n
        structName = [prefix num2str(channelNum(i))];
        start = eval([structName '.start']);
        interval = eval([structName '.interval']);
        stop = eval([structName '.length'])*interval;
        timeRaw = start:interval:stop; % seconds
        time = timeRaw(1:end-1)./60./60; % hours
        nexttile(i)
        plot(time,eval([structName '.values']))
        
        % xlabel
        if i == n
        xlabel('Time (hrs)')
        end

        % ylabel
        if eval([structName '.title']) == 'BP'
            ylab = 'BP (mm Hg)';
        elseif eval([structName '.title']) == 'CoBF'
            ylab = 'Coronary blood flow (mL/min)';
        else
            ylab = 'Cardiac output (L/min)';
        end

        ylabel(ylab)
    end

    saveas(gcf,plotName)
end
