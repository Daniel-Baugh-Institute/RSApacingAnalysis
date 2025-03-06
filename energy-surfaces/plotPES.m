function [PDF, PES] = plotPES(annotated_data,var1,var2,verbose)
% Function to plot potential energy surface
addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))
[num_slices, num_subjects] = size(annotated_data);
fig = figure;
fig.Position = [100, 100, 1600, 800];
rows = 2;
missing_data_flag = 0;
t = tiledlayout(fig,num_subjects,num_slices,'TileSpacing','compact','Padding','compact');
for jj = 1:num_subjects
    for i = 1:num_slices
        X = [];
        if strcmp(var1, 'CO') && strcmp(var2, 'CoBF')
            xx = 0:1:50;
            yy = 0:10:450;
            plot_type = 1;
            if jj == 10
                disp('CoBF data missing for animal 10')
                missing_data_flag = 1;
            else
                X(:,1) = annotated_data(i,jj).CO(5300000:5330000,1);
                X(:,2) = annotated_data(i,jj).CoBF(5300000:5330000,1);

            end

        elseif strcmp(var1, 'SBP') && strcmp(var2,'RR')
            xx = 0.18:0.02:1.4;
            yy = 70:2:200;
            plot_type = 2;
            l_RR = length(annotated_data(i,jj).RRint);
            l_MAP = length(annotated_data(i,jj).MAP);
            disp('RR size')
            size(annotated_data(i,jj).RRint)
            disp('MAP size')
            size(annotated_data(i,jj).MAP)
            if l_RR ~= l_MAP
                min_length = min (l_RR,l_MAP);
                annotated_data(i,jj).RRint = annotated_data(i,jj).RRint(1:min_length);
                annotated_data(i,jj).MAP = annotated_data(i,jj).MAP(1:min_length);
            end
            X =[annotated_data(i,jj).RRint, annotated_data(i,jj).MAP];
            disp('X size')
            size(X)
        else
            disp('var1 and var2 are invalid. Choose from CO and CoBF or SBP and RR')
            break
        end



        tt = 400; % number of heart beats for ~4-5 min period based on mean heart period of 0.6635 s
        % start index for peak at 2.22e4 seconds: 35528
        start = 35278;%35628;%35528;%1;%64657;%

        if missing_data_flag == 0
            % probability density plot
            [Xpts, Ypts] = meshgrid(xx,yy);
            pts1 = Xpts(:);
            pts2 = Ypts(:);
            pts = [pts1 pts2];
            [f, xi] = ksdensity(X,pts,'PlotFcn', 'surf');%,'Support','positive');
            F_grid = reshape(f, size(Xpts));

            % if jj == 6 || jj == 9
            %     figure;
            %     histogram(X(:,1))
            %     xlabel('CO')
            %     saveas(gcf,['COhist' num2str(jj) '.png'])
            %     figure;
            %     histogram(X(:,2))
            %     xlabel('CoBF')
            %     saveas(gcf,['CoBFhist' num2str(jj) '.png'])
            %     mean(F_grid)
            %     mean(-log(F_grid))
            % end

            nexttile;
            sc = surfc(Ypts, Xpts, -log(F_grid));
            hold on
            sc(2).EdgeColor = 'w';
            sc(1).EdgeColor = 'w';
            view(90,-90)

            % Axis labels
            if plot_type == 1
                ylabel(t,'Coronary blood flow (mL/min)')
                xlabel(t,'Cardiac output (L/min)')
                filename = ['plot_CO_CoBF_probability_subplot.png'];

                if jj == 6 || jj == 9
                    xlim([-5 5])
                    ylim([-5 5])
                end
            elseif plot_type == 2
                ylabel(t,'Systolic BP (mm Hg)')
                xlabel(t,'RR interval (s)')
                filename = 'plot_RR_SBP_probability_subplot.png';

                % Random sampling of traces
                % plot(X(start:start+100,2),X(start:start+100,1),'r-o','MarkerSize',2)
                % plot(X(35631:35635,2),X(35631:35635,1),'r-o','MarkerSize',2)
                % plot(X(start,2),X(start,1),'g-o','MarkerSize',8,'MarkerFaceColor','g')
                % plot(X(start+100,2),X(start+100,1),'r-o','MarkerSize',8,'MarkerFaceColor','r')
                % filename = ['plot_RR_SBP_probability_' num2str(jj) '.png'];
            else
                disp('Invalid plot type')
            end

            zlabel('Potential energy')
            % title(['Individual ' num2str(jj)])

            PES(i,jj).F_grid = -log(F_grid);
            PDF(i,jj).F_grid = F_grid;
            sprintf('Plotted animal %d',jj)
        end
        set(gcf,'Position',[0,0,3000,5000])
        saveas(gcf,filename)
        
        % TODO: need different loop variable for this
        % % plot histogram
        % if verbose == 1
        %     for jj = 1:length(annotated_data)
        %         X = annotated_data(i,jj).X(:,1:2);
        %         figure;
        %         h=histogram2(X(:,1),X(:,2),xx,yy);
        %         xlabel('RR interval (s)')
        %         ylabel('Systolic BP (mm Hg)')
        %         saveas(gcf,['plot_hist_15_' num2str(jj) '.png'])
        %     end
        % end
    end
end

end