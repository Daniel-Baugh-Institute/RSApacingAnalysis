function [PDF, PES] = plotPES(annotated_data,var1,var2,verbose)
% Function to plot potential energy surface
addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))
[num_slices, num_subjects] = size(annotated_data);
num_slices = 5;%num_slices - 1;

fig = figure;
fig.Position = [100, 100, 1600, 800];

missing_data_flag = 0;
t = tiledlayout(fig,num_subjects,num_slices,'TileSpacing','compact','Padding','compact');
for jj = 1:num_subjects
    for i = 1:num_slices
        X = [];
        if strcmp(var1, 'CO') && strcmp(var2, 'CoBF')
            xx = 0:1:50;
            yy = 0:10:400;
            plot_type = 1;
            if jj == 10
                disp('CoBF data missing for animal 10')
                missing_data_flag = 1;
            else
                X(:,1) = annotated_data(i,jj).CO(:,1);
                X(:,2) = annotated_data(i,jj).CoBF(:,1);

            end

            % probability density plot
            [Xpts, Ypts] = meshgrid(xx,yy);
            pts1 = Xpts(:);
            pts2 = Ypts(:);
            pts = [pts1 pts2];
            X(1:10,:)
            [f, xi] = ksdensity(X,pts,'PlotFcn', 'surf');%,'Support','positive');
            F_grid = reshape(f, size(Xpts));

        elseif strcmp(var1, 'SBP') && strcmp(var2,'RR')
            xx = 0.18:0.02:1.4;
            yy = 70:2:200;
            plot_type = 2;
            l_RR = length(annotated_data(i,jj).RRint);
            l_MAP = length(annotated_data(i,jj).MAP);
            % disp('RR size')
            % size(annotated_data(i,jj).RRint)
            % disp('MAP size')
            % size(annotated_data(i,jj).MAP)
            if l_RR ~= l_MAP
                min_length = min (l_RR,l_MAP);
                annotated_data(i,jj).RRint = annotated_data(i,jj).RRint(1:min_length);
                annotated_data(i,jj).MAP = annotated_data(i,jj).MAP(1:min_length);
            end
            X =[annotated_data(i,jj).RRint, annotated_data(i,jj).MAP];
            % disp('X size')
            % size(X)
            % probability density plot
            [Xpts, Ypts] = meshgrid(xx,yy);
            pts1 = Xpts(:);
            pts2 = Ypts(:);
            pts = [pts1 pts2];
            [f, xi] = ksdensity(X,pts,'PlotFcn', 'surf');%,'Support','positive');
            F_grid = reshape(f, size(Xpts));
        elseif strcmp(var1, 'all') && strcmp(var2,'all') % 4D plot
            gridx1 = 0.18:0.02:1.4;
            gridx2 = 70:2:200;
            gridx3 = 0:1:50;
            gridx4 = 0:10:400;
            [x1,x2,x3,x4] = ndgrid(gridx1,gridx2,gridx3,gridx4);
            x1 = x1(:,:)';
            x2 = x2(:,:)';
            x3 = x3(:,:)';
            x4 = x4(:,:)';
            xi = [x1(:) x2(:) x3(:) x4(:)];
            d = 4; % dimension of data
            n = length(annotated_data(i,jj).RRint); % number of observations
            sigma = zeros(1,d); % std of variate
            sigma(1) = std(annotated_data(i,jj).RRint,'omitmissing');
            sigma(2) = std(annotated_data(i,jj).MAP,'omitmissing');
            sigma(3) = std(annotated_data(i,jj).CO_mean,'omitmissing');
            sigma(4) = std(annotated_data(i,jj).CoBF_mean,'omitmissing');
            sprintf('Animal %d, slice %d',jj,i)
            % sum(any(isnan(annotated_data(i,jj).CoBF_mean)))
            % sum(any(isnan(annotated_data(i,jj).CO_mean)))
            % sum(any(isnan(annotated_data(i,jj).MAP)))
            % sum(any(isnan(annotated_data(i,jj).RRint)))
            bw = sigma*(4/((d+2)*n))^(1/(d+4)); % Silverman's rule of thumb
            clear x % reset from previous slice/animal
            x(:,1) = annotated_data(i,jj).RRint;
            x(:,2) = annotated_data(i,jj).MAP;
            x(:,3) = annotated_data(i,jj).CO_mean;
            x(:,4) = annotated_data(i,jj).CoBF_mean;

            f = mvksdensity(x,xi,'Bandwidth',bw);
            F_grid = reshape(f,[length(gridx1) length(gridx2) length(gridx3) length(gridx4)]);
            plot_type = 3;
        else
            disp('var1 and var2 are invalid. Choose from CO and CoBF or SBP and RR or all and all')
            break
        end



        tt = 400; % number of heart beats for ~4-5 min period based on mean heart period of 0.6635 s
        % start index for peak at 2.22e4 seconds: 35528
        start = 35278;%35628;%35528;%1;%64657;%

        if plot_type == 1 || plot_type == 2


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
                filename = ['plot_CO_CoBF_probability_subplot_12min.png'];

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

         
            sprintf('Plotted animal %d',jj)
            set(gcf,'Position',[0,0,3000,5000])
            saveas(gcf,filename)
        end
        PES(i,jj).F_grid = -log(F_grid);
        PDF(i,jj).F_grid = F_grid;
    end
end

end