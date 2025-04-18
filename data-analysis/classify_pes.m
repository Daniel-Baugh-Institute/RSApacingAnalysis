function trainednet = classify_pes(PES)
% Train and test classifier of PES for heart failure vs control animals
%
% Inputs:
%   PES - MxN matrix where the row is the time sample and the column is
%   the animal number
%
% Output:
%   D_B - Bhattacharyya distance

%% 2D data
% addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))
close all;
rng(48)
disp('num slices, num subjects')
[num_slices, num_subjects] = size(PES)

%
% Train the network Network options: while it would be nice to take
% advantage of the time series nature of the data, any neural net that does
% would likely only capture circadian rhythm. Unless this is thrown off in
% heart failure?? (It is, see
% https://www.ahajournals.org/doi/10.1161/CIRCRESAHA.122.321369) These
% types of networks include LSTM (see sequence classification using deep
% learning matlab. This looks like it classifies time series waveforms) and
% vector sequence classification networks
% For implementation see: https://www.mathworks.com/help/deeplearning/ref/dlarray.html

% X format for 3D image: h-by-w-by-d-by-c-by-n numeric array, where h, w,
% d, c and n are the height, width, depth, number of channels of the
% images, and number of image observations, respectively.
c = 3; % one channel, potential energy
numSamples = num_subjects * num_slices;
numClasses = 2; % normal and heart failure
holdout = 0.3;
% inputSizeTrain = [662 881 c NaN];%[541 684 c NaN]; %
inputFormat = 'SSCB';

%% Unpack 2D data and format into imagestore
% addpath(genpath('C:\Users\mmgee\Box\Michelle-Gee\Research\Patient-specific models\Auckland_physiology_data'))
addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))

%%%%%% CHANGE THIS PART FOR CO/COBF VS RR/MAP %%%%%%%%%%%%%
% RR and MAP grid
% xx = 0.18:0.02:1.4;
% yy = 70:2:200;
% gen_location = '/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/PESimages/RR_MAP_30m_48slices/';


% CO and CoBF grid
xx = 0:1:50;
yy = 0:10:400;
% gen_location = 'C:\Users\mmgee\Box\Michelle-Gee\Research\Patient-specific models\Auckland_physiology_data\RR_MAP_1hr\'; % location of image files
gen_location = '/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/PESimages/CO_CoBF_30m_48slices/';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[Xpts, Ypts] = meshgrid(xx,yy);
count = 1;

% Sort images into folders for labels
num_HF_animals = 16; %5;
num_ctrl_animals = 5;%4;
numRepeatsHF = num_slices*num_HF_animals;
numRepeatsCtrl = num_slices*num_ctrl_animals;
HFidx = [1:4 10 13 15 17 19 21 23 25 27 29 31 34];
AIDs = [1:10 13 15 17 19 21 23 25 27 29 31 34]; % animal IDs

% CHECK THAT IMAGES GO TO CORRECT FOLDER #########################################

i = 1;
j = 1;
for i = 1:length(AIDs)%num_subjects
    for j = 1:num_slices
        P = PES(j,AIDs(i)).F_grid;

        % Don't create image file if no data for that window
        if any(isnan(P))
            i
            j
            count = count + 1;
        else
            % Check for inf of -inf vlaues from taking -ln(0) to get PES from PDF
            P(isinf(P)) = 0;
            X(:,:,c, count) = P(:,:,1,1);

            % save image files to location
            figure;
            try
                sc = surfc(Ypts, Xpts, X(:,:,c, count));
            catch
                i
                j
                return
            end
            colormap(jet)
            % Check colormap limits
            clim_save(j,:) = clim;

            clim([0 33])% CO-CoBF but check for expanded dataset([0 33]) % RR-MAP, but check expanded dataset ([-1.5 22]) 
            hold on
            sc(2).EdgeColor = 'none';
            sc(1).EdgeColor = 'none';
            view(90,-90)
            % shading interp
            grid off
            axis off
            ax = gca;
            if ismember(AIDs(i),HFidx)
                location = [gen_location 'HF/'];
            else
                location = [gen_location 'C/'];
            end
            AID = AIDs(i);
            filename = [location 'CO_CoBF_30m_A' num2str(AID) '_S' num2str(j)];
            % exportgraphics(ax,filename,'Resolution',150)
            resolution = 150; % resolution in dpi

            % Randomly flip images to avoid over training
            % rand_flipx = randn(1);
            % if rand_flipx > 0
            %     set(gca,'xdir','reverse')
            % end
            % rand_flipy = randn(1);
            % if rand_flipy > 0
            %     set(gca,'ydir','reverse')
            % end

            set(gcf,'position',[0 0 600 500])
            set(gca,'position',[0 0 1 1]);
            print(filename,'-dpng',['-r' num2str(resolution)]);
            filename_ext = [filename '.png'];
            a = imread(filename_ext);
            [rows, columns, numberOfColorChannels] = size(a);

            count = count + 1;
            close all;
        end
    end
end
imageSize = [rows columns c]; %[781 938 c];% CO/CoBF[543 686 c];

disp('Clim max')
max(clim_save(:,2))

disp('Clim min')
min(clim_save(:,1))

imds = imageDatastore(gen_location,'IncludeSubfolders',true,'LabelSource','foldernames');


% Separate to training and test data
% Randomized train-test split
% [dataTrain, dataTest] = splitEachLabel(imds, 1-holdout, 'randomized');

% Train-test split by animal
% Get list of all .png files in the directory
% fileList = dir('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/PESimages/CO_CoBF_30m_48slices/**/*.png');
% % fileList = dir('C:\Users\mmgee\Box\Michelle-Gee\Research\Patient-specific models\Auckland_physiology_data\RR_MAP_1hr\**\*.png');
% 
% % Filter files for test set (A8 (HF), A1 (C), A (C))
% filteredFileList = fileList(~cellfun('isempty', regexp({fileList.name}, 'A7|A8|A5|A1')));
% num_HF_samples = length({fileList(~cellfun('isempty', regexp({fileList.name}, 'A7|A8'))).name});
% num_C_samples = length({fileList(~cellfun('isempty', regexp({fileList.name}, 'A1|A5'))).name});
% 
% % Get full file paths
% fullFilePaths = fullfile({filteredFileList.folder}, {filteredFileList.name});
% 
% % Create FileSet
% fs = matlab.io.datastore.FileSet(string(fullFilePaths));
% imds_test = imageDatastore(fs);
% imds_test.Labels = categorical([repmat({'HF'}, num_HF_samples, 1); repmat({'C'}, num_C_samples, 1)]);
% 
% % Training set
% filteredFileList = fileList(~cellfun('isempty', regexp({fileList.name}, 'A2|A3|A4|A6|A9|A10')));
% num_HF_samples = length({fileList(~cellfun('isempty', regexp({fileList.name}, 'A6|A9|A10'))).name});
% num_C_samples = length({fileList(~cellfun('isempty', regexp({fileList.name}, 'A2|A3|A4'))).name});
% 
% % Get full file paths
% fullFilePaths = fullfile({filteredFileList.folder}, {filteredFileList.name});
% 
% % Create FileSet
% fs_train = matlab.io.datastore.FileSet(string(fullFilePaths));
% imds_train = imageDatastore(fs_train);
% imds_train.Labels = categorical([repmat({'HF'}, num_HF_samples, 1); repmat({'C'}, num_C_samples, 1)]);
% 
% 
% dataTrain = imds_train;
% dataTest = imds_test;
% size(imds_test.Labels)
% 
% 
% % Split the training set into training and validation sets for k-fold cross-validation
% k = 5; % Number of folds
% cvp = cvpartition(dataTrain.Labels, 'KFold', k);
% % Access the training, validation, and test sets
% for fold = 1:k
%     trainIdx = training(cvp, fold);
%     valIdx = test(cvp, fold);
%     trainFoldImds = subset(dataTrain, trainIdx);
%     valFoldImds = subset(dataTrain, valIdx);
% end
% 
% 
% % Create network
% layers = layerGraph();
% 
% % Input layer
% il = imageInputLayer(imageSize,'Name','input','Normalization','zscore');
% % il = inputLayer(inputSizeTrain,inputFormat);
% 
% % Main branch
% outputChannels = 4;
% cv1 = convolution2dLayer(3,outputChannels,'Padding','same','Name','cv1',WeightsInitializer='he');
% bn1 = batchNormalizationLayer('Name', 'bn1');
% relu1 = reluLayer('Name', 'relu1');
% pool1 = maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1');
% 
% cv2 = convolution2dLayer(3,outputChannels,'Padding','same','Name','cv2',WeightsInitializer='he');
% bn2 = batchNormalizationLayer('Name', 'bn2');
% relu2 = reluLayer('Name', 'relu2');
% pool2 = maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2');
% 
% cv3 = convolution2dLayer(3,outputChannels,'Padding','same','Name','cv3',WeightsInitializer='he');
% bn3 = batchNormalizationLayer('Name', 'bn3');
% relu3 = reluLayer('Name', 'relu3');
% drop3 = dropoutLayer(0.3,'Name','drop3');
% pool3 = maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3');
% 
% 
% % Final layers
% flatten = flattenLayer('Name', 'flatten');
% 
% fc3 = fullyConnectedLayer(numClasses, 'Name', 'fc3');
% sigmoid = sigmoidLayer('Name', 'sigmoid');
% 
% % **Construct the layer graph**
% layers = addLayers(layers, il);
% layers = addLayers(layers, cv1);
% layers = addLayers(layers, bn1);
% layers = addLayers(layers, relu1);
% layers = addLayers(layers, pool1);
% layers = addLayers(layers, cv2);
% layers = addLayers(layers, bn2);
% layers = addLayers(layers, relu2);
% layers = addLayers(layers, pool2);
% layers = addLayers(layers, cv3);
% layers = addLayers(layers, bn3);
% layers = addLayers(layers, relu3);
% layers = addLayers(layers, drop3);
% layers = addLayers(layers, pool3);
% layers = addLayers(layers,flatten);
% layers = addLayers(layers, fc3);
% layers = addLayers(layers, sigmoid);
% 
% 
% 
% % **Connect the layers**
% layers = connectLayers(layers, 'input', 'cv1');
% layers = connectLayers(layers, 'cv1', 'bn1');
% layers = connectLayers(layers, 'bn1', 'relu1');
% layers = connectLayers(layers, 'relu1','pool1');
% layers = connectLayers(layers, 'pool1','cv2');
% layers = connectLayers(layers, 'cv2', 'bn2');
% layers = connectLayers(layers, 'bn2', 'relu2');
% layers = connectLayers(layers, 'relu2','pool2');
% layers = connectLayers(layers, 'pool2','cv3');
% layers = connectLayers(layers, 'cv3', 'bn3');
% layers = connectLayers(layers, 'bn3', 'relu3');
% layers = connectLayers(layers, 'relu3', 'drop3');
% layers = connectLayers(layers, 'drop3', 'pool3');
% layers = connectLayers(layers, 'pool3','flatten');
% layers = connectLayers(layers, 'flatten', 'fc3');
% layers = connectLayers(layers, 'fc3', 'sigmoid');
% 
% 
% 
% 
% 
% % Convert to dlnetwork
% net = dlnetwork(layers);
% 
% 
% % Train network
% lossFcn = "binary-crossentropy"; % for binary classification tasks
% options = trainingOptions("rmsprop", ...
%     Plots="training-progress", ...
%     Metrics="accuracy", ...
%     GradientThresholdMethod="l2norm",...
%     GradientThreshold=1,...
%     MaxEpochs=120,...
%     Shuffle="every-epoch",...
%     MiniBatchSize=16,...
%     InitialLearnRate=0.0000001,...
%     ValidationData=dataTest,...
%     Verbose=false,...
%     InputDataFormats=inputFormat,...
%     L2Regularization=0.00001,...
%     ValidationFrequency=2, ...
%     ValidationPatience=200,...
%     LearnRateSchedule = "piecewise",...
%     LearnRateDropFactor=0.2,...
%     LearnRateDropPeriod=50);
% 
% 
% 
% 
% % [trainednet, info] = trainnet(dataTrain, net, lossFcn, options)
% % save('C:\Users\mmgee\Box\Michelle-Gee\Research\Patient-specific models\Auckland_physiology_data\trainednet_RR_MAP_24h_1h-bySheep.mat','trainednet')
% load('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/PESimages/RR_MAP_1hr/trainednet_CO_CoBF_30m-bySheep.mat','trainednet')
% % load('C:\Users\mmgee\Box\Michelle-Gee\Research\Patient-specific models\Auckland_physiology_data\trainednet_RR_MAP_24h_1hwindow.mat','trainednet')
% 
% 
% % This saves gui as image but doesn't seem to work if verbose=true in
% % trainnet
% h= findall(groot,'Type','Figure');
% %searching fig tag for Training Progress
% % for i=1: size(h,1)
% %     if strcmp(h(i).Tag,'DEEPMONITOR_UIFIGURE')
% %        savefig(h(i),"TrainedTest-30mwindow-24h-CO-CoBF.fig");
% %        break
% %     end
% % end
% 
% % Classify the test images
% num_samples = length(dataTest.Files);
% classNames = {'HF','C'};
% for i = 1:num_samples
%     im = imread(dataTest.Files{i});
%     xx = single(im); % makes data single precision
%     scores(:,i) = predict(trainednet,xx);
% end
% 
% scores
% 
% % plot AUC
% [X,Y,T,AUC] = perfcurve(dataTest.Labels, scores(2,:), 'HF');
% figure;
% plot(X,Y)
% xlabel('False positive rate') 
% ylabel('True positive rate')
% title('ROC for Classification by Logistic Regression')
% saveas(gcf,'AUC.png')
% 
% % Convert scores to labels
% for i = 1:num_samples
%     if scores(1,i) > scores(2,i)
%         label{i} = 'C';
%     else
%         label{i} = 'HF';
%     end
% end
% 
% % plot confusion matrix
% confusionchart(dataTest.Labels, label);
% 
% % metrics for test set
% % positive is considered heart failure
% groundTruth = dataTest.Labels;
% 
% for i = 1:length(label)
%     if label{i} == groundTruth(i)
%         correct_class(i) = 1;
%     else
%         correct_class(i) = 0;
%     end
% end
% 
% % accuracy
% accuracy = sum(correct_class)/length(correct_class)
% 
% % recall
% num_HF = 0;
% for i = 1:length(label)
%     if 'HF' == groundTruth(i)
%         num_HF = num_HF + 1;
%     end
% end
% 
% TP = 0; % true positives
% for i = 1:num_HF
%     if label{i} == groundTruth(i)
%         TP = TP + 1;
%     end
% end
% 
% FN = num_HF - TP; % false negative
% recall = TP/(TP+FN)
% 
% % false positive rate (negative considered control)
% ctrl_idx = num_HF + 1:length(label);
% 
% TN = 0; % true negatives
% for i = ctrl_idx(1):ctrl_idx(end)
%     if label{i} == groundTruth(i)
%         TN = TN + 1;
%     end
% end
% 
% FP = length(ctrl_idx) - TN; % false positive
% FPR = FP / (FP + TN)
% 
% % precision
% precision = TP / (TP + FP)
% 
% %% Interpret network: explain network predictions and identify which part of data network focuses on
% num_plots = 4;
% num_labels = length(label);
% interp_idx = 1:num_labels;
% 
% for i = 1:num_labels
%     X_interp = imread(dataTest.Files{interp_idx(i)});
%     size(X_interp)
%     label_interp = dataTest.Labels(interp_idx(i));
%     % load('label.mat','label_interp')
%     % These only work with image data: 
%     scoreMap = gradCAM(trainednet,X_interp,label_interp); % Doesn't work when the label is 'C' because it's network activation for a given label
% 
%     % Plot overlaid figure
%     % Extract subject number and time window from filename
%     interp_name = dataTest.Files{interp_idx(i)}
% 
%     % Find A identifier in filename
%     Aidx = find(interp_name == 'A');
%     subjectID = str2num(interp_name(Aidx(1) + 1));
%     Sidx = find(interp_name == 'S')
%     sampleID = str2num(interp_name(Sidx + 1));
%     % Check for 2 digit sample number
%     if ~isempty(str2num(interp_name(Sidx + 2)))
%         sampleID = str2num(interp_name(Sidx + 1: Sidx + 2));
%     end
% 
%     subjectID
%     sampleID
% 
% 
%     figure;
%     tiledlayout(1,2,"TileSpacing","compact")
%     nexttile;
%     imshow(interp_name)
% 
%     nexttile;
%     imshow(interp_name)
%     hold on
%     imagesc(scoreMap,AlphaData=1)
%     colormap("jet")
% 
%     set(gcf,'Position',[0 0 800 500])
%     % filename = ['C:\Users\mmgee\Box\Michelle-Gee\Research\Patient-specific models\Auckland_physiology_data\GradCAM-24h-CO-CoBF-split-by-animal-reflect-smooth-1\gradCAM_A' num2str(subjectID) '_S' num2str(sampleID) '.png'];
%     filename = ['/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/PESimages/GradCAM-24h-CO-CoBF-30mwindow-split-by-animal-PDF\gradCAM_A' num2str(subjectID) '_S' num2str(sampleID) '.png'];
%     saveas(gcf,filename)
% end

end
