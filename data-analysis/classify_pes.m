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
[num_slices, num_subjects] = size(PES);

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
imageSize = [541 684 c];
holdout = 0.3;
inputSizeTrain = [541 684 c NaN]; %
inputFormat = 'SSCB';

%% Unpack 2D data and format into imagestore
addpath(genpath('C:\Users\mmgee\Box\Michelle-Gee\Research\Patient-specific models\Auckland_physiology_data'))
gen_location = 'C:\Users\mmgee\Box\Michelle-Gee\Research\Patient-specific models\Auckland_physiology_data\CO_CoBF_1hr\'; % location of image files
count = 1;

% CO and CoBF grid
xx = 0:1:50;
yy = 0:10:400;
[Xpts, Ypts] = meshgrid(xx,yy);

% Sort images into folders for labels
num_HF_animals = 5;
num_ctrl_animals = 4;
numRepeatsHF = num_slices*num_HF_animals;
numRepeatsCtrl = num_slices*num_ctrl_animals;

% for i = 1:num_subjects
%     for j = 1:num_slices
%         P = PES(j,i).F_grid;
%         % Check for inf of -inf vlaues from taking -ln(0) to get PES from PDF
%         P(isinf(P)) = 0;
%         X(:,:,c, count) = normalize(P(:,:,1,1),'range');
%
%         % save image files to location
%         figure;
%         sc = surfc(Ypts, Xpts, X(:,:,c, count));
%         hold on
%         sc(2).EdgeColor = 'w';
%         sc(1).EdgeColor = 'w';
%         view(90,-90)
%         grid off
%         axis off
%         ax = gca;
%         if count < numRepeatsHF
%             location = [gen_location 'HF\'];
%         else
%             location = [gen_location 'C\'];
%         end
%
%         filename = [location 'CO_CoBF_1hr_A' num2str(i) '_S' num2str(j) '.png'];
%         exportgraphics(ax,filename)
%         a = imread(filename);
%         [rows, columns, numberOfColorChannels] = size(a)
%
%         count = count + 1;
%     end
% end
% close all;
imds = imageDatastore(gen_location,'IncludeSubfolders',true,'LabelSource','foldernames');


% Separate to training and test data
[dataTrain, dataTest] = splitEachLabel(imds, 1-holdout, 'randomized');

% Split the training set into training and validation sets for k-fold cross-validation
k = 5; % Number of folds
cvp = cvpartition(dataTrain.Labels, 'KFold', k);
% Access the training, validation, and test sets
for fold = 1:k
    trainIdx = training(cvp, fold);
    valIdx = test(cvp, fold);
    trainFoldImds = subset(dataTrain, trainIdx);
    valFoldImds = subset(dataTrain, valIdx);
end


% Create network
layers = layerGraph();

% Input layer
il = imageInputLayer(imageSize,'Name','input');
% il = inputLayer(inputSizeTrain,inputFormat);

% Main branch
outputChannels = 4;
cv1 = convolution2dLayer(3,outputChannels,'Padding','same','Name','cv1',WeightsInitializer='he');
bn1 = batchNormalizationLayer('Name', 'bn1');
relu1 = reluLayer('Name', 'relu1');
pool1 = maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1');

cv2 = convolution2dLayer(3,outputChannels,'Padding','same','Name','cv2',WeightsInitializer='he');
bn2 = batchNormalizationLayer('Name', 'bn2');
relu2 = reluLayer('Name', 'relu2');
pool2 = maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2');

cv3 = convolution2dLayer(3,outputChannels,'Padding','same','Name','cv3',WeightsInitializer='he');
bn3 = batchNormalizationLayer('Name', 'bn3');
relu3 = reluLayer('Name', 'relu3');
pool3 = maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3');


% Final layers
flatten = flattenLayer('Name', 'flatten');

fc3 = fullyConnectedLayer(numClasses, 'Name', 'fc3');
softmax = softmaxLayer('Name', 'softmax');

% **Construct the layer graph**
layers = addLayers(layers, il);
layers = addLayers(layers, cv1);
layers = addLayers(layers, bn1);
layers = addLayers(layers, relu1);
layers = addLayers(layers, pool1);
layers = addLayers(layers, cv2);
layers = addLayers(layers, bn2);
layers = addLayers(layers, relu2);
layers = addLayers(layers, pool2);
layers = addLayers(layers, cv3);
layers = addLayers(layers, bn3);
layers = addLayers(layers, relu3);
layers = addLayers(layers, pool3);
layers = addLayers(layers,flatten);
layers = addLayers(layers, fc3);
layers = addLayers(layers, softmax);



% **Connect the layers**
layers = connectLayers(layers, 'input', 'cv1');
layers = connectLayers(layers, 'cv1', 'bn1');
layers = connectLayers(layers, 'bn1', 'relu1');
layers = connectLayers(layers, 'relu1','pool1');
layers = connectLayers(layers, 'pool1','cv2');
layers = connectLayers(layers, 'cv2', 'bn2');
layers = connectLayers(layers, 'bn2', 'relu2');
layers = connectLayers(layers, 'relu2','pool2');
layers = connectLayers(layers, 'pool2','cv3');
layers = connectLayers(layers, 'cv3', 'bn3');
layers = connectLayers(layers, 'bn3', 'relu3');
layers = connectLayers(layers, 'relu3', 'pool3');
layers = connectLayers(layers, 'pool3','flatten');
layers = connectLayers(layers, 'flatten', 'fc3');
layers = connectLayers(layers, 'fc3', 'softmax');





% Convert to dlnetwork
net = dlnetwork(layers);


% Train network
lossFcn = "binary-crossentropy"; % for binary classification tasks
options = trainingOptions("rmsprop", ...
    Plots="training-progress", ...
    Metrics="accuracy", ...
    GradientThresholdMethod="l2norm",...
    GradientThreshold=1,...
    MaxEpochs=100,...
    Shuffle="every-epoch",...
    MiniBatchSize=16,...
    InitialLearnRate=0.00001,...
    ValidationData=dataTest,...
    Verbose=true);




% [trainednet, info] = trainnet(dataTrain, net, lossFcn, options)
% save('trainednet.mat','trainednet')
load('trainednet.mat','trainednet')

% This saves gui as image but doesn't seem to work if verbose=true in
% trainnet
% h= findall(groot,'Type','Figure');
% %searching fig tag for Training Progress
% for i=1: size(h,1)
%     if strcmp(h(i).Tag,'DEEPMONITOR_UIFIGURE')
%        savefig(h(i),"TrainedTest.fig");
%        break
%     end
% end

% Classify the test images
num_samples = length(dataTest.Files);
classNames = {'HF','C'};
for i = 1:num_samples
    im = imread(dataTest.Files{i});
    xx = single(im); % makes data single precision
    scores(:,i) = predict(trainednet,xx);
end

% Convert scores to labels
for i = 1:num_samples
    if scores(1,i) > scores(2,i)
        label{i} = 'C';
    else
        label{i} = 'HF';
    end
end

% metrics for test set
% positive is considered heart failure
groundTruth = dataTest.Labels;

for i = 1:length(label)
    if label{i} == groundTruth(i)
        correct_class(i) = 1;
    else
        correct_class(i) = 0;
    end
end

% accuracy
accuracy = sum(correct_class)/length(correct_class)

% recall
num_HF = 0;
for i = 1:length(label)
    if 'HF' == groundTruth(i)
        num_HF = num_HF + 1;
    end
end

TP = 0; % true positives
for i = 1:num_HF
    if label{i} == groundTruth(i)
        TP = TP + 1;
    end
end

FN = num_HF - TP; % false negative
recall = TP/(TP+FN)

% false positive rate (negative considered control)
ctrl_idx = num_HF + 1:length(label);

TN = 0; % true negatives
for i = ctrl_idx(1):ctrl_idx(end)
    if label{i} == groundTruth(i)
        TN = TN + 1;
    end
end

FP = length(ctrl_idx) - TN; % false positive
FPR = FP / (FP + TN)

% precision
precision = TP / (TP + FP)

%% Interpret network: explain network predictions and identify which part of data network focuses on
num_plots = 4;
num_labels = length(label);
interp_idx = 1:num_labels;

for i = 1:num_labels
    X_interp = imread(dataTest.Files{interp_idx(i)});
    label_interp = dataTest.Labels(interp_idx(i));
    load('label.mat','label_interp')
    % These only work with image data: 
    scoreMap = gradCAM(net,X_interp,label_interp); % Doesn't work when the label is 'C' because it's network activation for a given label

    % Plot overlaid figure
    % Extract subject number and time window from filename
    interp_name = dataTest.Files{interp_idx(i)};

    % Find A identifier in filename
    Aidx = find(interp_name == 'A');
    subjectID = str2num(interp_name(Aidx(2) + 1));
    Sidx = find(interp_name == 'S');
    sampleID = str2num(interp_name(Sidx + 1));
    % Check for 2 digit sample number
    if ~isempty(str2num(interp_name(Sidx + 2)))
        sampleID = str2num(interp_name(Sidx + 1: Sidx + 2));
    end


    figure;
    tiledlayout(1,2,"TileSpacing","compact")
    nexttile;
    imshow(interp_name)

    nexttile;
    imshow(interp_name)
    hold on
    imagesc(scoreMap,AlphaData=1)
    colormap("jet")

    set(gcf,'Position',[0 0 800 500])
    filename = ['gradCAM_A' num2str(subjectID) '_S' num2str(sampleID) '.png'];
    saveas(gcf,filename)
end

end
