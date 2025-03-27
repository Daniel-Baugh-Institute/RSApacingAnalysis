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
c = 1; % one channel, potential energy
numSamples = num_subjects * num_slices;
numClasses = 2; % normal and heart failure
imageSize = size(PES(1,1).F_grid);
holdout = 0.3;
inputSizeTrain = [imageSize(1:2) c round(numSamples*(1-holdout))];
inputFormat = 'SSCB';

% X = zeros(imageSize(1),imageSize(2),imageSize(3),imageSize(4),c,numSamples);

%% unpack 2D data and format into numeric array for dlarray object
% count = 1;
% for i = 1:num_subjects
%     for j = 1:num_slices        
%         P = PES(j,i).F_grid;
%         % Check for inf of -inf vlaues from taking -ln(0) to get PES from PDF
%         P(isinf(P)) = 0;
%         X(:,:,c, count) = normalize(P(:,:,1,1),'range');
%         count = count + 1;
%     end
% end
% size(X(:,:,1,1))

% Cross validation (train: 70%, test: 30%)
cv = cvpartition(size(X,4),'HoldOut',holdout);
idx = cv.test;
% Separate to training and test data
dataTrain = X(:,:,:,~idx);
dataTest  = X(:,:,:,idx);
dlX = dlarray(dataTrain,inputFormat);
dlX_test = dlarray(dataTest,inputFormat);

% Reindex targets to match
num_HF_animals = 5;
num_ctrl_animals = 4;
numRepeatsHF = num_slices*num_HF_animals;
numRepeatsCtrl = num_slices*num_ctrl_animals;
targets = categorical([repmat({'HF'}, numRepeatsHF, 1); repmat({'C'}, numRepeatsCtrl, 1)]);
targetsTrain = targets(~idx);
targetsTest = targets(idx);
% validationData = table('dataTest','targetsTest')

layers = layerGraph();

% Input layer
il = inputLayer(inputSizeTrain,inputFormat);

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
% layers = addLayers(layers, conv_skip);
% layers = addLayers(layers, add1);


% **Connect the layers**
layers = connectLayers(layers, 'input', 'cv1');
layers = connectLayers(layers, 'cv1', 'bn1');
layers = connectLayers(layers, 'bn1', 'relu1');
layers = connectLayers(layers, 'relu1','pool1');
% layers = connectLayers(layers, 'relu1', 'conv_skip');
% layers = connectLayers(layers, 'conv_skip', 'add1/in1');
layers = connectLayers(layers, 'pool1','cv2');
layers = connectLayers(layers, 'cv2', 'bn2');
layers = connectLayers(layers, 'bn2', 'relu2');
layers = connectLayers(layers, 'relu2','pool2');
layers = connectLayers(layers, 'pool2','cv3');
layers = connectLayers(layers, 'cv3', 'bn3');
layers = connectLayers(layers, 'bn3', 'relu3');
layers = connectLayers(layers, 'relu3', 'pool3');
% layers = connectLayers(layers, 'add1', 'pool3');
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
    MaxEpochs=2500,...
    Shuffle="every-epoch",...
    MiniBatchSize=16,...
    InitialLearnRate=0.00001,...
    Verbose=true);



disp('dlX size')
size(dlX)
[trainednet, info] = trainnet(dlX, targetsTrain, net, lossFcn, options)



% Classify the test images
scores = predict(trainednet, dlX_test);
[num_labels,num_samples] = size(scores);
for i = 1:num_samples
    if scores(1,i) > scores(2,i)
        label{i} = 'C';
    else
        label{i} = 'HF';
    end
end

% metrics for test set
% positive is considered heart failure
label
targetsTest

for i = 1:length(label)
    if label{i} == targetsTest(i)
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
    if 'HF' == targetsTest(i)
        num_HF = num_HF + 1;
    end
end

TP = 0; % true positives
for i = 1:num_HF
    if label{i} == targetsTest(i)
        TP = TP + 1;
    end
end

FN = num_HF - TP; % false negative
recall = TP/(TP+FN)

% false positive rate (negative considered control)
ctrl_idx = num_HF + 1:length(label);

TN = 0; % true negatives
for i = ctrl_idx(1):ctrl_idx(end)
    if label{i} == targetsTest(i)
        TN = TN + 1;
    end
end

FP = length(ctrl_idx) - TN; % false positive
FPR = FP / (FP + TN)

% precision
precision = TP / (TP + FP)

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

for i = 1:num_subjects
    for j = 1:num_slices
        P = PES(j,i).F_grid;
        % Check for inf of -inf vlaues from taking -ln(0) to get PES from PDF
        P(isinf(P)) = 0;
        X(:,:,c, count) = normalize(P(:,:,1,1),'range');
        % save image files to location
        figure;
        sc = surfc(Ypts, Xpts, X(:,:,c, count));
        hold on
        sc(2).EdgeColor = 'w';
        sc(1).EdgeColor = 'w';
        view(90,-90)
        grid off
        axis off
        ax = gca;
        if count > numRepeatsHF
            location = [gen_location 'HF'];
        else
            location = [gen_location 'C'];
        end

        filename = [location 'CO_CoBF_1hr_A' num2str(i) '_S' num2str(j) '.png'];
        exportgraphics(ax,filename)

        count = count + 1;
    end
end

imds = imageDatastore(gen_location,'IncludeSubfolders',true,'LabelSource','foldernames');


% Separate to training and test data
[dataTrain, dataTest] = splitEachLabel(imds, 0.7, 'randomized');
dataTrain = X(:,:,:,~idx);
dataTest  = X(:,:,:,idx);
dlX = dlarray(dataTrain,inputFormat);
dlX_test = dlarray(dataTest,inputFormat);

% Reindex targets to match
num_HF_animals = 5;
num_ctrl_animals = 4;
numRepeatsHF = num_slices*num_HF_animals;
numRepeatsCtrl = num_slices*num_ctrl_animals;
targets = categorical([repmat({'HF'}, numRepeatsHF, 1); repmat({'C'}, numRepeatsCtrl, 1)]);
targetsTrain = targets(~idx);
targetsTest = targets(idx);
% validationData = table('dataTest','targetsTest')



layers = layerGraph();

% Input layer
il = inputLayer(inputSizeTrain,inputFormat);

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

% Skip layer
% conv_skip = convolution2dLayer(1, outputChannels, 'Padding', 'same', 'Name', 'conv_skip');
% add1 = additionLayer(2, 'Name', 'add1');

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
% layers = addLayers(layers, conv_skip);
% layers = addLayers(layers, add1);


% **Connect the layers**
layers = connectLayers(layers, 'input', 'cv1');
layers = connectLayers(layers, 'cv1', 'bn1');
layers = connectLayers(layers, 'bn1', 'relu1');
layers = connectLayers(layers, 'relu1','pool1');
% layers = connectLayers(layers, 'relu1', 'conv_skip');
% layers = connectLayers(layers, 'conv_skip', 'add1/in1');
layers = connectLayers(layers, 'pool1','cv2');
layers = connectLayers(layers, 'cv2', 'bn2');
layers = connectLayers(layers, 'bn2', 'relu2');
layers = connectLayers(layers, 'relu2','pool2');
layers = connectLayers(layers, 'pool2','cv3');
layers = connectLayers(layers, 'cv3', 'bn3');
layers = connectLayers(layers, 'bn3', 'relu3');
layers = connectLayers(layers, 'relu3', 'pool3');
% layers = connectLayers(layers, 'add1', 'pool3');
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
    MaxEpochs=2500,...
    Shuffle="every-epoch",...
    MiniBatchSize=16,...
    InitialLearnRate=0.00001,...
    Verbose=true);
% Not an option for adam:     GradientTolerance=1e-5, ...     MaxIterations=1000, ...

    
% monitor = trainingProgressMonitor;
% monitor.Info = ["LearningRate","Epoch","Iteration"];
% monitor.Metrics = ["TrainingLoss"];
% monitor.XLabel = "Iteration";
% groupSubPlot(monitor,"Loss",["TrainingLoss","ValidationLoss"]);
% groupSubPlot(monitor,"Accuracy",["TrainingAccuracy","ValidationAccuracy"]);
% yscale(monitor,"Loss","log")


disp('dlX size')
size(dlX)
[trainednet, info] = trainnet(dlX, targetsTrain, net, lossFcn, options)

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
scores = predict(trainednet, dlX_test);
[num_labels,num_samples] = size(scores);
for i = 1:num_samples
    if scores(1,i) > scores(2,i)
        label{i} = 'C';
    else
        label{i} = 'HF';
    end
end

% metrics for test set
% positive is considered heart failure
label
targetsTest

for i = 1:length(label)
    if label{i} == targetsTest(i)
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
    if 'HF' == targetsTest(i)
        num_HF = num_HF + 1;
    end
end

TP = 0; % true positives
for i = 1:num_HF
    if label{i} == targetsTest(i)
        TP = TP + 1;
    end
end

FN = num_HF - TP; % false negative
recall = TP/(TP+FN)

% false positive rate (negative considered control)
ctrl_idx = num_HF + 1:length(label);

TN = 0; % true negatives
for i = ctrl_idx(1):ctrl_idx(end)
    if label{i} == targetsTest(i)
        TN = TN + 1;
    end
end

FP = length(ctrl_idx) - TN; % false positive
FPR = FP / (FP + TN)

% precision
precision = TP / (TP + FP)

%% Interpret network: explain network predictions and identify which part of data network focuses on
% interp_idx = randi(num_labels,1);
% X_interp = dataTest(:,:,1,interp_idx);

% These only work with image data: scoreMap =
% occlusionSensitivity(net,X_interp,1); scoreMap = imageLIME(net,X_interp,1)
% scoreMap = gradCAM(net,X_interp,label{interp_idx});  
% 
% % Plot overlaid figure
% figure;
% surf(dataTest(:,:,1,interp_idx))
% hold on
% imagesc(scoreMap,AlphaData=0.5)
% saveas(gcf,'gradCAM.png')


%% 4D data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TODO: USE AUTOENCODER TO REDUCE DATA DIMENSION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Dataset: load('C:\Users\mmgee\Downloads\PDF_test.mat')
% % Add path
% % addpath(genpath('/lustre/ogunnaike/users/2420/matlab_example/NZ-physiology-data/'))
% [num_slices, num_subjects] = size(PES);
% 
% % Train the network Network options: while it would be nice to take
% % advantage of the time series nature of the data, any neural net that does
% % would likely only capture circadian rhythm. Unless this is thrown off in
% % heart failure?? (It is, see
% % https://www.ahajournals.org/doi/10.1161/CIRCRESAHA.122.321369) These
% % types of networks include LSTM (see sequence classification using deep
% % learning matlab. This looks like it classifies time series waveforms) and
% % vector sequence classification networks
% % For implementation see: https://www.mathworks.com/help/deeplearning/ref/dlarray.html
% 
% % Instead use deep learning for images
% 
% 
% % X format for 3D image: h-by-w-by-d-by-c-by-n numeric array, where h, w,
% % d, c and n are the height, width, depth, number of channels of the
% % images, and number of image observations, respectively.
% c = 1; % one channel, potential energy
% numSamples = num_subjects * num_slices;
% imageSize = size(PES(1,1).F_grid);
% X = zeros(imageSize(1),imageSize(2),imageSize(3),imageSize(4),c,numSamples);
% 
% % unpack 4D data and format into numeric array for dlarray object
% count = 1;
% for i = 1:num_subjects
%     for j = 1:num_slices        
%         P = PES(j,i).F_grid;
%         X(:,:,:,:,c,count) = P;
%         count = count + 1;
%     end
% end
% 
% fmt = "SSSSCB";
% dlX = dlarray(X,fmt);
% 
% % Create pretrained residual network CNN
% numClasses = 2; % normal and heart failure
% 
% net = resnetNetwork(imageSize,numClasses); % function only available after r2024a
% %net = replaceLayer(net,'fc1000',newLayers); % modify last layer to
% %suit classification task
% 
% % Train network
% lossFcn = "binary-crossentropy"; % for binary classification tasks
% options = trainingOptions("lbfgs");
% net = trainnet(dlX, net, lossFcn, options);
% 
% % Classify the test images
% % predictedLabels = classify(net, stackedTestImages);
% 
% % compare predicted to ground truth labels
% % R2024b: accuracy = testnet(net,XTest,labelsTest,"accuracy")

end
