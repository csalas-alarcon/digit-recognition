% 1. Unzip and load data
unzip("DigitsData.zip");
dataFolder = "DigitsData";

% imageDatastore automatically labels images based on the folder name
imds = imageDatastore(dataFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split: 70% for training, 30% for testing your accuracy
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

% We Define Layers
layers = [
    imageInputLayer([28 28 1]) % Input: 28x28 grayscale images
    
    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10) % 10 output classes (0-9)
    softmaxLayer
    classificationLayer];

% The Actual Training
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress'); % This opens a cool live graph

net = trainNetwork(imdsTrain, layers, options);

% Testing
YPred = classify(net, imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);