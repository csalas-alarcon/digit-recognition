% 1. Descomprimimos y cargamos la Información
unzip("DigitsData.zip");
dataFolder = "DigitsData";

% imageDatastore automáticamente pone labels a la imágenes en base al
% directorio.
% Crea un objeto imageDatastore del DEeep Learning Toolbox
imds = imageDatastore(dataFolder, ... % Empieza a buscar en "DigitsData"
    'IncludeSubfolders', true, ... % Incluye subdirectorios
    'LabelSource', 'foldernames'); % Asigna label como Nom. Directorio

% Divide: 70% para Entrenamiento, 30% para Validación
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

% Definimos las Capas de la Red / layers ~ capas
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer % <--- The stabilizer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    fullyConnectedLayer(10) 
    softmaxLayer
    classificationLayer];

% El Entrenamiento
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress'); % Esto enseña un grafo

net = trainNetwork(imdsTrain, layers, options);

% Validacion
YPred = classify(net, imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);