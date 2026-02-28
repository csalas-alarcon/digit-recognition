function [label, topLabels, topScores] = classifyImageWithGoogleNet(imagePath, K)
    % Default top-K
    if nargin < 2
        K = 5;
    end

    % Load pretrained GoogLeNet
    [net, classNames] = imagePretrainedNetwork("googlenet");

    % Read and resize image
    I = imread(imagePath);
    inputSize = net.Layers(1).InputSize;
    I = imresize(I, inputSize(1:2));

    % Predict scores
    scores = predict(net, single(I));

    % Top-1 label
    label = scores2label(scores, classNames);

    % Top-K labels and scores
    [sortedScores, idx] = maxk(scores, K);
    topLabels = classNames(idx);
    topScores = sortedScores;

    % Display image
    figure
    imshow(I)
    hold on

    % Build multiline text for overlay
    lines = strings(K,1);
    for i = 1:K
        lines(i) = sprintf("%d) %s (%.2f%%)", i, topLabels(i), topScores(i)*100);
    end
    overlayText = strjoin(lines, newline);

    % Draw text on image
    text(10, 20, overlayText, ...
        "Color", "yellow", ...
        "FontSize", 14, ...
        "FontWeight", "bold", ...
        "Interpreter", "none", ...
        "BackgroundColor", "black", ...
        "Margin", 4);

    hold off
end