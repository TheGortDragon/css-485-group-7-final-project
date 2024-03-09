%% load in trained model
% Load the trained model file
load('trained_model.mat');

%% create csv
%{
%get id labels (to use for output file)
testID = readmatrix('data/test.csv', 'Range', 'A2:A7173');

% test network
testResults = zeros(size(testID)); %classification (w forward) of test data

for i = 1:size(testID, 1)
    input = test3D(:, :, i);
    output = cnn.predict(input);
    num = find(output == max(output)) - 1;
    if num >= 9
       num = num + 1;
    end
    testResults(i, 1) = num;
end

% create table and export
columnNames = {'id', 'label'};
fileName = 'convResults.csv';
outputTable = array2table([testID, testResults], 'VariableNames', columnNames);
writetable(outputTable, fileName);
%}

%% validate network performance and abillity w dirty
% setup noise and accuracy
noiseLevels = [0 50 100 200 400];
accuracyMatrix = zeros(length(noiseLevels),1);

% test accuracy at each of the noise levels at same amount at training
for i = 1:length(noiseLevels)
    noiseLevel = noiseLevels(i);
    correctCount = 0;
    % iter through the images in test data
    for k = 1:size(train3D, 3)
        %get patterns
        input = train3D(:, :, k);
        target = label2D(:, k);

        % add noise to input data
        noisyInput = addNoise(input, noiseLevel);

        %classify
        output = cnn.predict(noisyInput);
        output = softmaxToOneHot(output);

        %check correctness
        if isCorrect(output, target)
            correctCount = correctCount + 1;
        end
    end

    accuracyMatrix(i) = (correctCount / size(train3D, 3)) * 100;
end

%% produce figs for doc ?
%accuracy chart
figure;
hold on;
plot(noiseLevels, accuracyMatrix, '-o', 'LineWidth',2);
hold off;
xticks(noiseLevels);
xticklabels({'0', '50', '100', '200', '400'});  % Set custom labels for the x-ticks
grid on;
xlabel('Number Of Pixels Flipped');
ylabel('Classification Accuracy (%)');
title('Network Performance of A Convolutional Neural Network With Noisy Inputs');


%% helper functions
% add noise to an image
function noisyImg = addNoise(img, num)
    % Add noise to the input image by resetting pixels to random values

    % Create a copy of the input image
    noisyImg = img;

    % Get the dimensions of the image
    [rows, cols] = size(img);

    % Calculate the maximum number of pixels to reset based on the noise level
    maxNumPixels = min(floor(0.01 * num * numel(img)), numel(img));

    % Generate random row and column indices for the pixels to reset
    inds = randperm(numel(img), maxNumPixels);
    [row_inds, col_inds] = ind2sub([rows, cols], inds);

    % Reset the selected pixels to random values between 0 and 1
    for i = 1:maxNumPixels
        noisyImg(row_inds(i), col_inds(i)) = rand;
    end
end


%check if the output matches the target
function correct = isCorrect(target, actual)
    % Check if all elements in the vectors are equal, correct one hot code
    if all(target == actual)
        correct = true; % Vectors are equal
    else
        correct = false; % Vectors are not equal
    end
end

function oneHotVector = softmaxToOneHot(softmaxVector)
    % Find the index of the maximum value in the softmax vector
    [~, maxIndex] = max(softmaxVector);

    % Create a one-hot encoded vector
    oneHotVector = zeros(size(softmaxVector));

    % Set the element with the maximum probability to 1
    oneHotVector(maxIndex) = 1;
end
