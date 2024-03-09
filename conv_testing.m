%% get data
%training data -> do we want to shuffle?
trainData = readmatrix('data/train.csv', 'Range', 'C2:ADF27456');
trainLabel = readmatrix('data/train.csv', 'Range', 'B2:B27456');
%testing data
testData = readmatrix('data/test.csv', 'Range', 'B2:ADE7173');
%get id labels (to use for output file)
testID = readmatrix('data/test.csv', 'Range', 'A2:A7173');

%% reformat data (784x1 -> 28x28)
% training data
numImages = size(trainData, 1);
imgSize = [28, 28];
outputSize = 24;
train3D = zeros([imgSize, numImages]);
for i = 1:numImages
    
    if trainLabel(i, 1) == 9 || trainLabel(i, 1) == 25
        train3D(:, :, i) = [];
        continue
    end
    
    train3D(:, :, i) = reshape(trainData(i, :), imgSize);
end

% convert labels to binary
label2D = zeros(outputSize, numImages);
for i = 1:size(trainLabel, 1)
    num = trainLabel(i, 1);
    
    if num == 9 || num == 25
        label2D(:, i) = [];
        continue
    end  
    if num > 9
        num = num - 1;
    end
    
    label2D(num + 1, i) = 1;
end

% test data
numImages = size(testData, 1);
test3D = zeros([imgSize, numImages]);
for i = 1:numImages
    test3D(:, :, i) = reshape(testData(i, :), imgSize);
end

%% normalize data?
% scale to 0-1
% thresholding
train3D = train3D / 255;
train3D(train3D < 0.3) = 0;
train3D(train3D >= 0.7) = 1;

test3D = test3D / 255;
train3D(train3D < 0.3) = 0;
train3D(train3D >= 0.7) = 1;

%% setup network
cnn = CNN();

%% train network
cnn.train(train3D(:,:,:), label2D(:,:), 3, 12, outputSize, 20, .01);

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

        %check correctness
        if isCorrect(output, target)
            correctCount = correctCount + 1;
        end
    end

    accuracyMatrix(i) = (correctCount / size(train3D, 3)) * 100;
end

%% test network
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

%% create table and export
columnNames = {'id', 'label'};
fileName = 'convResults.csv';
outputTable = array2table([testID, testResults], 'VariableNames', columnNames);
writetable(outputTable, fileName);

%% helper functions

% addNoise to a vector, distort it 
function pvec = addNoise(pvec, num)
    % ADDNOISE Add noise to "binary" vector
    % pvec pattern vector (-1 and 1)
    % num number of elements to flip randomly
    % Handle special case where there's no noise
    if num == 0
        return;
    end
    % first, generate a random permutation of all indices into pvec
    inds = randperm(length(pvec));
    % then, use the first n elements to flip pixels
    pvec(inds(1:num)) = -pvec(inds(1:num));
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