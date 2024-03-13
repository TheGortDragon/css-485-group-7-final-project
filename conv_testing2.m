%% copy all code from conv_testing and create cross entropy graphs for the 9 different networks:
%% load in image and label data
load('trainingData.mat');
load('trainingLabels.mat');
outputSize = 24;

%% setup and train networks
% kernel size 1x1, channel size 3
cnn13 = CNN();
cnn13.train(trainingData(:,:,:), trainingLabels(:,:), 1, 3, outputSize, 7, .01);
save('cnn13.mat', 'cnn13');

% kernel size 1x1, channel size 6
cnn16 = CNN();
cnn16.train(trainingData(:,:,:), trainingLabels(:,:), 1, 6, outputSize, 7, .01);
save('cnn16.mat', 'cnn16');

% kernel size 1x1, channel size 12
cnn112 = CNN();
cnn112.train(trainingData(:,:,:), trainingLabels(:,:), 1, 12, outputSize, 7, .01);
save('cnn112.mat', 'cnn112');

% kernel size 3x3, channel size 3
cnn33 = CNN();
cnn33.train(trainingData(:,:,:), trainingLabels(:,:), 3, 3, outputSize, 7, .01);
save('cnn33.mat', 'cnn33');

% kernel size 3x3, channel size 6
cnn36 = CNN();
cnn36.train(trainingData(:,:,:), trainingLabels(:,:), 3, 6, outputSize, 7, .01);
save('cnn36.mat', 'cnn36');

% kernel size 3x3, channel size 12
cnn312 = CNN();
cnn312.train(trainingData(:,:,:), trainingLabels(:,:), 3, 12, outputSize, 7, .01);
save('cnn312.mat', 'cnn312');

% kernel size 5x5, channel size 3
cnn53 = CNN();
cnn53.train(trainingData(:,:,:), trainingLabels(:,:), 5, 3, outputSize, 7, .01);
save('cnn53.mat', 'cnn53');

% kernel size 5x5, channel size 6
cnn56 = CNN();
cnn56.train(trainingData(:,:,:), trainingLabels(:,:), 5, 6, outputSize, 7, .01);
save('cnn56.mat', 'cnn56');

% kernel size 5x5, channel size 12
cnn512 = CNN();
cnn512.train(trainingData(:,:,:), trainingLabels(:,:), 5, 12, outputSize, 7, .01);
save('cnn512.mat', 'cnn512');