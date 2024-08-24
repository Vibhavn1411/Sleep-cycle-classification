% Load the feature matrix
load('featureMat.mat', 'featureMatrix');

% Separate features and labels
X = featureMatrix(:, 1:(end - 1));
Y = featureMatrix(:, end);
X(isnan(X)) = 0;
X(isinf(X)) = 0;

% Remove class 7 (if applicable)
k = (Y == 7);
X(k, :) = [];
Y(k, :) = [];

% Balance the classes using oversampling
classes = unique(Y);
maxClassSize = max(histc(Y, classes));
X_balanced = [];
Y_balanced = [];

for i = 1:numel(classes)
    Xi = X(Y == classes(i), :);
    Yi = Y(Y == classes(i));
    reps = maxClassSize - numel(Yi);
    if reps > 0
        Xi = [Xi; repmat(Xi, ceil(reps / numel(Yi)), 1)];
        Yi = [Yi; repmat(Yi, ceil(reps / numel(Yi)), 1)];
    end
    X_balanced = [X_balanced; Xi(1:maxClassSize, :)];
    Y_balanced = [Y_balanced; Yi(1:maxClassSize)];
end

% Shuffle the balanced data
idx = randperm(size(X_balanced, 1));
X_balanced = X_balanced(idx, :);
Y_balanced = Y_balanced(idx);

% Split the data into training (70%), validation (15%), and testing (15%)
cv = cvpartition(size(X_balanced, 1), 'HoldOut', 0.3);
idxTrain = ~cv.test;

% Further split the held-out data into validation (50% of 30%) and testing (50% of 30%)
cv2 = cvpartition(sum(cv.test), 'HoldOut', 0.5);
idxValidation = cv.test;
idxValidation(cv2.test) = false;
idxTest = cv.test;
idxTest(~cv2.test) = false;

trainLabel = Y_balanced(idxTrain);
trainData = X_balanced(idxTrain, :);
validationLabel = Y_balanced(idxValidation);
validationData = X_balanced(idxValidation, :);
testLabel = Y_balanced(idxTest);
testData = X_balanced(idxTest, :);

% Calculate class weights and adjust them more aggressively
classWeights = 1 ./ histc(Y_balanced, classes);
classWeights = classWeights / sum(classWeights);
classWeights = classWeights .^ 2; % Apply a stronger power to emphasize minority classes

% Hyperparameter options
numTrees = 200; % Further increased number of trees
maxNumSplits = 40; % Max splits
minLeafSize = 10; % Min leaf size
numEpochs = 50; % Number of epochs for training simulation

% Initialize arrays to store accuracy and loss over epochs
trainAccuracy = zeros(numEpochs, 1);
validationAccuracy = zeros(numEpochs, 1);
trainLoss = zeros(numEpochs, 1);
validationLoss = zeros(numEpochs, 1);

% Train the model over epochs and collect accuracy/loss metrics
for epoch = 1:numEpochs
    % Train model using updated parameters
    model = fitcensemble(trainData, trainLabel, 'Method', 'Bag', ...
        'NumLearningCycles', epoch, ...
        'Learners', templateTree('MinLeafSize', minLeafSize, 'MaxNumSplits', maxNumSplits), ...
        'Weights', classWeights(trainLabel));
    
    % Training accuracy and loss
    trainPred = predict(model, trainData);
    trainAccuracy(epoch) = sum(trainPred == trainLabel) / numel(trainLabel);
    trainLoss(epoch) = 1 - trainAccuracy(epoch); % Simplified loss calculation
    
    % Validation accuracy and loss
    valPred = predict(model, validationData);
    validationAccuracy(epoch) = sum(valPred == validationLabel) / numel(validationLabel);
    validationLoss(epoch) = 1 - validationAccuracy(epoch); % Simplified loss calculation
end

% Plot accuracy vs. number of epochs
figure;
plot(1:numEpochs, trainAccuracy, '-o', 'DisplayName', 'Training Accuracy', 'LineWidth', 1);
hold on;
plot(1:numEpochs, validationAccuracy, '-o', 'DisplayName', 'Validation Accuracy', 'LineWidth', 1);
xlabel('Epoch');
ylabel('Accuracy');
title('Accuracy vs. Epoch');
legend('Location', 'Best');
grid on;
hold off;

% Plot loss vs. number of epochs
figure;
plot(1:numEpochs, trainLoss, '-o', 'DisplayName', 'Training Loss', 'LineWidth', 1);
hold on;
plot(1:numEpochs, validationLoss, '-o', 'DisplayName', 'Validation Loss', 'LineWidth', 1);
xlabel('Epoch');
ylabel('Loss');
title('Loss vs. Epoch');
legend('Location', 'Best');
grid on;
hold off;

% Evaluate the final model on test data
[testPred, ~] = predict(model, testData);
testAccuracy = sum(testPred == testLabel) / numel(testLabel);
disp(['Test Accuracy: ', num2str(testAccuracy)]);

% Display the number of datasets for training, validation, and testing
disp(['Number of training datasets: ', num2str(sum(idxTrain))]);
disp(['Number of validation datasets: ', num2str(sum(idxValidation))]);
disp(['Number of testing datasets: ', num2str(sum(idxTest))]);

% Calculate confusion matrix
confusionMatTest = confusionmat(testLabel, testPred);
figure;
confusionchart(confusionMatTest, {'NREM1', 'NREM2', 'NREM3', 'NREM4', 'REM', 'AWAKE'});
title('Confusion Matrix for Test Data');

% Classification metrics
numClasses = numel(unique(testLabel));
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1score = zeros(numClasses, 1);
support = zeros(numClasses, 1);

for i = 1:numClasses
    tp = sum((testPred == i) & (testLabel == i));
    fp = sum((testPred == i) & (testLabel ~= i));
    fn = sum((testPred ~= i) & (testLabel == i));
    
    precision(i) = tp / (tp + fp);
    recall(i) = tp / (tp + fn);
    f1score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
    support(i) = sum(testLabel == i);
end

classificationTable = table(precision, recall, f1score, support, ...
    'VariableNames', {'Precision', 'Recall', 'F1_Score', 'Support'}, ...
    'RowNames', {'AWAKE','NREM1', 'NREM2', 'NREM3', 'NREM4', 'REM'});

disp('Classification Table:');
disp(classificationTable);

macroPrecision = mean(precision);
macroRecall = mean(recall);
macroF1Score = mean(f1score);

weightedPrecision = sum(precision .* support) / sum(support);
weightedRecall = sum(recall .* support) / sum(support);
weightedF1Score = sum(f1score .* support) / sum(support);

% Display the results
disp(['Macro Average Precision: ', num2str(macroPrecision)]);
disp(['Weighted Average Precision: ', num2str(weightedPrecision)]);

% Create epoch metrics table
epochMatrix = table((1:numEpochs)', trainAccuracy, validationAccuracy, trainLoss, validationLoss, ...
    'VariableNames', {'Epoch', 'TrainingAccuracy', 'ValidationAccuracy', 'TrainingLoss', 'ValidationLoss'});

disp('Epoch Metrics Matrix:');
disp(epochMatrix);
