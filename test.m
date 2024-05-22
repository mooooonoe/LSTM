clc; clear; close all;

load WaveformData

data(1:4);
numChannels = size(data{1},2);

figure
tiledlayout(2,2)
for i = 1:4
    nexttile
    stackedplot(data{i})

    xlabel("Time stpe")
end

%% data set
numObservations = numel(data);
idxTrain = 1:floor(0.9*numObservations);
idxTest = floor(0.9*numObservations)+1:numObservations;
dataTrain = data(idxTrain);
dataTest = data(idxTest);

%% prepare data
numObservationsTrain = numel(dataTrain);
XTrain = cell(numObservationsTrain,1);
TTrain = cell(numObservationsTrain,1);

for n = 1:numObservationsTrain
    X = dataTrain{n};
    XTrain{n} = X(1:end-1,:);
    TTrain{n} = X(2:end,:);
end

muX = mean(cell2mat(XTrain));
sigmaX = std(cell2mat(XTrain),0);

muT = mean(cell2mat(TTrain));
sigmaT = std(cell2mat(TTrain),0);

for n = 1:numel(XTrain)
    XTrain{n} = (XTrain{n} - muX) ./ sigmaX;
    TTrain{n} = (TTrain{n} - muT) ./ sigmaT;
end

%% define lstm
layers = [
    sequenceInputLayer(numChannels)
    lstmLayer(128)
    fullyConnectedLayer(numChannels)];

options = trainingOptions("adam", ...
    MaxEpochs=200, ...
    SequencePaddingDirection="left", ...
    Shuffle="every-epoch", ...
    Plots="training-progress", ...
    Verbose=false);

%% train
net = trainnet(XTrain,TTrain,layers,"mse",options);

%% test
numObservationsTest = numel(dataTest);
XTest = cell(numObservationsTest,1);
TTest = cell(numObservationsTest,1);
for n = 1:numObservationsTest
    X = dataTest{n};
    XTest{n} = (X(1:end-1,:) - muX) ./ sigmaX;
    TTest{n} = (X(2:end,:) - muT) ./ sigmaT;
end

YTest = minibatchpredict(net,XTest, ...
    SequencePaddingDirection="left", ...
    UniformOutput=false);

for n = 1:numObservationsTest
    T = TTest{n};

    sequenceLength = size(T,1);    

    Y = YTest{n}(end-sequenceLength+1:end,:);

    err(n) = rmse(Y,T,"all");
end

figure
histogram(err)
xlabel("RMSE")
ylabel("Frequency")

mean(err,"all")

%% forecast future time steps
idx = 2;
X = XTest{idx};
T = TTest{idx};

figure
stackedplot(X,DisplayLabels="Channel " + (1:numChannels))
xlabel("Time Step")
title("Test Observation " + idx)

%% open loop forecasting
net = resetState(net);
offset = 75;
[Z,state] = predict(net,X(1:offset,:));
net.State = state;

numTimeSteps = size(X,1);
numPredictionTimeSteps = numTimeSteps - offset;
Y = zeros(numPredictionTimeSteps,numChannels);
Y(1,:) = Z(end,1);

for t = 1:numPredictionTimeSteps-1
    Xt = X(offset+t,:);
    [Y(t+1,:),state] = predict(net,Xt);
    net.State = state;
end

figure
t = tiledlayout(numChannels,1);
title(t,"Open Loop Forecasting")

for i = 1:numChannels
    nexttile
    plot(X(:,i))
    hold on
    plot(offset:numTimeSteps,[X(offset,i) Y(:,i)'],"--")
    ylabel("Channel " + i)
end

xlabel("Time Step")
nexttile(1)
legend(["Input" "Forecasted"])

%% close loop
net = resetState(net);
offset = size(X,1);
[Z,state] = predict(net,X(1:offset,:));
net.State = state;

numPredictionTimeSteps = 200;
Y = zeros(numPredictionTimeSteps,numChannels);
Y(1,:) = Z(end,:);

for t = 2:numPredictionTimeSteps
    [Y(t,:),state] = predict(net,Y(t-1,:));
    net.State = state;
end

numTimeSteps = offset + numPredictionTimeSteps;

figure
t = tiledlayout(numChannels,1);
title(t,"Closed Loop Forecasting")

for i = 1:numChannels
    nexttile
    plot(X(1:offset,i))
    hold on
    plot(offset:numTimeSteps,[X(offset,i) Y(:,i)'],"--")
    ylabel("Channel " + i)
end

xlabel("Time Step")
nexttile(1)
legend(["Input" "Forecasted"])