clc; clear; close all;

% Load your waveform data
load WaveformData

% Split data into training and testing sets
numObservations = numel(data);
idxTrain = 1:floor(0.9*numObservations);
idxTest = floor(0.9*numObservations)+1:numObservations;
dataTrain = data(idxTrain);
dataTest = data(idxTest);

% Prepare training data
numObservationsTrain = numel(dataTrain);
XTrain = cell(numObservationsTrain,1);
TTrain = cell(numObservationsTrain,1);
for n = 1:numObservationsTrain
    X = dataTrain{n};
    XTrain{n} = X(1:end-1,:);
    TTrain{n} = X(2:end,:);
end

% Initialize LSTM parameters
numChannels = size(data{1},2);
numHiddenUnits = 128;
[lstmParams, state] = initializeLSTM(numChannels, numHiddenUnits);

% Training parameters
learningRate = 0.01;
numEpochs = 200;
for epoch = 1:numEpochs
    for n = 1:numObservationsTrain
        [grads, loss, state] = lstmForwardBackward(lstmParams, XTrain{n}, TTrain{n}, state);
        lstmParams = updateLSTM(lstmParams, grads, learningRate);
    end
end

% Testing and error calculation
numObservationsTest = numel(dataTest);
errors = zeros(numObservationsTest,1);
for n = 1:numObservationsTest
    [predicted, ~] = lstmForward(lstmParams, dataTest{n}(1:end-1,:), state);
    errors(n) = sqrt(mean((predicted - dataTest{n}(2:end,:)).^2, 'all'));
end

% Visualize errors
figure
histogram(errors)
xlabel('RMSE')
ylabel('Frequency')
title('Test Error Distribution')

% Display mean error
meanError = mean(errors);
disp(['Mean Error: ', num2str(meanError)]);

%% Helper functions to initialize, update, and run the LSTM
function [params, initState] = initializeLSTM(inputDim, hiddenUnits)
    % Xavier initialization for weights
    params.Wf = randn(hiddenUnits, inputDim + hiddenUnits) * sqrt(2/(inputDim + hiddenUnits));
    params.Wi = randn(hiddenUnits, inputDim + hiddenUnits) * sqrt(2/(inputDim + hiddenUnits));
    params.Wc = randn(hiddenUnits, inputDim + hiddenUnits) * sqrt(2/(inputDim + hiddenUnits));
    params.Wo = randn(hiddenUnits, inputDim + hiddenUnits) * sqrt(2/(inputDim + hiddenUnits));
    params.bf = zeros(hiddenUnits, 1);
    params.bi = zeros(hiddenUnits, 1);
    params.bc = zeros(hiddenUnits, 1);
    params.bo = zeros(hiddenUnits, 1);
    params.Wy = randn(inputDim, hiddenUnits) * sqrt(2/(hiddenUnits));
    params.by = zeros(inputDim, 1);
    initState.h = zeros(hiddenUnits, 1);
    initState.c = zeros(hiddenUnits, 1);
end

function [grads, loss, state] = lstmForwardBackward(params, X, T, state)
    % Forward pass
    [Y, state, cache] = lstmForward(params, X, state);
    
    % Compute loss
    loss = mean((Y - T).^2, 'all');
    
    % Backward pass
    grads = lstmBackward(params, X, T, Y, state, cache);
end

function params = updateLSTM(params, grads, learningRate)
    fields = fieldnames(params);
    for i = 1:numel(fields)
        params.(fields{i}) = params.(fields{i}) - learningRate * grads.(fields{i});
    end
end

function [Y, state, cache] = lstmForward(params, X, state)
    [numTimeSteps, inputDim] = size(X);
    hiddenUnits = size(params.Wf, 1);
    
    h = state.h;
    c = state.c;
    
    Y = zeros(numTimeSteps, inputDim);
    cache = struct();
    cache.h = zeros(numTimeSteps, hiddenUnits);
    cache.c = zeros(numTimeSteps, hiddenUnits);
    
    for t = 1:numTimeSteps
        xt = X(t, :)';
        combined = [xt; h];
        
        ft = sigmoid(params.Wf * combined + params.bf);
        it = sigmoid(params.Wi * combined + params.bi);
        ct = ft .* c + it .* tanh(params.Wc * combined + params.bc);
        ot = sigmoid(params.Wo * combined + params.bo);
        ht = ot .* tanh(ct);
        
        yt = params.Wy * ht + params.by;
        
        Y(t, :) = yt';
        h = ht;
        c = ct;
        
        cache.h(t, :) = h';
        cache.c(t, :) = c';
        cache.ft(t, :) = ft';
        cache.it(t, :) = it';
        cache.ot(t, :) = ot';
        cache.ctilde(t, :) = tanh(params.Wc * combined + params.bc)';
        cache.combined(t, :) = combined';
    end
    
    state.h = h;
    state.c = c;
end

function grads = lstmBackward(params, X, T, Y, state, cache)
    [numTimeSteps, inputDim] = size(X);
    hiddenUnits = size(params.Wf, 1);
    
    dWf = zeros(size(params.Wf));
    dWi = zeros(size(params.Wi));
    dWc = zeros(size(params.Wc));
    dWo = zeros(size(params.Wo));
    dbf = zeros(size(params.bf));
    dbi = zeros(size(params.bi));
    dbc = zeros(size(params.bc));
    dbo = zeros(size(params.bo));
    dWy = zeros(size(params.Wy));
    dby = zeros(size(params.by));
    
    dh_next = zeros(hiddenUnits, 1);
    dc_next = zeros(hiddenUnits, 1);
    
    for t = numTimeSteps:-1:1
        xt = X(t, :)';
        combined = cache.combined(t, :)';
        
        dy = Y(t, :)' - T(t, :)';
        
        dWy = dWy + dy * cache.h(t, :)';
        dby = dby + dy;
        
        dh = params.Wy' * dy + dh_next;
        do = dh .* tanh(cache.c(t, :)');
        dot = do .* cache.ot(t, :)' .* (1 - cache.ot(t, :)');
        
        dc = dh .* cache.ot(t, :)' .* (1 - tanh(cache.c(t, :)').^2) + dc_next;
        df = dc .* state.c .* cache.ft(t, :)' .* (1 - cache.ft(t, :)');
        di = dc .* cache.ctilde(t, :)' .* cache.it(t, :)' .* (1 - cache.it(t, :)');
        dctilde = dc .* cache.it(t, :)' .* (1 - cache.ctilde(t, :)'.^2);
        
        dWf = dWf + df * combined';
        dbf = dbf + df;
        dWi = dWi + di * combined';
        dbi = dbi + di;
        dWc = dWc + dctilde * combined';
        dbc = dbc + dctilde;
        dWo = dWo + dot * combined';
        dbo = dbo + dot;
        
        dcombined = params.Wf' * df + params.Wi' * di + params.Wc' * dctilde + params.Wo' * dot;
        dh_next = dcombined(hiddenUnits+1:end);
        dc_next = dc .* cache.ft(t, :)';
    end
    
    grads.Wf = dWf;
    grads.Wi = dWi;
    grads.Wc = dWc;
    grads.Wo = dWo;
    grads.bf = dbf;
    grads.bi = dbi;
    grads.bc = dbc;
    grads.bo = dbo;
    grads.Wy = dWy;
    grads.by = dby;
end

function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

function y = sigmoid_derivative(x)
    y = sigmoid(x) .* (1 - sigmoid(x));
end

function y = tanh_derivative(x)
    y = 1 - tanh(x).^2;
end

idx = 2;
X = dataTest{idx}(1:end-1,:);
T = dataTest{idx}(2:end,:);

figure
stackedplot(X, 'DisplayLabels', "Channel " + (1:numChannels))
xlabel("Time Step")
title("Test Observation " + idx)

offset = 75;
[Z, state] = lstmForward(lstmParams, X(1:offset,:), state);

numTimeSteps = size(X,1);
numPredictionTimeSteps = numTimeSteps - offset;
Y = zeros(numPredictionTimeSteps, numChannels);
Y(1,:) = Z(end, :);

for t = 2:numPredictionTimeSteps
    [Y(t,:), state] = lstmForward(lstmParams, Y(t-1,:), state);
end

figure
t = tiledlayout(numChannels,1);
title(t,"Open Loop Forecasting")

for i = 1:numChannels
    nexttile
    plot(X(:,i))
    hold on
    plot(offset:numTimeSteps,[X(offset,i); Y(:,i)],'--')
    ylabel("Channel " + i)
end

xlabel("Time Step")
nexttile(1)
legend(["Input" "Forecasted"])
