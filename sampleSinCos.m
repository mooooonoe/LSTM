% Define the sigmoid and tanh functions
sigmoid = @(x) 1 ./ (1 + exp(-x));
tanh = @(x) (exp(x) - exp(-x)) ./ (exp(x) + exp(-x));

% Initialize LSTM parameters
input_size = 1;   % input dimension
hidden_size = 10; % number of hidden units
output_size = 1;  % output dimension
learning_rate = 0.01; % learning rate

% Initialize weights
Wf = randn(hidden_size, input_size + hidden_size) * 0.01; % forget gate weights
Wi = randn(hidden_size, input_size + hidden_size) * 0.01; % input gate weights
Wc = randn(hidden_size, input_size + hidden_size) * 0.01; % cell gate weights
Wo = randn(hidden_size, input_size + hidden_size) * 0.01; % output gate weights

% Initialize biases
bf = zeros(hidden_size, 1); % forget gate bias
bi = zeros(hidden_size, 1); % input gate bias
bc = zeros(hidden_size, 1); % cell gate bias
bo = zeros(hidden_size, 1); % output gate bias

% LSTM forward step
function [h_next, c_next, y, cache] = lstm_cell(x, h_prev, c_prev, Wf, Wi, Wc, Wo, bf, bi, bc, bo, sigmoid, tanh)
    % Concatenate input and previous hidden state
    z = [x; h_prev];

    % Forget gate
    ft = sigmoid(Wf * z + bf);

    % Input gate
    it = sigmoid(Wi * z + bi);

    % Cell gate
    ct_hat = tanh(Wc * z + bc);

    % Cell state
    c_next = ft .* c_prev + it .* ct_hat;

    % Output gate
    ot = sigmoid(Wo * z + bo);

    % Hidden state
    h_next = ot .* tanh(c_next);

    % Output
    y = h_next(1); % Use the first hidden state as the output

    % Cache values for backpropagation
    cache.z = z;
    cache.ft = ft;
    cache.it = it;
    cache.ct_hat = ct_hat;
    cache.c_next = c_next;
    cache.ot = ot;
    cache.h_next = h_next;
    cache.c_prev = c_prev;
end

% Training function
function [Wf, Wi, Wc, Wo, bf, bi, bc, bo] = train_lstm(x_data, y_data, Wf, Wi, Wc, Wo, bf, bi, bc, bo, sigmoid, tanh, learning_rate, epochs)
    hidden_size = size(Wf, 1);
    input_size = size(Wf, 2) - hidden_size;
    for epoch = 1:epochs
        h = zeros(hidden_size, 1); % initial hidden state
        c = zeros(hidden_size, 1); % initial cell state
        total_loss = 0; % initialize total loss
        caches = cell(length(x_data), 1); % to store cache for each time step

        % Forward pass
        for t = 1:length(x_data)
            x = x_data(t);
            y_true = y_data(t);
            [h, c, y_pred, cache] = lstm_cell(x, h, c, Wf, Wi, Wc, Wo, bf, bi, bc, bo, sigmoid, tanh);
            caches{t} = cache;

            % Calculate loss (mean squared error)
            loss = (y_pred - y_true)^2;
            total_loss = total_loss + loss;
        end

        % Backward pass (simplified)
        dWf = zeros(size(Wf));
        dWi = zeros(size(Wi));
        dWc = zeros(size(Wc));
        dWo = zeros(size(Wo));
        dbf = zeros(size(bf));
        dbi = zeros(size(bi));
        dbc = zeros(size(bc));
        dbo = zeros(size(bo));

        dh_next = zeros(hidden_size, 1);
        dc_next = zeros(hidden_size, 1);

        for t = length(x_data):-1:1
            cache = caches{t};
            z = cache.z;
            ft = cache.ft;
            it = cache.it;
            ct_hat = cache.ct_hat;
            c_next = cache.c_next;
            ot = cache.ot;
            h_next = cache.h_next;
            c_prev = cache.c_prev;

            % Gradients of output
            dy = 2 * (h_next(1) - y_data(t)); % derivative of loss w.r.t. y_pred

            % Gradients of LSTM gates
            dot = dy * tanh(c_next) .* ot .* (1 - ot);
            dc_next = dy * ot .* (1 - tanh(c_next).^2) + dc_next;
            dct_hat = dc_next .* it .* (1 - ct_hat.^2);
            dit = dc_next .* ct_hat .* it .* (1 - it);
            dft = dc_next .* c_prev .* ft .* (1 - ft);

            % Gradients of weights and biases
            dWf = dWf + dft * z';
            dWi = dWi + dit * z';
            dWc = dWc + dct_hat * z';
            dWo = dWo + dot * z';

            dbf = dbf + dft;
            dbi = dbi + dit;
            dbc = dbc + dct_hat;
            dbo = dbo + dot;

            % Gradients of previous hidden and cell states
            dz = Wf' * dft + Wi' * dit + Wc' * dct_hat + Wo' * dot;
            dh_next = dz(input_size + 1:end);
            dc_next = dc_next .* ft;
        end

        % Update weights and biases
        Wf = Wf - learning_rate * dWf;
        Wi = Wi - learning_rate * dWi;
        Wc = Wc - learning_rate * dWc;
        Wo = Wo - learning_rate * dWo;

        bf = bf - learning_rate * dbf;
        bi = bi - learning_rate * dbi;
        bc = bc - learning_rate * dbc;
        bo = bo - learning_rate * dbo;

        % Print average loss for every epoch
        fprintf('Epoch %d, Loss: %f\n', epoch, total_loss / length(x_data));
    end
end

% Generate some dummy data
time_steps = 200;
x_data = sin(0.1 * (1:time_steps)); % example sequence
y_data = cos(0.1 * (1:time_steps)); % target sequence (for training purpose)

% Train the LSTM
epochs = 1000; % increase the number of epochs
[Wf, Wi, Wc, Wo, bf, bi, bc, bo] = train_lstm(x_data, y_data, Wf, Wi, Wc, Wo, bf, bi, bc, bo, sigmoid, tanh, learning_rate, epochs);

% Initialize output array
predicted = zeros(1, time_steps);
h = zeros(hidden_size, 1); % initial hidden state
c = zeros(hidden_size, 1); % initial cell state

% LSTM forward pass through time
for t = 1:time_steps
    x = x_data(t);
    [h, c, predicted(t)] = lstm_cell(x, h, c, Wf, Wi, Wc, Wo, bf, bi, bc, bo, sigmoid, tanh);
end

% Plot the results
figure;
plot(1:time_steps, x_data, 'b', 'LineWidth', 2);
hold on;
plot(1:time_steps, predicted, 'r--', 'LineWidth', 2);
legend('Actual', 'Predicted');
xlabel('Time step');
ylabel('Value');
title('LSTM Sequence Prediction');
hold off;
