% Simple LSTM implementation in MATLAB without using built-in functions
% with longer sequence and visualization

% Define the sigmoid and tanh functions
sigmoid = @(x) 1 ./ (1 + exp(-x));
tanh = @(x) (exp(x) - exp(-x)) ./ (exp(x) + exp(-x));

% Initialize LSTM parameters
input_size = 1;   % input dimension
hidden_size = 10; % number of hidden units
output_size = 1;  % output dimension

% Initialize weights
Wf = randn(hidden_size, input_size + hidden_size); % forget gate weights
Wi = randn(hidden_size, input_size + hidden_size); % input gate weights
Wc = randn(hidden_size, input_size + hidden_size); % cell gate weights
Wo = randn(hidden_size, input_size + hidden_size); % output gate weights

% Initialize biases
bf = randn(hidden_size, 1); % forget gate bias
bi = randn(hidden_size, 1); % input gate bias
bc = randn(hidden_size, 1); % cell gate bias
bo = randn(hidden_size, 1); % output gate bias

% LSTM forward step
function [h_next, c_next] = lstm_cell(x, h_prev, c_prev, Wf, Wi, Wc, Wo, bf, bi, bc, bo, sigmoid, tanh)
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
end

% Generate some dummy data
time_steps = 200;
x_data = sin(0.1 * (1:time_steps)); % example sequence
h = zeros(hidden_size, 1); % initial hidden state
c = zeros(hidden_size, 1); % initial cell state

% Initialize output array
predicted = zeros(1, time_steps);

% LSTM forward pass through time
for t = 1:time_steps
    x = x_data(t);
    [h, c] = lstm_cell(x, h, c, Wf, Wi, Wc, Wo, bf, bi, bc, bo, sigmoid, tanh);
    predicted(t) = h(1); % Use the first hidden state as the prediction
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
