% Define the range of x values
x = -5:0.1:5;

% Define the activation functions
step = double(x >= 0);
sigmoid = 1 ./ (1 + exp(-x));
tanh_func = tanh(x);
relu = max(0, x);

% Plot the activation functions
figure;
hold on;
plot(x, step, 'r', 'LineWidth', 1.5); % Step function in red
plot(x, sigmoid, 'g--', 'LineWidth', 1.5); % Sigmoid function in green dashed
plot(x, tanh_func, 'b', 'LineWidth', 1.5); % Tanh function in blue
plot(x, relu, 'm-.', 'LineWidth', 1.5); % ReLU function in magenta dash-dot

% Add labels and title
xlabel('x');
ylabel('Activation value');
title('Activation functions');

% Add legend
legend('Step', 'Sigmoid', 'Tanh', 'ReLU');

% Set the axis limits
axis([-5 5 -1.5 1.5]);

hold off;