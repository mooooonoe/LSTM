% Define the range of x values
x = -5:0.1:5;

% Define the activation functions
step = double(x >= 0);
sigmoid = 1 ./ (1 + exp(-x));
tanh_func = tanh(x);
relu = max(0, x);

% CORDIC algorithm function
function [cordic_result] = cordic_function(input_values)
    num_iterations = 15; % Number of iterations for CORDIC
    cordic_gain = 0.45;
    for i = 0:num_iterations-1
        cordic_gain = cordic_gain * sqrt(1 + 2^(-2 * i));
    end
    
    % Initialize the result vector
    cordic_result = zeros(size(input_values));
    
    % CORDIC algorithm
    for idx = 1:length(input_values)
        angle = input_values(idx);
        K = 1;
        current_angle = 0;
        X = cordic_gain;
        Y = 0;
        Z = angle;
        
        for i = 0:num_iterations-1
            if Z < 0
                d = -1;
            else
                d = 1;
            end
            
            X_new = X - Y * d * 2^(-i);
            Y = Y + X * d * 2^(-i);
            X = X_new;
            
            Z = Z - d * atan(2^(-i));
        end
        
        cordic_result(idx) = Y;
    end
end

% Apply CORDIC to each activation function
cordic_step = cordic_function(step);
cordic_sigmoid = cordic_function(sigmoid);
cordic_tanh = cordic_function(tanh_func);
cordic_relu = cordic_function(relu);

% Save the results to a text file
fileID = fopen('cordic_activation_functions.txt', 'w');
fprintf(fileID, 'x,cordic_step,cordic_sigmoid,cordic_tanh,cordic_relu\n');
for i = 1:length(x)
    fprintf(fileID, '%.1f,%.4f,%.4f,%.4f,%.4f\n', x(i), cordic_step(i), cordic_sigmoid(i), cordic_tanh(i), cordic_relu(i));
end
fclose(fileID);

% Plot the activation functions
figure;

subplot(2, 1, 1); % Create the first subplot for the activation functions
hold on;
plot(x, step, 'r', 'LineWidth', 1.5); % Step function in red
plot(x, sigmoid, 'g--', 'LineWidth', 1.5); % Sigmoid function in green dashed
plot(x, tanh_func, 'b', 'LineWidth', 1.5); % Tanh function in blue
plot(x, relu, 'm-.', 'LineWidth', 1.5); % ReLU function in magenta dash-dot
xlabel('x');
ylabel('Activation value');
title('Activation Functions');
legend('Step', 'Sigmoid', 'Tanh', 'ReLU');
axis([-5 5 -1.5 1.5]);
hold off;

% Plot the CORDIC-transformed activation functions
subplot(2, 1, 2); % Create the second subplot for the CORDIC-transformed functions
hold on;
plot(x, cordic_step, 'r.', 'LineWidth', 1.5); % CORDIC Step function in red
plot(x, cordic_sigmoid, 'g.', 'LineWidth', 1.5); % CORDIC Sigmoid function in green
plot(x, cordic_tanh, 'b.', 'LineWidth', 1.5); % CORDIC Tanh function in blue
plot(x, cordic_relu, 'm.', 'LineWidth', 1.5); % CORDIC ReLU function in magenta
xlabel('x');
ylabel('CORDIC value');
title('CORDIC Transformed Activation Functions');
legend('CORDIC Step', 'CORDIC Sigmoid', 'CORDIC Tanh', 'CORDIC ReLU');
axis([-5 5 -1.5 1.5]);
hold off;
