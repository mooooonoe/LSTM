% Define the range of x values
x = -5:0.1:5;

% Define the input functions
sine_func = sin(x);
rect_func = double(abs(x) <= 1);

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

% Apply CORDIC to each input function
cordic_sine = cordic_function(sine_func);
cordic_rect = cordic_function(rect_func);

% Plot the input functions
figure;

subplot(2, 1, 1); % Create the first subplot for the input functions
hold on;
plot(x, sine_func, 'r', 'LineWidth', 1.5); % Sine function in red
plot(x, rect_func, 'g--', 'LineWidth', 1.5); % Rect function in green dashed
xlabel('x');
ylabel('Function value');
title('Input Functions');
legend('Sine', 'Rect');
axis([-5 5 -1.5 1.5]);
hold off;

% Plot the CORDIC-transformed functions
subplot(2, 1, 2); % Create the second subplot for the CORDIC-transformed functions
hold on;
plot(x, cordic_sine, 'r.', 'LineWidth', 1.5); % CORDIC Sine function in red
plot(x, cordic_rect, 'g.', 'LineWidth', 1.5); % CORDIC Rect function in green dashed
xlabel('x');
ylabel('CORDIC value');
title('CORDIC Transformed Functions');
legend('CORDIC Sine', 'CORDIC Rect');
axis([-5 5 -1.5 1.5]);
hold off;
