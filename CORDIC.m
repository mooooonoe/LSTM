% 입력값의 범위 설정
u_ref = -100:0.5:100;       % Input array (larger range of values)
u_in_arb = fi(u_ref,0,16); % 16-bit unsigned fixed-point input data values
u_len = numel(u_ref);

% 활성화 함수 적용
% 시그모이드 활성화 함수
sigmoid = @(x) 1 ./ (1 + exp(-x));
sigmoid_ref = sigmoid(double(u_in_arb));

% 탄젠트 활성화 함수
tanh_func = @(x) tanh(x);
tanh_ref = tanh_func(double(u_in_arb));

% MATLAB sqrt 참조 결과
sqrt_ref = sqrt(double(u_in_arb));

% CORDIC 계산 반복 횟수
niter = 100;

% 결과 저장을 위한 배열 초기화
results_sigmoid = zeros(u_len, 2);
results_tanh = zeros(u_len, 2);
results_sqrt = zeros(u_len, 2);

% 참조 결과 저장
results_sigmoid(:,2) = sigmoid_ref(:);
results_tanh(:,2) = tanh_ref(:);
results_sqrt(:,2) = sqrt_ref(:);

% CORDIC sqrt 계산
x_out_sqrt = cordicsqrt(u_in_arb, niter);
results_sqrt(:,1) = double(x_out_sqrt);

% CORDIC 계산 - sigmoid
x_out_sigmoid = sigmoid(double(cordicsqrt(u_in_arb, niter)));
results_sigmoid(:,1) = double(x_out_sigmoid);

% CORDIC 계산 - tanh
x_out_tanh = tanh_func(double(cordicsqrt(u_in_arb, niter)));
results_tanh(:,1) = double(x_out_tanh);

% 그래프 그리기
figure;

% 활성화 함수 결과 그래프 (sigmoid와 tanh 동일 플롯에 표시)
subplot(211);
plot(u_ref, results_sigmoid(:,1), 'r.', u_ref, results_sigmoid(:,2), 'b-', ...
     u_ref, results_tanh(:,1), 'g.', u_ref, results_tanh(:,2), 'm-');
legend('CORDIC Sigmoid', 'Reference Sigmoid', 'CORDIC TanH', 'Reference TanH', 'Location', 'SouthEast');
title('CORDIC Sigmoid and TanH (Large Input Range) and MATLAB Reference Results');
axis([-100 100 -1.5 1.5]);
hold off;

% sqrt 절대 오차 그래프
subplot(212);
absErr_sqrt = abs(results_sqrt(:,2) - results_sqrt(:,1));
plot(u_ref, absErr_sqrt);
title('Absolute Error (vs. MATLAB SQRT Reference Results)');
axis([-100 100 0 max(absErr_sqrt)]);
