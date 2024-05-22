% MATLAB 코드
% 고정 소수점 설정
wordLength = 16;  % 전체 비트 수
fractionLength = 8;  % 소수점 이하 비트 수
fip = fimath('RoundingMethod', 'Floor', 'OverflowAction', 'Wrap');

% sigmoid 함수 정의
sigmoid = @(x) 1 ./ (1 + exp(-x));

% 입력 범위 설정
x_min = -8; % 최소 입력 값
x_max = 8;  % 최대 입력 값
numPoints = 2^wordLength; % LUT의 크기

% 입력 값 생성
x = linspace(x_min, x_max, numPoints);

% sigmoid 계산 및 고정 소수점 변환
y = sigmoid(x);
y_fixed = fi(y, 1, wordLength, fractionLength, 'fimath', fip);

% LUT로 저장
lut = y_fixed;

% 결과를 파일로 저장
fid = fopen('sigmoid_lut.hex', 'w');
for i = 1:length(lut)
    fprintf(fid, '%04x\n', bin2dec(lut.bin(i, :)));
end
fclose(fid);

