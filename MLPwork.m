clc;
clear;
close all;

%% ===========================
% User Inputs
%% ===========================
N = input('Please type number of input samples (>= 100 recommended): ');
eta = input('Please type learning rate (e.g. 0.01): ');
epochs = input('Please type max epochs (e.g. 5000): ');

%% ===========================
% Dataset
%% ===========================
x = linspace(0, 2*pi, N);
t = sin(x) .* cos(x);              % True function

% Normalize input to [0,1]
x = x / (2*pi);

% Normalize target to [-1,1]
t = t / max(abs(t));

%% ===========================
% Train / Test Split
%% ===========================
rng(42);
idx = randperm(N);
nTrain = round(0.7 * N);

trainIdx = idx(1:nTrain);
testIdx  = idx(nTrain+1:end);

x_train = x(trainIdx);
t_train = t(trainIdx);

x_test = x(testIdx);
t_test = t(testIdx);

%% ===========================
% MLP Architecture
%% ===========================
nHidden = 20;   % Increased capacity

W1 = randn(nHidden,1) * 0.5;
b1 = zeros(nHidden,1);

W2 = randn(1,nHidden) * 0.5;
b2 = 0;

%% ===========================
% Training (SGD + Backprop)
%% ===========================
for epoch = 1:epochs
    
    mse = 0;
    
    for i = 1:length(x_train)
        
        % ----- Forward -----
        z1 = W1 * x_train(i) + b1;
        a1 = tanh(z1);
        
        y = W2 * a1 + b2;
        
        % ----- Error -----
        e = y - t_train(i);
        mse = mse + e^2;
        
        % ----- Backprop -----
        delta2 = e;
        delta1 = (W2' * delta2) .* (1 - a1.^2);
        
        % ----- Update -----
        W2 = W2 - eta * delta2 * a1';
        b2 = b2 - eta * delta2;
        
        W1 = W1 - eta * delta1 * x_train(i);
        b1 = b1 - eta * delta1;
    end
    
    mse = mse / length(x_train);
    
    if mod(epoch,500)==0
        disp(['Epoch ',num2str(epoch),' | Train MSE = ',num2str(mse)]);
    end
end

%% ===========================
% Testing
%% ===========================
y_test = zeros(size(x_test));
for i = 1:length(x_test)
    a1 = tanh(W1 * x_test(i) + b1);
    y_test(i) = W2 * a1 + b2;
end

testMSE = mean((y_test - t_test).^2);
disp(['Final Test MSE = ', num2str(testMSE)]);

%% ===========================
% Visualization
%% ===========================
x_plot = linspace(0,1,400);
y_plot = zeros(size(x_plot));

for i = 1:length(x_plot)
    a1 = tanh(W1 * x_plot(i) + b1);
    y_plot(i) = W2 * a1 + b2;
end

figure;
plot(x_plot*2*pi, sin(x_plot*2*pi).*cos(x_plot*2*pi), ...
     'b','LineWidth',2);
hold on;
plot(x_plot*2*pi, y_plot * max(abs(sin(x).*cos(x))), ...
     'r--','LineWidth',2);
legend('True Function','Manual MLP');
grid on;
xlabel('x (radians)');
ylabel('sin(x)cos(x)');
title('Manual MLP Correctly Learning sin(x)cos(x)');