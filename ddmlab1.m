%Step 1 generate your data 

%uild the discrete-time test system

[b, a] = cheby2(2, 3, [0.3 0.6]);
sys = tf(b, a, 1);
[h, t] = impulse(sys, 100);
N_h = 30; %Impulse respond drop to 1% of his peak
G_0 = h(1:N_h);

%Generate the excitation and output signals
N_e = 10 * N_h;               % Estimation length (≥10×nH)
N_v = N_e;                    % Validation length similar
u_e = randn(N_e, 1);         % Input for estimation
u_v = randn(N_v, 1);         % Input for validation

% 3.2.2 Compute exact output sequences
y_e = filter(G_0, 1, u_e);   % Exact output estimation
y_v = filter(G_0, 1, u_v);   % Exact output validation

SNR_lin = 10^(6/10); %SNR[dB]=6dB
% Signal power for each dataset
Ps_e = mean(y_e.^2);
Ps_v = mean(y_v.^2);
% Noise standard deviation
sigma_e = sqrt(Ps_e / SNR_lin);
sigma_v = sqrt(Ps_v / SNR_lin);
% Generate noise
ve = sigma_e * randn(N_e, 1);
vv = sigma_v * randn(N_v, 1);
% Measured outputs
y_me = y_e + ve;
y_mv = y_v + vv;


%step 2 Estimate model parameters 

n_max = 100;                     % maximum model order to test
V_LS = zeros(n_max, 1);         % store cost function values
theta_est = cell(n_max, 1);     % store parameter estimates

% Build regressor matrix for maximum order
H_max = toeplitz(u_e, [u_e(1); zeros(n_max-1, 1)]');

for n = 1:n_max
    H_n = H_max(:, 1:n);
    theta_n = H_n \ y_me;
    theta_est{n} = theta_n;
    e_n = y_me - H_n * theta_n;
    V_LS(n) = (1 / (N_e * sigma_e^2)) * (e_n' * e_n);
end

% 3.5 Plot V_LS curve
figure;
plot(1:n_max, V_LS, 'b-', 'LineWidth', 1.5);
xlabel('Model order n');
ylabel('V_{LS}(\theta_n, N_e)');
title('Least Squares Cost Function vs Model Order');
grid on;

[min_V_LS, n_opt_LS] = min(V_LS);
hold on;
plot(n_opt_LS, min_V_LS, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
legend('V_{LS}', sprintf('Minimum at n = %d', n_opt_LS), 'Location', 'best');

saveas(gcf, 'V_LS_curve_NormalSNR.png');

%step 3 select the optimal model order 
%AIC
V_AIC = zeros(n_max, 1);
for n = 1:n_max
    V_AIC(n) = V_LS(n) * (1 + (2*n)/N_e);
end
%Validation Cost
H_val_max = toeplitz(u_v, [u_v(1); zeros(n_max-1, 1)]');
V_val = zeros(n_max, 1);
for n = 1:n_max
    H_val_n = H_val_max(:, 1:n);
    residuals_val = y_mv - H_val_n * theta_est{n};
    V_val(n) = (1/(N_v * sigma_v^2)) * (residuals_val' * residuals_val);
end

% 3.4.3 Find optimal model orders
[~, n_opt_LS] = min(V_LS);
[~, n_opt_AIC] = min(V_AIC);
[~, n_opt_val] = min(V_val);

% Plot all three curves together
figure;
plot(1:n_max, V_LS, 'b-', 'LineWidth', 1.5); hold on;
plot(1:n_max, V_AIC, 'r--', 'LineWidth', 1.5);
plot(1:n_max, V_val, 'g-.', 'LineWidth', 1.5);
xlabel('Model order n');
ylabel('Cost function value');
title('Model Order Selection Criteria');
grid on;
legend('V_{LS} (training)', 'V_{AIC}', 'V_{val} (validation)', 'Location', 'best');

plot(n_opt_LS, V_LS(n_opt_LS), 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
plot(n_opt_AIC, V_AIC(n_opt_AIC), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
plot(n_opt_val, V_val(n_opt_val), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
xline(N_h, '--k', 'LineWidth', 1.5);

saveas(gcf, 'model_selection_criteria_NormalSNR.png');
% % Zoom plot
% figure;
% subplot(2,1,1);
% plot(1:n_max, V_LS, 'b-', 1:n_max, V_AIC, 'r--', 1:n_max, V_val, 'g-.', 'LineWidth', 1.5);
% xlabel('n'); ylabel('Cost');
% title('Full Range');
% legend('V_{LS}', 'V_{AIC}', 'V_{val}');
% grid on;
% 
% subplot(2,1,2);
% plot(1:50, V_LS(1:50), 'b-', 1:50, V_AIC(1:50), 'r--', 1:50, V_val(1:50), 'g-.', 'LineWidth', 1.5);
% xlabel('n'); ylabel('Cost');
% title('Zoom n=1 to 50');
% grid on;
% hold on;
% xline(30, '--k', 'LineWidth', 1.5);

%Step 4 Robustness experiment 
n_max_robust = 100;
n_runs = 100;

n_opt_LS_all = zeros(n_runs, 1);
n_opt_AIC_all = zeros(n_runs, 1);
n_opt_val_all = zeros(n_runs, 1);

for run = 1:n_runs
    u_e = randn(N_e, 1);
    u_v = randn(N_v, 1);
    
    y_e = filter(G_0, 1, u_e);
    y_v = filter(G_0, 1, u_v);
    
    ve = sigma_e * randn(N_e, 1);
    vv = sigma_v * randn(N_v, 1);
    
    y_me = y_e + ve;
    y_mv = y_v + vv;
    
    H_max = toeplitz(u_e, [u_e(1); zeros(n_max_robust-1, 1)]');
    H_val_max = toeplitz(u_v, [u_v(1); zeros(n_max_robust-1, 1)]');
    
    V_LS = zeros(n_max_robust, 1);
    theta_est = cell(n_max_robust, 1);
    
    for n = 1:n_max_robust
        H_n = H_max(:, 1:n);
        theta_n = H_n \ y_me;
        theta_est{n} = theta_n;
        residuals = y_me - H_n * theta_n;
        V_LS(n) = (1/(N_e * sigma_e^2)) * (residuals' * residuals);
    end
    
    V_AIC = V_LS .* (1 + 2*(1:n_max_robust)' / N_e);
    
    V_val = zeros(n_max_robust, 1);
    for n = 1:n_max_robust
        H_val_n = H_val_max(:, 1:n);
        residuals_val = y_mv - H_val_n * theta_est{n};
        V_val(n) = (1/(N_v * sigma_v^2)) * (residuals_val' * residuals_val);
    end
    
    [~, n_opt_LS_all(run)] = min(V_LS);
    [~, n_opt_AIC_all(run)] = min(V_AIC);
    [~, n_opt_val_all(run)] = min(V_val);
end

% Plot histograms
figure;
subplot(3,1,1);
histogram(n_opt_LS_all, 0.5:1:n_max_robust+0.5, 'FaceColor', 'b');
xlabel('Selected model order n'); ylabel('Frequency');
title('LS (Training) - Model Order Selection');
xlim([0, n_max_robust]);
grid on;

subplot(3,1,2);
histogram(n_opt_AIC_all, 0.5:1:n_max_robust+0.5, 'FaceColor', 'r');
xlabel('Selected model order n'); ylabel('Frequency');
title('AIC - Model Order Selection');
xlim([0, n_max_robust]);
grid on;

subplot(3,1,3);
histogram(n_opt_val_all, 0.5:1:n_max_robust+0.5, 'FaceColor', 'g');
xlabel('Selected model order n'); ylabel('Frequency');
title('Validation - Model Order Selection');
xlim([0, n_max_robust]);
grid on;

for i = 1:3
    subplot(3,1,i);
    hold on;
    xline(N_h, '--k', 'LineWidth', 1.5);
end

saveas(gcf, 'robustness_histograms_NormalSNR.png');