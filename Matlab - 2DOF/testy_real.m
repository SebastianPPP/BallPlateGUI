clear; clc; close all;
out_dir = 'testy_dane_realistyczne';
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
txt_filename = fullfile(out_dir, 'wskazniki_jakosci.txt');
fileID = fopen(txt_filename, 'w');
fprintf(fileID, '======================================================\n');
fprintf(fileID, 'WYNIKI SYMULACJI (MODEL REALISTYCZNY Z OPÓŹNIENIEM I SZUMEM)\n');
fprintf(fileID, '======================================================\n');
fprintf(fileID, 'Błąd liczony jako odległość od (0,0): r = sqrt(x^2 + y^2)\n');
fprintf(fileID, 'Energia całkowita układu: E = u_x^2 + u_y^2\n');
fprintf(fileID, '======================================================\n\n');

g = 9.81;
C_g = (5/7) * g; 
L_x = 0.30; 
L_y = 0.30; 
A = [0 1 0 0; 
     0 0 C_g 0; 
     0 0 0 1; 
     0 0 0 0];
B = [0; 0; 0; 1];
C_mat = [1 0 0 0];
D_mat = 0;

state_easy_x = [0.05; 0; 0; 0];  
state_easy_y = [0.05; 0; 0; 0];
state_hard_x = [0.12; 0.05; deg2rad(2); 0]; 
state_hard_y = [0.12; -0.05; deg2rad(-2); 0];

max_pos = 0.15; 
max_vel = 0.5; 
max_ang = deg2rad(15); 
max_ang_vel = deg2rad(45); 
max_u = deg2rad(15);

Q_bryson = diag([1/max_pos^2, 1/max_vel^2, 1/max_ang^2, 1/max_ang_vel^2]);
R_bryson = 1/max_u^2;

scenarios = {
    '1_Jednostkowe', diag([1, 1, 1, 1]), 1;
    '2_Zwiekszone_Q', diag([1000, 50, 10, 1]), 1;
    '3_Zwiekszone_R', diag([1, 1, 1, 1]), 100;
    '4a_Typowe_Agresywne', diag([2000, 100, 10, 1]), 0.1;
    '4b_Typowe_Zachowawcze', diag([50, 10, 100, 10]), 10;
    '4c_Typowe_Zrownowazone', diag([500, 20, 10, 0.1]), 1;
    '5_Regula_Brysona', Q_bryson, R_bryson
};

dt = 0.01;              % Czas próbkowania dyskretnego (np. mikrokontroler)
T_sim = 6.0;
time = 0:dt:T_sim;

% Parametry odwzorowujące rzeczywiste ograniczenia (hardware)
transport_delay_sec = 0.03; % 30 ms opóźnienia transportowego
delay_steps = round(transport_delay_sec / dt);
u_saturation = deg2rad(12); % Maksymalny kąt / prędkość sterowania
dead_zone_val = 0.002;      % Strefa martwa siłowników
noise_std = 0.001;          % Odchylenie standardowe szumu pomiarowego pozycji [m]

fig_coll_easy = figure('Name', 'Wykres porównawczy - prosty start', 'Position', [50 50 900 600]); hold on; grid on;
fig_coll_hard = figure('Name', 'Wykres porównawczy - trudny start', 'Position', [100 100 900 600]); hold on; grid on;
fig_bode_all = figure('Name', 'Zbiorcze charakterystyki Bode', 'Position', [100 100 900, 700]);
fig_nyq_all  = figure('Name', 'Zbiorcze wykresy Nyquista', 'Position', [150 150 900, 700]);

opt_bode = bodeoptions;
opt_bode.Grid = 'on';
opt_bode.PhaseVisible = 'on';
opt_bode.Title.String = 'Charakterystyka Bode układu zamkniętego';
opt_bode.XLabel.String = 'Częstotliwość [rad/s]';

opt_nyq = nyquistoptions;
opt_nyq.Grid = 'on';
opt_nyq.Title.String = 'Wykres Nyquista układu zamkniętego';
opt_nyq.XLabel.String = 'Część rzeczywista';
opt_nyq.YLabel.String = 'Część urojona';

fprintf('Start symulacji z uwzględnieniem opóźnień, nieliniowości i szumu\n');

for i = 1:size(scenarios, 1)
    name = scenarios{i, 1};
    Q = scenarios{i, 2};
    R = scenarios{i, 3};
    K = lqr(A, B, Q, R); 
    
    sys_cl = ss(A - B*K, B, C_mat, D_mat);
    [Gm, Pm, Wcg, Wcp] = margin(sys_cl);
    Gm_dB = 20*log10(Gm);
    if isinf(Gm_dB), Gm_dB = 999; end
    
    [mag, phase, w] = bode(sys_cl);
    mag = squeeze(mag);
    phase = squeeze(phase);
    bode_csv = fullfile(out_dir, sprintf('%s_bode.csv', name));
    writetable(table(w, mag, phase, 'VariableNames', {'Czesotl_rad_s', 'Amplituda', 'Faza_deg'}), bode_csv);
    
    [re, im, w_nyq] = nyquist(sys_cl);
    re = squeeze(re);
    im = squeeze(im);
    nyq_csv = fullfile(out_dir, sprintf('%s_nyquist.csv', name));
    writetable(table(w_nyq(:), re(:), im(:), 'VariableNames', {'Czesotl_rad_s', 'Re', 'Im'}), nyq_csv);
    
    fig_bode_single = figure('Visible', 'off');
    bodeplot(sys_cl, opt_bode);
    saveas(fig_bode_single, fullfile(out_dir, ['Bode_' name '.png']));
    close(fig_bode_single);
    
    fig_nyq_single = figure('Visible', 'off');
    nyquistplot(sys_cl, opt_nyq);
    saveas(fig_nyq_single, fullfile(out_dir, ['Nyquist_' name '.png']));
    close(fig_nyq_single);
    
    figure(fig_bode_all); bodeplot(sys_cl, opt_bode); hold on;
    figure(fig_nyq_all); nyquistplot(sys_cl, opt_nyq); hold on;
    
    diff_names = {'Prosta', 'Trudna'};
    
    for d_idx = 1:2
        diff_str = diff_names{d_idx};
        if d_idx == 1
            x0 = state_easy_x; y0 = state_easy_y;
        else
            x0 = state_hard_x; y0 = state_hard_y;
        end
        
        state_x = x0; state_y = y0;
        hist_x = zeros(1, length(time)); hist_y = zeros(1, length(time));
        hist_ux = zeros(1, length(time)); hist_uy = zeros(1, length(time));
        hist_alpha = zeros(1, length(time)); hist_beta = zeros(1, length(time));
        
        % Bufor na opóźnienie transportowe sterowania
        u_x_buffer = zeros(1, delay_steps + 1);
        u_y_buffer = zeros(1, delay_steps + 1);
        
        base_filename = sprintf('%s_%s', name, diff_str);
        png_filename = fullfile(out_dir, ['Wykres_' base_filename '.png']);
        csv_filename = fullfile(out_dir, [base_filename '_dane.csv']);
        is_fallen = false;
        
        for t_idx = 1:length(time)
            % Dodanie szumu pomiarowego do stanu (obserwacja pozycji z szumem)
            measured_x = state_x + [randn*noise_std; 0; 0; 0];
            measured_y = state_y + [randn*noise_std; 0; 0; 0];
            
            % Wyznaczenie surowego sterowania z regulatora
            raw_u_x = -K * measured_x; 
            raw_u_y = -K * measured_y;
            
            % Obsługa opóźnienia transportowego (bufor FIFO)
            u_x_buffer = [u_x_buffer(2:end), raw_u_x];
            u_y_buffer = [u_y_buffer(2:end), raw_u_y];
            delayed_u_x = u_x_buffer(1);
            delayed_u_y = u_y_buffer(1);
            
            % Strefa martwa (Dead Zone)
            if abs(delayed_u_x) < dead_zone_val
                u_x_eff = 0;
            else
                u_x_eff = sign(delayed_u_x) * (abs(delayed_u_x) - dead_zone_val);
            end
            
            if abs(delayed_u_y) < dead_zone_val
                u_y_eff = 0;
            else
                u_y_eff = sign(delayed_u_y) * (abs(delayed_u_y) - dead_zone_val);
            end
            
            % Ograniczenie wartości (Saturation)
            u_x = max(min(u_x_eff, u_saturation), -u_saturation);
            u_y = max(min(u_y_eff, u_saturation), -u_saturation);
            
            lim = deg2rad(15);
            alpha = max(min(state_x(3), lim), -lim);
            beta  = max(min(state_y(3), lim), -lim);
            
            hist_x(t_idx) = state_x(1); hist_y(t_idx) = state_y(1);
            hist_ux(t_idx) = u_x; hist_uy(t_idx) = u_y;
            hist_alpha(t_idx) = alpha; hist_beta(t_idx) = beta;
            
            if abs(state_x(1)) > L_x/2 || abs(state_y(1)) > L_y/2
                is_fallen = true;
                hist_x(t_idx:end) = NaN; hist_y(t_idx:end) = NaN;
                hist_ux(t_idx:end) = NaN; hist_uy(t_idx:end) = NaN;
                break;
            end
            
            % Krok całkowania numerycznego (dyskretny Euler)
            state_x = state_x + (A * state_x + B * u_x) * dt;
            state_y = state_y + (A * state_y + B * u_y) * dt;
        end
        
        valid_idx = ~isnan(hist_x);
        v_t = time(valid_idx); v_x = hist_x(valid_idx); v_y = hist_y(valid_idx);
        v_ux = hist_ux(valid_idx); v_uy = hist_uy(valid_idx);
        met = calc_metrics_2d(v_x, v_y, v_ux, v_uy, v_t);
        
        sim_data = table(v_t', v_x', v_y', hist_alpha(valid_idx)', hist_beta(valid_idx)', v_ux', v_uy', ...
            'VariableNames', {'Czas_s', 'Pozycja_X_m', 'Pozycja_Y_m', 'Kat_Alpha_rad', 'Kat_Beta_rad', 'Sterowanie_Ux', 'Sterowanie_Uy'});
        writetable(sim_data, csv_filename);
        
        fprintf(fileID, '### Scenariusz: %s | Sytuacja: %s ###\n', strrep(name, '_', ' '), diff_str);
        fprintf(fileID, '  Zapas fazy (Pm):        %7.2f deg\n', Pm);
        fprintf(fileID, '  Zapas amplitudy (Gm):   %7.2f dB\n', Gm_dB);
        
        if is_fallen
            fprintf(fileID, 'Wynik: Piłka spadła z płyty w czasie T = %.2f s (wpływ opóźnień/nieliniowości)\n\n', v_t(end));
        else
            fprintf(fileID, '  Czas regulacji 2D (Ts): %7.3f s\n', met.Ts);
            fprintf(fileID, '  Czas narastania 2D (Tr):%7.3f s\n', met.Tr);
            fprintf(fileID, '  Przeregulowanie (Mp):   %7.2f %%\n', met.Mp);
            fprintf(fileID, '  Wskaźnik IAE (2D):      %7.4f\n', met.IAE);
            fprintf(fileID, '  Wskaźnik ISE (2D):      %7.4f\n', met.ISE);
            fprintf(fileID, '  Wskaźnik ITAE (2D):      %7.4f\n', met.ITAE);
            fprintf(fileID, '  Całkowita energia (E):  %7.4f\n\n', met.Energy);
        end
        
        fig_res = figure('Name', ['Wyniki: ' base_filename], 'Color', 'w', 'Position', [100 100 1000 600], 'Visible', 'off');
        title_str = sprintf('%s (Realistyczny)', strrep(name, '_', ' '));
        if is_fallen, title_str = [title_str ' [Piłka spadła]']; end
        sgtitle(title_str, 'FontWeight', 'bold');
        
        subplot(2,2,1); plot(v_t, v_x, 'b', v_t, v_y, 'g', 'LineWidth', 1.5);
        yline(0, 'k--'); title('Pozycja piłki (z szumem)'); ylabel('Pozycja [m]'); xlabel('Czas [s]'); legend('X', 'Y'); grid on;
        
        subplot(2,2,2); plot(v_t, rad2deg(hist_alpha(valid_idx)), 'b', v_t, rad2deg(hist_beta(valid_idx)), 'g', 'LineWidth', 1.5);
        title('Kąty nachylenia płyty'); ylabel('Kąt [deg]'); xlabel('Czas [s]'); legend('\alpha', '\beta'); grid on;
        
        subplot(2,2,3); plot(v_t, v_ux, 'b', v_t, v_uy, 'g', 'LineWidth', 1.5);
        title('Sygnał sterujący (z nasyceniem i opóźnieniem)'); ylabel('u [rad/s^2]'); xlabel('Czas [s]'); legend('U_x', 'U_y'); grid on;
        
        subplot(2,2,4); plot(v_x, v_y, 'b-', 'LineWidth', 1.5); hold on;
        plot(v_x(1), v_y(1), 'ro', 'MarkerFaceColor', 'r'); plot(0, 0, 'rx', 'LineWidth', 2, 'MarkerSize', 8);
        rectangle('Position', [-L_x/2, -L_y/2, L_x, L_y], 'EdgeColor', 'k', 'LineStyle', '--');
        title('Trajektoria na płycie XY'); xlabel('X [m]'); ylabel('Y [m]'); axis equal; grid on; xlim([-0.2 0.2]); ylim([-0.2 0.2]);
        
        saveas(fig_res, png_filename);
        close(fig_res);
        
        r_dist = sqrt(v_x.^2 + v_y.^2);
        if d_idx == 1
            figure(fig_coll_easy); plot(v_t, r_dist, 'LineWidth', 1.5, 'DisplayName', strrep(name, '_', ' '));
        else
            figure(fig_coll_hard); plot(v_t, r_dist, 'LineWidth', 1.5, 'DisplayName', strrep(name, '_', ' '));
        end
        
        fprintf('Zapisano: %s\n', base_filename);
    end
end
fclose(fileID);

figure(fig_coll_easy); title('Odległość piłki od celu (0,0) - Sytuacja prosta (Realistyczna)'); xlabel('Czas [s]'); ylabel('Odległość [m]'); legend;
saveas(fig_coll_easy, fullfile(out_dir, 'ZBIORCZY_Prosta.png'));
figure(fig_coll_hard); title('Odległość piłki od celu (0,0) - Sytuacja trudna (Realistyczna)'); xlabel('Czas [s]'); ylabel('Odległość [m]'); legend;
saveas(fig_coll_hard, fullfile(out_dir, 'ZBIORCZY_Trudna.png'));

figure(fig_bode_all);
legend(scenarios(:, 1), 'Location', 'southwest', 'Interpreter', 'none');
saveas(fig_bode_all, fullfile(out_dir, 'ZBIORCZY_Bode.png'));

figure(fig_nyq_all);
legend(scenarios(:, 1), 'Location', 'northeast', 'Interpreter', 'none');
saveas(fig_nyq_all, fullfile(out_dir, 'ZBIORCZY_Nyquist.png'));

function metrics = calc_metrics_2d(x, y, ux, uy, t)
    dt = t(2) - t(1);
    r = sqrt(x.^2 + y.^2);
    r0 = r(1);
    if abs(r0) < 1e-4, r0 = 1e-4; end
    
    metrics.IAE = sum(r) * dt;
    metrics.ISE = sum(r.^2) * dt;
    metrics.ITAE = sum(t .* r) * dt;
    metrics.Energy = sum(ux.^2 + uy.^2) * dt;
    
    threshold = max(0.001, 0.05 * r0); 
    idx_ts = find(r > threshold, 1, 'last');
    if isempty(idx_ts), metrics.Ts = 0; else, metrics.Ts = t(idx_ts); end
    
    x0 = x(1);
    if x0 > 0
        min_x = min(x); 
        metrics.Mp = max(0, (abs(min_x) / x0) * 100);
    else
        max_x = max(x); 
        metrics.Mp = max(0, (abs(max_x) / abs(x0)) * 100);
    end
    
    idx90 = find(r <= 0.9 * r0, 1, 'first');
    idx10 = find(r <= 0.1 * r0, 1, 'first');
    if isempty(idx90) || isempty(idx10) || (idx10 < idx90)
        metrics.Tr = NaN; 
    else
        metrics.Tr = t(idx10) - t(idx90);
    end
end