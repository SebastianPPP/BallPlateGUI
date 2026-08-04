clear; clc; close all;

out_dir = 'wyniki_3dof_bryson_dane';
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

txt_filename = fullfile(out_dir, 'wskazniki_jakosci_bryson_3dof.txt');
fileID = fopen(txt_filename, 'w', 'native', 'UTF-8');
fprintf(fileID, '===================================================================\n');
fprintf(fileID, 'WYNIKI SYMULACJI - TESTY REGUŁY BRYSONA (3DOF) I ZAPASY STABILNOŚCI\n');
fprintf(fileID, '===================================================================\n');
fprintf(fileID, 'Błąd liczony jako odległość od (0,0): r = sqrt(x^2 + y^2)\n');
fprintf(fileID, 'Energia całkowita układu: E = u_x^2 + u_y^2\n');
fprintf(fileID, '===================================================================\n\n');

state_easy_x = [0.05; 0; 0; 0];  
state_easy_y = [0.05; 0; 0; 0];

state_hard_x = [0.12; 0.05; deg2rad(2); 0]; 
state_hard_y = [0.12; -0.05; deg2rad(-2); 0];

g     = 9.81;
Cg    = (5/7) * g;
R_plate = 0.30;
H_base  = 0.15;
dt    = 0.01;
Tsim  = 6.0;
time  = 0:dt:Tsim;
N     = length(time);

bryson_params = [
    0.15,   0.50,   0.26,   1.0,   5.0;   
    0.10,   1.00,   0.26,   1.5,  10.0;   
    0.20,   0.30,   0.20,   0.5,   3.0;   
    0.05,   2.00,   0.30,   3.0,  20.0;   
    0.15,   0.50,   0.15,   1.0,   5.0;   
];
iter_labels = {'B1_bazowa', 'B2_szybka', 'B3_ostrozna', 'B4_agresywna', 'B5_maly_kat'};
n_iter = size(bryson_params, 1);

A = [0  1   0  0;
     0  0  Cg  0;
     0  0   0  1;
     0  0   0  0];
B = [0; 0; 0; 1];
C_mat = [1 0 0 0];
D_mat = 0;

controllers = struct();
for k = 1:n_iter
    mp  = bryson_params(k, :);
    Q = diag([1/mp(1)^2, 1/mp(2)^2, 1/mp(3)^2, 1/mp(4)^2]);
    R = 1/mp(5)^2;
    
    [~, ~, K_gain] = care(A, B, Q, R);
    
    controllers(k).label  = iter_labels{k};
    controllers(k).K      = K_gain; 
    controllers(k).Q_diag = diag(Q)';
    controllers(k).R_val  = R;
end

fig_bode_all = figure('Name', 'Zbiorcze charakterystyki Bode - Bryson 3DOF', 'Position', [100 100 900, 700]);
fig_nyq_all  = figure('Name', 'Zbiorcze wykresy Nyquista - Bryson 3DOF', 'Position', [150 150 900, 700]);

opt_bode = bodeoptions;
opt_bode.Grid = 'on';
opt_bode.PhaseVisible = 'on';
opt_bode.Title.String = 'Charakterystyka Bode - Reguła Brysona (3DOF)';
opt_bode.XLabel.String = 'Częstotliwość [rad/s]';

opt_nyq = nyquistoptions;
opt_nyq.Grid = 'on';
opt_nyq.Title.String = 'Wykres Nyquista - Reguła Brysona (3DOF)';
opt_nyq.XLabel.String = 'Część rzeczywista';
opt_nyq.YLabel.String = 'Część urojona';

for k = 1:n_iter
    K_gain = controllers(k).K;
    name = controllers(k).label;
    
    sys_cl = ss(A - B*K_gain, B, C_mat, D_mat);
    [Gm, Pm, ~, ~] = margin(sys_cl);
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
    
    figure(fig_bode_all);
    bodeplot(sys_cl, opt_bode);
    hold on;
    
    figure(fig_nyq_all);
    nyquistplot(sys_cl, opt_nyq);
    hold on;
end

figure(fig_bode_all);
legend(iter_labels, 'Location', 'southwest', 'Interpreter', 'none');
saveas(fig_bode_all, fullfile(out_dir, 'ZBIORCZY_Bryson_3DOF_Bode.png'));
close(fig_bode_all);

figure(fig_nyq_all);
legend(iter_labels, 'Location', 'northeast', 'Interpreter', 'none');
saveas(fig_nyq_all, fullfile(out_dir, 'ZBIORCZY_Bryson_3DOF_Nyquist.png'));
close(fig_nyq_all);

test_scenarios = struct();
test_scenarios(1).name = 'Prosta';
test_scenarios(1).x0_x = state_easy_x;
test_scenarios(1).x0_y = state_easy_y;
test_scenarios(2).name = 'Trudna';
test_scenarios(2).x0_x = state_hard_x;
test_scenarios(2).x0_y = state_hard_y;

theta_circ = linspace(0, 2*pi, 60);

for s = 1:length(test_scenarios)
    scen = test_scenarios(s);
    x0_x = scen.x0_x;
    x0_y = scen.x0_y;
    
    f = figure('Name', ['Weryfikacja Brysona 3DOF - Sytuacja: ' scen.name], 'Color', 'white', 'Position', [50 150 1500 450], 'Visible', 'off');
    
    subplot(1,3,1); hold on; grid on; 
    title(['Odległość od środka — ' scen.name]); xlabel('Czas [s]'); ylabel('r [m]');
    yline(0, 'k--', 'LineWidth', 0.8, 'HandleVisibility', 'off');
    
    subplot(1,3,2); hold on; grid on; 
    title(['Sygnał sterujący u_x(t) — ' scen.name]); xlabel('Czas [s]'); ylabel('u [rad/s^2]');
    yline(0, 'k--', 'LineWidth', 0.8, 'HandleVisibility', 'off');
    
    subplot(1,3,3); hold on; grid on; 
    title(['Trajektoria XY — ' scen.name]);
    plot(R_plate*cos(theta_circ), R_plate*sin(theta_circ), 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    plot(0, 0, 'rx', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'Cel');
    plot(x0_x(1), x0_y(1), 'ko', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'Start');
    xlabel('X [m]'); ylabel('Y [m]'); axis equal; xlim([-0.35 0.35]); ylim([-0.35 0.35]);
    
    for k = 1:n_iter
        K_gain = controllers(k).K;
        name = controllers(k).label;
        
        sx = x0_x;  sy = x0_y;
        hx = zeros(1, N);  hy = zeros(1, N);
        hux = zeros(1, N); huy = zeros(1, N);
        h_alpha = zeros(1, N); h_beta = zeros(1, N);
        fallen = false;  fallen_t = NaN;
        
        for i = 1:N
            ux = -K_gain * sx;
            uy = -K_gain * sy;
            
            lim_val = deg2rad(15);
            alpha = max(min(sx(3), lim_val), -lim_val);
            beta  = max(min(sy(3), lim_val), -lim_val);
            
            hx(i)  = sx(1);
            hy(i)  = sy(1);
            hux(i) = ux;
            huy(i) = uy;
            h_alpha(i) = alpha;
            h_beta(i) = beta;
            
            if sqrt(sx(1)^2 + sy(1)^2) > R_plate
                fallen = true;
                fallen_t = time(i);
                hx(i:end)  = NaN;  hy(i:end)  = NaN;
                hux(i:end) = NaN; huy(i:end) = NaN;
                h_alpha(i:end) = NaN; h_beta(i:end) = NaN;
                break;
            end
            
            sx = sx + (A*sx + B*ux) * dt;
            sy = sy + (A*sy + B*uy) * dt;
        end
        
        valid_idx = ~isnan(hx);
        t_sim = time(valid_idx);
        v_hx = hx(valid_idx);
        v_hy = hy(valid_idx);
        v_hux = hux(valid_idx);
        v_hu_y = huy(valid_idx);
        
        dist = sqrt(v_hx.^2 + v_hy.^2);
        dist0 = sqrt(x0_x(1)^2 + x0_y(1)^2);
        
        thr = max(0.001, 0.02 * dist0);
        idx_ts = find(dist > thr, 1, 'last');
        Ts = 0; if ~isempty(idx_ts), Ts = t_sim(idx_ts); end
        
        Mp = max(0, (max(dist) - dist0) / dist0 * 100);
        IAE = sum(dist) * dt;
        Energia = sum(v_hux.^2 + v_hu_y.^2) * dt;
        
        sys_cl = ss(A - B*K_gain, B, C_mat, D_mat);
        [Gm, Pm, ~, ~] = margin(sys_cl);
        Gm_dB = 20*log10(Gm);
        if isinf(Gm_dB), Gm_dB = 999; end
        
        fprintf(fileID, '### Sytuacja: %s | Regulator: %s ###\n', scen.name, name);
        fprintf(fileID, '  Zapas fazy (Pm):        %7.2f deg\n', Pm);
        fprintf(fileID, '  Zapas amplitudy (Gm):   %7.2f dB\n', Gm_dB);
        if fallen
            fprintf(fileID, 'Wynik: Piłka spadła z płyty w czasie T = %.2f s\n\n', fallen_t);
        else
            fprintf(fileID, '  Czas regulacji (Ts):    %7.3f s\n', Ts);
            fprintf(fileID, '  Przeregulowanie (Mp):   %7.2f %%\n', Mp);
            fprintf(fileID, '  Wskaźnik IAE:           %7.4f\n', IAE);
            fprintf(fileID, '  Całkowita energia (E):  %7.4f\n\n', Energia);
        end
        
        sim_data = table(t_sim', v_hx', v_hy', h_alpha(valid_idx)', h_beta(valid_idx)', v_hux', v_hu_y', ...
            'VariableNames', {'Czas_s', 'Pozycja_X_m', 'Pozycja_Y_m', 'Kat_Alpha_rad', 'Kat_Beta_rad', 'Sterowanie_Ux', 'Sterowanie_Uy'});
        writetable(sim_data, fullfile(out_dir, sprintf('bryson_3dof_%s_%s_dane.csv', scen.name, name)));
        
        subplot(1,3,1); plot(t_sim, dist, 'LineWidth', 1.8, 'DisplayName', name);
        subplot(1,3,2); plot(t_sim, v_hux, 'LineWidth', 1.8, 'DisplayName', name);
        subplot(1,3,3); plot(v_hx, v_hy, 'LineWidth', 1.8, 'DisplayName', name);
    end
    
    subplot(1,3,1); legend('Location', 'best', 'FontSize', 8, 'Interpreter', 'none');
    subplot(1,3,2); legend('Location', 'best', 'FontSize', 8, 'Interpreter', 'none');
    subplot(1,3,3); legend('Location', 'best', 'FontSize', 8, 'Interpreter', 'none');
    
    saveas(f, fullfile(out_dir, sprintf('wykres_bryson_3dof_%s.png', scen.name)));
    close(f);
end

fclose(fileID);
fprintf('Wszystkie testy reguły Brysona dla 3DOF zakończone. Dane i wykresy zapisano w folderze: %s\n', out_dir);