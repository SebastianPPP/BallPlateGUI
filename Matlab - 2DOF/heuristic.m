clear; clc; close all;

%% =========================================================
%  SEKCJA FOLDERU WYNIKOWEGO
% =========================================================
out_dir = 'wyniki_heurystyka_dane';
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

txt_filename = fullfile(out_dir, 'wskazniki_jakosci_heurystyka.txt');
fileID = fopen(txt_filename, 'w', 'native', 'UTF-8');
fprintf(fileID, '===================================================================\n');
fprintf(fileID, 'WYNIKI SYMULACJI - OPTYMALIZACJA HEURYSTYCZNA (2DOF) I ZAPASY STABILNOŚCI\n');
fprintf(fileID, '===================================================================\n');
fprintf(fileID, 'Błąd liczony jako odległość od (0,0): r = sqrt(x^2 + y^2)\n');
fprintf(fileID, 'Energia całkowita układu: E = u_x^2 + u_y^2\n');
fprintf(fileID, '===================================================================\n\n');

%% =========================================================
%  SEKCJA PARAMETRÓW I KONFIGURACJI (NA GÓRZE)
% =========================================================
state_easy_x = [0.05; 0; 0; 0];  
state_easy_y = [0.05; 0; 0; 0];

state_hard_x = [0.12; 0.05; deg2rad(2); 0]; 
state_hard_y = [0.12; -0.05; deg2rad(-2); 0];

x0_opt_x = state_hard_x;
x0_opt_y = state_hard_y;

dt    = 0.01;
Tsim  = 6.0;
lim   = deg2rad(15);
Lx    = 0.30;  
Ly    = 0.30;
FALL_PENALTY = 1e4;

lb = [ 0  -2   0  -2  -3];
ub = [ 4   3   4   3   1];
n_vars = 5;

w1 = 1.0;    
w2 = 0.1;   
w3 = 0.3;    

N_p   = 50;  N_i   = 100; 
w_max = 0.9; w_min = 0.4;  
c1_p  = 2.0; c2_p  = 2.0;  
N_ants = 40; N_ia = 100;  
T_arch = 20; q_a  = 0.4; xi_a = 0.9; 
N_nsga  = 80; N_ins = 100; 
p_cross = 0.9; p_mut = 1/n_vars; 
eta_c   = 10; eta_m = 15;

g  = 9.81;  Cg = (5/7)*g;
A  = [0 1 0 0; 0 0 Cg 0; 0 0 0 1; 0 0 0 0];
B  = [0; 0; 0; 1];
C_mat = [1 0 0 0];
D_mat = 0;

%% =========================================================
%  FAZA 1: OPTYMALIZACJA WAG LQR (NA STANIE BAZOWYM)
% =========================================================
fprintf('=========================================================\n');
fprintf('FAZA 1: URUCHAMIANIE OPTYMALIZACJI HEURYSTYCZNEJ\n');
fprintf('=========================================================\n');

fprintf('=== Uruchamianie PSO... ===\n');
rng(42);
Vmax  = 0.3*(ub-lb);
pos = lb + rand(N_p,n_vars).*(ub-lb);
vel = -Vmax + rand(N_p,n_vars).*(2*Vmax);
pbest = pos; pbest_J = inf(N_p,1);
for i=1:N_p
    [J_,~,~,~,~,~,~,~] = sim(pos(i,:),A,B,x0_opt_x,x0_opt_y,dt,Tsim,lim,Lx,Ly,FALL_PENALTY,w1,w2,w3);
    pbest_J(i) = J_;
end
[gbest_J, gi] = min(pbest_J); gbest = pbest(gi,:);
pso_conv = zeros(N_i,1); pso_best_iter = zeros(N_i,1);
pso_param_history = zeros(N_i, n_vars);
for iter = 1:N_i
    w_inert = w_max - (w_max-w_min)*(iter-1)/(N_i-1);
    iter_best = inf;
    for i = 1:N_p
        r1=rand(1,n_vars); r2=rand(1,n_vars);
        vel(i,:) = w_inert*vel(i,:) + c1_p*r1.*(pbest(i,:)-pos(i,:)) + c2_p*r2.*(gbest-pos(i,:));
        vel(i,:) = max(min(vel(i,:),Vmax),-Vmax);
        pos(i,:) = max(min(pos(i,:)+vel(i,:),ub),lb);
        [J_,~,~,~,~,~,~,~] = sim(pos(i,:),A,B,x0_opt_x,x0_opt_y,dt,Tsim,lim,Lx,Ly,FALL_PENALTY,w1,w2,w3);
        if J_ < pbest_J(i), pbest_J(i)=J_; pbest(i,:)=pos(i,:); end
        if J_ < gbest_J,    gbest_J=J_;    gbest=pos(i,:); end
        iter_best = min(iter_best, J_);
    end
    pso_conv(iter) = gbest_J;
    pso_best_iter(iter) = iter_best;
    pso_param_history(iter, :) = gbest;
end
theta_pso = gbest;
fprintf('  -> PSO zakonczone powodzeniem.\n');

fprintf('=== Uruchamianie ACO... ===\n');
rng(7);
arch_pos  = lb + rand(T_arch,n_vars).*(ub-lb); arch_J    = inf(T_arch,1);
for i=1:T_arch
    [arch_J(i),~,~,~,~,~,~,~] = sim(arch_pos(i,:),A,B,x0_opt_x,x0_opt_y,dt,Tsim,lim,Lx,Ly,FALL_PENALTY,w1,w2,w3);
end
[arch_J,idx_s] = sort(arch_J); arch_pos=arch_pos(idx_s,:);
aco_conv = zeros(N_ia,1); aco_best_iter = zeros(N_ia,1);
aco_param_history = zeros(N_ia, n_vars);
for iter = 1:N_ia
    ranks  = (1:T_arch)'; w_p = exp(-((ranks-1).^2)/(2*q_a^2*T_arch^2)); w_p = w_p/sum(w_p);
    sigma = zeros(1,n_vars);
    for d=1:n_vars
        sigma(d) = xi_a * sum(abs(arch_pos(:,d)-mean(arch_pos(:,d))))/(T_arch-1+eps);
        sigma(d) = max(sigma(d), 1e-3*(ub(d)-lb(d)));
    end
    new_pos  = zeros(N_ants,n_vars); new_J    = inf(N_ants,1);
    iter_best = inf;
    for ant=1:N_ants
        k   = randsample(T_arch,1,true,w_p);
        th  = arch_pos(k,:) + sigma.*randn(1,n_vars); th  = max(min(th,ub),lb);
        new_pos(ant,:) = th;
        [new_J(ant),~,~,~,~,~,~,~] = sim(th,A,B,x0_opt_x,x0_opt_y,dt,Tsim,lim,Lx,Ly,FALL_PENALTY,w1,w2,w3);
        iter_best = min(iter_best, new_J(ant));
    end
    all_pos = [arch_pos; new_pos]; all_J   = [arch_J;   new_J];
    [all_J,idx_s] = sort(all_J);
    arch_J   = all_J(1:T_arch); arch_pos = all_pos(idx_s(1:T_arch),:);
    
    aco_conv(iter) = arch_J(1);
    aco_best_iter(iter) = iter_best;
    aco_param_history(iter, :) = arch_pos(1,:);
end
theta_aco = arch_pos(1,:);
fprintf('  -> ACO zakonczone powodzeniem.\n');

fprintf('=== Uruchamianie NSGA-II... ===\n');
rng(13);
pop = lb + rand(N_nsga,n_vars).*(ub-lb); F1  = zeros(N_nsga,1); F2 = zeros(N_nsga,1);
for i=1:N_nsga, [F1(i),F2(i)]=sim2(pop(i,:),A,B,x0_opt_x,x0_opt_y,dt,Tsim,lim,Lx,Ly,FALL_PENALTY); end
nsga_best_f1  = zeros(N_ins,1);
nsga_front_sz = zeros(N_ins,1);
nsga_param_history = zeros(N_ins, n_vars);
for iter = 1:N_ins
    idx_p = randperm(N_nsga); offspring = zeros(N_nsga,n_vars);
    for i=1:2:N_nsga-1
        p1=pop(idx_p(i),:); p2=pop(idx_p(i+1),:);
        if rand<p_cross, [c1i,c2i] = sbx(p1,p2,lb,ub,eta_c); else, c1i=p1; c2i=p2; end
        offspring(i,:)=pmut(c1i,lb,ub,p_mut,eta_m); offspring(i+1,:)=pmut(c2i,lb,ub,p_mut,eta_m);
    end
    if mod(N_nsga,2)==1, offspring(end,:)=pmut(pop(end,:),lb,ub,p_mut,eta_m); end
    oF1=zeros(N_nsga,1); oF2=zeros(N_nsga,1);
    for i=1:N_nsga, [oF1(i),oF2(i)]=sim2(offspring(i,:),A,B,x0_opt_x,x0_opt_y,dt,Tsim,lim,Lx,Ly,FALL_PENALTY); end
    comb=[pop;offspring]; cF1=[F1;oF1]; cF2=[F2;oF2];
    fronts = nds(cF1,cF2);
    new_pop=zeros(N_nsga,n_vars); nF1=zeros(N_nsga,1); nF2=zeros(N_nsga,1); filled=0;
    for fi=1:length(fronts)
        fi_idx=fronts{fi};
        if filled+length(fi_idx)<=N_nsga
            r=filled+1:filled+length(fi_idx); new_pop(r,:)=comb(fi_idx,:); nF1(r)=cF1(fi_idx); nF2(r)=cF2(fi_idx); filled=filled+length(fi_idx);
        else
            need=N_nsga-filled; cd_=crowd(cF1(fi_idx),cF2(fi_idx)); [~,cdi]=sort(cd_,'descend'); sel=fi_idx(cdi(1:need));
            r=filled+1:N_nsga; new_pop(r,:)=comb(sel,:); nF1(r)=cF1(sel); nF2(r)=cF2(sel); filled=N_nsga; break;
        end
        if filled==N_nsga, break; end
    end
    pop=new_pop; F1=nF1; F2=nF2;
    
    fr_cur=nds(F1,F2); pf=fr_cur{1};
    nsga_best_f1(iter) = min(F1(pf));
    nsga_front_sz(iter) = length(pf);
    
    f1n = F1(pf)/max(F1(pf)+eps); f2n = F2(pf)/max(F2(pf)+eps);
    [~,bp_cur] = min(f1n+f2n);
    nsga_param_history(iter, :) = pop(pf(bp_cur), :);
end
fr_fin=nds(F1,F2); pf_idx=fr_fin{1}; pf_F1=F1(pf_idx); pf_F2=F2(pf_idx); pf_pop=pop(pf_idx,:);
f1n_=pf_F1/max(pf_F1+eps); f2n_=pf_F2/max(pf_F2+eps); [~,bp]=min(f1n_+f2n_);
theta_nsga = pf_pop(bp,:);
fprintf('  -> NSGA-II zakonczone powodzeniem.\n\n');

%% =========================================================
%  FAZA 1.2: CHARAKTERYSTYKI CZĘSTOTLIWOŚCIOWE I ZAPASY STABILNOŚCI
%% =========================================================
methods = {'PSO', 'ACO', 'NSGA-II'};
thetas  = {theta_pso, theta_aco, theta_nsga};
c_pso  = [0.13 0.47 0.71]; c_aco  = [0.17 0.63 0.17]; c_nsga = [0.84 0.15 0.16];
colors = {c_pso, c_aco, c_nsga};

fig_bode_all = figure('Name', 'Zbiorcze charakterystyki Bode - Heurystyka', 'Position', [100 100 900, 700]);
fig_nyq_all  = figure('Name', 'Zbiorcze wykresy Nyquista - Heurystyka', 'Position', [150 150 900, 700]);

opt_bode = bodeoptions;
opt_bode.Grid = 'on';
opt_bode.PhaseVisible = 'on';
opt_bode.Title.String = 'Charakterystyka Bode - Optymalizacja Heurystyczna';
opt_bode.XLabel.String = 'Częstotliwość [rad/s]';

opt_nyq = nyquistoptions;
opt_nyq.Grid = 'on';
opt_nyq.Title.String = 'Wykres Nyquista - Optymalizacja Heurystyczna';
opt_nyq.XLabel.String = 'Część rzeczywista';
opt_nyq.YLabel.String = 'Część urojona';

for m = 1:length(methods)
    th = thetas{m};
    name = methods{m};
    q = 10.^th(1:4);
    r_ = 10^th(5);
    [~,~,K_gain] = care(A, B, diag(q), r_);
    
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
legend(methods, 'Location', 'southwest', 'Interpreter', 'none');
saveas(fig_bode_all, fullfile(out_dir, 'ZBIORCZY_Heurystyka_Bode.png'));
close(fig_bode_all);

figure(fig_nyq_all);
legend(methods, 'Location', 'northeast', 'Interpreter', 'none');
saveas(fig_nyq_all, fullfile(out_dir, 'ZBIORCZY_Heurystyka_Nyquist.png'));
close(fig_nyq_all);

%% =========================================================
%  FAZA 1.5: WYKRESY ZBIEŻNOŚCI I HISTORII PARAMETRÓW
%% =========================================================
fig_handles = [];
f1_h = figure('Name','Analiza Zbieżności Algorytmów','Color','white','Position',[30 30 1400 400]);
fig_handles = [fig_handles, f1_h];

subplot(1,3,1); hold on; grid on;
plot(1:N_i, pso_conv, '-', 'Color', c_pso, 'LineWidth', 2, 'DisplayName', 'Global Best');
plot(1:N_i, pso_best_iter, '--', 'Color', c_pso*0.6, 'LineWidth', 1, 'DisplayName', 'Iter Best');
set(gca, 'YScale', 'log'); title('PSO — Zbieżność kosztu J'); xlabel('Iteracja'); ylabel('J'); legend;

subplot(1,3,2); hold on; grid on;
plot(1:N_ia, aco_conv, '-', 'Color', c_aco, 'LineWidth', 2, 'DisplayName', 'Best w Archiwum');
plot(1:N_ia, aco_best_iter, '--', 'Color', c_aco*0.6, 'LineWidth', 1, 'DisplayName', 'Iter Best');
set(gca, 'YScale', 'log'); title('ACO — Zbieżność kosztu J'); xlabel('Iteracja'); ylabel('J'); legend;

subplot(1,3,3); hold on; grid on;
yyaxis left
plot(1:N_ins, nsga_best_f1, '-', 'Color', c_nsga, 'LineWidth', 2); ylabel('min f_1 (\int dist^2 dt)');
yyaxis right
plot(1:N_ins, nsga_front_sz, '--', 'Color', [0.5 0.2 0.8], 'LineWidth', 1.2); ylabel('Liczba punktów na fronte');
title('NSGA-II — Przebieg optymalizacji'); xlabel('Iteracja');

f2_h = figure('Name','Historia Zmian Parametrów (Wag LQR)','Color','white','Position',[30 480 1400 400]);
fig_handles = [fig_handles, f2_h];
labels = {'\log_{10}(q_1)', '\log_{10}(q_2)', '\log_{10}(q_3)', '\log_{10}(q_4)', '\log_{10}(r)'};

subplot(1,3,1); hold on; grid on;
for d = 1:n_vars, plot(1:N_i, pso_param_history(:,d), 'LineWidth', 1.5, 'DisplayName', labels{d}); end
title('PSO — Ewolucja wag gbest'); xlabel('Iteracja'); ylabel('Wartość parametru'); legend('Location','best');

subplot(1,3,2); hold on; grid on;
for d = 1:n_vars, plot(1:N_ia, aco_param_history(:,d), 'LineWidth', 1.5, 'DisplayName', labels{d}); end
title('ACO — Ewolucja wag lidera'); xlabel('Iteracja'); ylabel('Wartość parametru'); legend('Location','best');

subplot(1,3,3); hold on; grid on;
for d = 1:n_vars, plot(1:N_ins, nsga_param_history(:,d), 'LineWidth', 1.5, 'DisplayName', labels{d}); end
title('NSGA-II — Ewolucja pkt kompromisowego'); xlabel('Iteracja'); ylabel('Wartość parametru'); legend('Location','best');

%% =========================================================
%  FAZA 2: TESTOWANIE ZOPTYMALIZOWANYCH WAG NA SYTUACJACH "PROSTA" I "TRUDNA"
% =========================================================
fprintf('=========================================================\n');
fprintf('FAZA 2: TESTOWANIE REGULATORÓW W ROŻNYCH SYTUACJACH\n');
fprintf('=========================================================\n');
test_scenarios = struct();
test_scenarios(1).name = 'Prosta';
test_scenarios(1).x0_x = state_easy_x;
test_scenarios(1).x0_y = state_easy_y;
test_scenarios(2).name = 'Trudna';
test_scenarios(2).x0_x = state_hard_x;
test_scenarios(2).x0_y = state_hard_y;

results_table = [];

for s = 1:length(test_scenarios)
    scen = test_scenarios(s);
    
    f_scen = figure('Name',['Weryfikacja - Sytuacja: ' scen.name],'Color','white','Position',[50 150 1500 450]);
    fig_handles = [fig_handles, f_scen];
    
    subplot(1,3,1); hold on; grid on; title(['Odległość od środka — ' scen.name]);
    xlabel('Czas [s]'); ylabel('r [m]');
    yline(0,'k--','LineWidth',0.8, 'HandleVisibility','off');
    
    subplot(1,3,2); hold on; grid on; title(['Sygnał sterujący u_x(t) — ' scen.name]);
    xlabel('Czas [s]'); ylabel('u [rad/s^2]');
    yline(0,'k--','LineWidth',0.8, 'HandleVisibility','off');
    
    subplot(1,3,3); hold on; grid on; title(['Trajektoria XY — ' scen.name]);
    rectangle('Position',[-Lx/2,-Ly/2,Lx,Ly],'EdgeColor','k','LineStyle','--','LineWidth',1.2,'HandleVisibility','off');
    plot(0,0,'rx','MarkerSize',12,'LineWidth',2,'DisplayName','Cel');
    xlabel('X [m]'); ylabel('Y [m]'); axis equal; xlim([-0.2 0.2]); ylim([-0.2 0.2]);
    
    line_styles = {'-', '--', ':'};
    for m = 1:3
        [~, ~, ~, ~, hx, hy, hux, t_out, h_alpha, h_beta, fallen, fallen_t] = sim_full(thetas{m}, A, B, scen.x0_x, scen.x0_y, dt, Tsim, lim, Lx, Ly, FALL_PENALTY, w1, w2, w3);
        d_ = sqrt(hx.^2 + hy.^2);
        dist0 = sqrt(scen.x0_x(1)^2 + scen.x0_y(1)^2);
        
        thr = max(0.001, 0.02 * dist0);
        idx_ts = find(d_ > thr, 1, 'last');
        Ts_val = 0; if ~isempty(idx_ts), Ts_val = t_out(idx_ts); end
        
        Mp_val = max(0, (max(d_) - dist0) / dist0 * 100);
        IAE_val = sum(d_) * dt;
        Energia_val = sum(hux.^2 * 2) * dt; 
        
        q = 10.^thetas{m}(1:4); r_ = 10^thetas{m}(5);
        [~,~,K_] = care(A,B,diag(q),r_);
        
        sys_cl = ss(A - B*K_, B, C_mat, D_mat);
        [Gm, Pm, ~, ~] = margin(sys_cl);
        Gm_dB = 20*log10(Gm);
        if isinf(Gm_dB), Gm_dB = 999; end
        
        fprintf(fileID, '### Sytuacja: %s | Metoda: %s ###\n', scen.name, methods{m});
        fprintf(fileID, '  Zapas fazy (Pm):        %7.2f deg\n', Pm);
        fprintf(fileID, '  Zapas amplitudy (Gm):   %7.2f dB\n', Gm_dB);
        if fallen
            fprintf(fileID, 'Wynik: Piłka spadła z płyty w czasie T = %.2f s\n\n', fallen_t);
        else
            fprintf(fileID, '  Czas regulacji (Ts):    %7.3f s\n', Ts_val);
            fprintf(fileID, '  Przeregulowanie (Mp):   %7.2f %%\n', Mp_val);
            fprintf(fileID, '  Wskaźnik IAE:           %7.4f\n', IAE_val);
            fprintf(fileID, '  Całkowita energia (E):  %7.4f\n\n', Energia_val);
        end
        
        sim_data = table(t_out', hx', hy', h_alpha', h_beta', hux', hux', ...
            'VariableNames', {'Czas_s', 'Pozycja_X_m', 'Pozycja_Y_m', 'Kat_Alpha_rad', 'Kat_Beta_rad', 'Sterowanie_Ux', 'Sterowanie_Uy'});
        writetable(sim_data, fullfile(out_dir, sprintf('heurystyka_%s_%s_dane.csv', scen.name, methods{m})));
        
        subplot(1,3,1); plot(t_out, d_, 'LineStyle', line_styles{m}, 'Color', colors{m}, 'LineWidth', 2, 'DisplayName', methods{m});
        subplot(1,3,2); plot(t_out, hux, 'LineStyle', line_styles{m}, 'Color', colors{m}, 'LineWidth', 2, 'DisplayName', methods{m});
        subplot(1,3,3); plot(hx, hy, 'LineStyle', line_styles{m}, 'Color', colors{m}, 'LineWidth', 1.8, 'DisplayName', methods{m});
        
        res.scenario = scen.name; res.method = methods{m}; 
        res.Ts = Ts_val; res.Mp = Mp_val; res.IAE = IAE_val; res.Energia = Energia_val;
        res.K = K_; res.R = r_;
        results_table = [results_table; res];
    end
    subplot(1,3,1); legend('Location','northeast','Interpreter','none');
    subplot(1,3,2); legend('Location','northeast','Interpreter','none');
    subplot(1,3,3); legend('Location','best','Interpreter','none');
    
    saveas(f_scen, fullfile(out_dir, sprintf('wykres_heurystyka_weryfikacja_%s.png', scen.name)));
end

fclose(fileID);

%% =========================================================
%  ZAPIS POZOSTAŁYCH WYKRESÓW DO PLIKÓW
%% =========================================================
fprintf('\nTrwa zapisywanie grafiki symulacyjnej do plików...\n');
fig_names = {'wykres_heurystyka_zbieznosc', 'wykres_heurystyka_ewolucja_wag', ...
             'wykres_heurystyka_weryfikacja_Prosta', 'wykres_heurystyka_weryfikacja_Trudna'};
for f_idx = 1:length(fig_handles)
    filename_png = fullfile(out_dir, sprintf('%s.png', fig_names{f_idx}));
    filename_fig = fullfile(out_dir, sprintf('%s.fig', fig_names{f_idx}));
    
    print(fig_handles(f_idx), filename_png, '-dpng', '-r300');
    saveas(fig_handles(f_idx), filename_fig);
end
fprintf('Wszystkie pliki i raporty dla optymalizacji heurystycznej 2DOF zostały zapisane w: %s\n', out_dir);

%% =========================================================
%  FUNKCJE POMOCNICZE
%% =========================================================
function [J, f1, f2, Ts, hx, hy, hux, t_out] = sim(theta, ...
        A, B, x0_x, x0_y, dt, Tsim, lim, Lx, Ly, FP, w1, w2, w3)
    q = 10.^theta(1:4);
    r = 10^theta(5);
    try [~,~,K] = care(A, B, diag(q), r); catch
        J=FP; f1=FP; f2=FP; Ts=Tsim; hx=[]; hy=[]; hux=[]; t_out=[]; return;
    end
    N = round(Tsim/dt)+1; time = (0:N-1)*dt;
    sx = x0_x; sy = x0_y; hx = zeros(1,N); hy = zeros(1,N); hux = zeros(1,N); fallen = false;
    for i = 1:N
        ux = -K*sx; uy = -K*sy; hx(i)=sx(1); hy(i)=sy(1); hux(i)=ux;
        if abs(sx(1))>Lx/2 || abs(sy(1))>Ly/2
            fallen=true; hx=hx(1:i); hy=hy(1:i); hux=hux(1:i); time=time(1:i); break;
        end
        sx = sx + (A*sx + B*ux)*dt; sy = sy + (A*sy + B*uy)*dt;
    end
    t_out = time; f1 = sum(hx.^2 + hy.^2)*dt; f2 = sum(hux.^2)*dt;
    dist_ = sqrt(hx.^2+hy.^2); dist0 = sqrt(x0_x(1)^2+x0_y(1)^2);
    idx_ts = find(dist_>max(0.002,0.02*dist0),1,'last');
    Ts = 0; if ~isempty(idx_ts), Ts=time(idx_ts); end
    J = w1*f1 + w2*f2 + w3*Ts; if fallen, J=J+FP; f1=f1+FP; end
end

function [J, f1, f2, Ts, hx, hy, hux, t_out, h_alpha, h_beta, fallen, fallen_t] = sim_full(theta, ...
        A, B, x0_x, x0_y, dt, Tsim, lim, Lx, Ly, FP, w1, w2, w3)
    q = 10.^theta(1:4);
    r = 10^theta(5);
    [~,~,K] = care(A, B, diag(q), r);
    N = round(Tsim/dt)+1; time = (0:N-1)*dt;
    sx = x0_x; sy = x0_y; 
    hx = zeros(1,N); hy = zeros(1,N); hux = zeros(1,N); 
    h_alpha = zeros(1,N); h_beta = zeros(1,N);
    fallen = false; fallen_t = NaN;
    for i = 1:N
        ux = -K*sx; uy = -K*sy; 
        alpha = max(min(sx(3), lim), -lim);
        beta  = max(min(sy(3), lim), -lim);
        hx(i)=sx(1); hy(i)=sy(1); hux(i)=ux;
        h_alpha(i)=alpha; h_beta(i)=beta;
        if abs(sx(1))>Lx/2 || abs(sy(1))>Ly/2
            fallen=true; fallen_t=time(i);
            hx(i:end)=NaN; hy(i:end)=NaN; hux(i:end)=NaN; 
            h_alpha(i:end)=NaN; h_beta(i:end)=NaN;
            break;
        end
        sx = sx + (A*sx + B*ux)*dt; sy = sy + (A*sy + B*uy)*dt;
    end
    valid = ~isnan(hx);
    t_out = time(valid); hx = hx(valid); hy = hy(valid); hux = hux(valid);
    h_alpha = h_alpha(valid); h_beta = h_beta(valid);
    f1 = sum(hx.^2 + hy.^2)*dt; f2 = sum(hux.^2)*dt;
    dist_ = sqrt(hx.^2+hy.^2); dist0 = sqrt(x0_x(1)^2+x0_y(1)^2);
    idx_ts = find(dist_>max(0.002,0.02*dist0),1,'last');
    Ts = 0; if ~isempty(idx_ts), Ts=t_out(idx_ts); end
    J = w1*f1 + w2*f2 + w3*Ts; if fallen, J=J+FP; end
end

function [f1,f2] = sim2(theta,A,B,x0_x,x0_y,dt,Tsim,lim,Lx,Ly,FP)
    [~,f1,f2,~,~,~,~,~] = sim(theta,A,B,x0_x,x0_y,dt,Tsim,lim,Lx,Ly,FP,1,0,0);
end

function [c1,c2] = sbx(p1,p2,lb,ub,eta)
    n=length(p1); c1=p1; c2=p2;
    for i=1:n
        if rand>0.5||abs(p2(i)-p1(i))<1e-10, continue; end
        y1=min(p1(i),p2(i)); y2=max(p1(i),p2(i));
        b1=1+2*(y1-lb(i))/(y2-y1+eps); b2=1+2*(ub(i)-y2)/(y2-y1+eps); u=rand;
        a1=2-b1^(-(eta+1)); a2=2-b2^(-(eta+1)); bq1 = u<=1/a1; bq2 = u<=1/a2;
        beta1 = bq1*(u*a1)^(1/(eta+1)) + (~bq1)*(1/(2-u*a1))^(1/(eta+1));
        beta2 = bq2*(u*a2)^(1/(eta+1)) + (~bq2)*(1/(2-u*a2))^(1/(eta+1));
        c1(i)=max(lb(i),min(ub(i), 0.5*(y1+y2)-0.5*beta1*(y2-y1)));
        c2(i)=max(lb(i),min(ub(i), 0.5*(y1+y2)+0.5*beta2*(y2-y1)));
    end
end

function x = pmut(x,lb,ub,pm,eta)
    n=length(x);
    for i=1:n
        if rand>pm, continue; end
        d1=(x(i)-lb(i))/(ub(i)-lb(i)+eps); d2=(ub(i)-x(i))/(ub(i)-lb(i)+eps); u=rand;
        if u<=0.5
            dq=(2*u+(1-2*u)*(1-d1)^(eta+1))^(1/(eta+1))-1;
        else
            dq=1-(2*(1-u)+2*(u-0.5)*(1-d2)^(eta+1))^(1/(eta+1));
        end
        x(i)=max(lb(i),min(ub(i), x(i)+dq*(ub(i)-lb(i))));
    end
end

function fronts = nds(F1,F2)
    n=length(F1); S=cell(n,1); dc=zeros(n,1);
    for i=1:n, S{i}=[]; end
    fronts={}; fronts{1}=[];
    for i=1:n
        for j=1:n
            if i==j, continue; end
            if F1(i)<=F1(j)&&F2(i)<=F2(j)&&(F1(i)<F1(j)||F2(i)<F2(j))
                S{i}=[S{i} j];
            elseif F1(j)<=F1(i)&&F2(j)<=F2(i)&&(F1(j)<F1(i)||F2(j)<F2(i))
                dc(i)=dc(i)+1;
            end
        end
        if dc(i)==0, fronts{1}=[fronts{1} i]; end
    end
    fi=1;
    while fi<=length(fronts) && ~isempty(fronts{fi})
        Q_=[];
        for i=fronts{fi}
            for j=S{i}
                dc(j)=dc(j)-1; if dc(j)==0, Q_=[Q_ j]; end
            end
        end
        fi=fi+1; if isempty(Q_), break; end
        fronts{fi}=Q_;
    end
end

function cd=crowd(F1,F2)
    n=length(F1); cd=zeros(n,1);
    for obj=1:2
        v=F1; if obj==2, v=F2; end
        [sv,ix]=sort(v); cd(ix(1))=inf; cd(ix(end))=inf;
        rng_=sv(end)-sv(1); if rng_<eps, continue; end
        for i=2:n-1, cd(ix(i))=cd(ix(i))+(sv(i+1)-sv(i-1))/rng_; end
    end
end