clear; clc; close all;

out_dir = 'wizualizacja_dane';
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

SPEED_FACTOR = 1.0;   
ANIMATION_ON = true; 

g = 9.81;
C_g = (5/7) * g; 
R_plate = 0.3;  
H_base = 0.20;  
R_ball = 0.015;

A = [0 1 0 0; 
     0 0 C_g 0; 
     0 0 0 1; 
     0 0 0 0];
B = [0; 0; 0; 1];

Q = diag([500, 20, 10, 0.1]); 
R = 1;
K = care(A, B, Q, R);

angles = [0; 2*pi/3; 4*pi/3]; 
J_inv = [ -R_plate*sin(angles(1)), R_plate*cos(angles(1)), 1;
          -R_plate*sin(angles(2)), R_plate*cos(angles(2)), 1;
          -R_plate*sin(angles(3)), R_plate*cos(angles(3)), 1 ];

if ANIMATION_ON
    fig_anim = figure('Name', 'Wizualizacja 3DoF', 'Color', 'white', 'Position', [50 50 800 600]);
    view(45, 30); axis equal; grid on; hold on;
    xlim([-0.4 0.4]); ylim([-0.4 0.4]); zlim([0 0.5]);
    xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
    
    theta_circ = linspace(0, 2*pi, 60); 
    p_circ_x = R_plate * cos(theta_circ);
    p_circ_y = R_plate * sin(theta_circ);
    p_circ_z = zeros(size(p_circ_x)) + H_base;
    
    h_plate = patch(p_circ_x, p_circ_y, p_circ_z, 'c', 'FaceAlpha', 0.7, 'EdgeColor', 'b', 'LineWidth', 1.5);
    
    base_x = R_plate * cos(angles); 
    base_y = R_plate * sin(angles);
    plot3([base_x; base_x(1)], [base_y; base_y(1)], zeros(4,1), 'k--', 'LineWidth', 1); 
    
    h_legs = plot3(nan, nan, nan, 'k-', 'LineWidth', 3); 
    
    [sx, sy, sz] = sphere(12);
    h_ball_surf = surface(sx*R_ball, sy*R_ball, sz*R_ball + H_base + R_ball, ...
                          'FaceColor', 'r', 'EdgeColor', 'none');
                     
    h_title = title(sprintf('Symulacja 3DOF (Prędkość: %.1fx)', SPEED_FACTOR));
end

dt = 0.01;
T_sim = 6.0;
time = 0:dt:T_sim;

state_x = [0.12; 0.0; 0; 0];
state_y = [-0.10; 0.0; 0; 0];
state_z = 0; 

hist_x = []; 
hist_y = []; 
hist_h = [];
hist_alpha = [];
hist_beta = [];

for t_idx = 1:length(time)
    u_x = -K * state_x;
    u_y = -K * state_y;
    
    dx = A * state_x + B .* u_x;
    dy = A * state_y + B .* u_y;
    state_x = state_x + dx * dt;
    state_y = state_y + dy * dt;
    
    lim = deg2rad(12);
    curr_alpha = max(min(state_x(3), lim), -lim);
    curr_beta  = max(min(state_y(3), lim), -lim);
    
    pose = [curr_alpha; curr_beta; state_z];
    motor_h = J_inv * pose; 
    
    hist_x = [hist_x, state_x(1)];
    hist_y = [hist_y, state_y(1)];
    hist_h = [hist_h, motor_h];
    hist_alpha = [hist_alpha, curr_alpha];
    hist_beta = [hist_beta, curr_beta];
    
    if ANIMATION_ON
        if ~isvalid(fig_anim), break; end 
        
        new_plate_z = H_base + state_z - p_circ_y * curr_alpha + p_circ_x * curr_beta;
        set(h_plate, 'ZData', new_plate_z);
        attach_z = H_base + motor_h;
        
        lx = [base_x(1) base_x(1) NaN base_x(2) base_x(2) NaN base_x(3) base_x(3)];
        ly = [base_y(1) base_y(1) NaN base_y(2) base_y(2) NaN base_y(3) base_y(3)];
        lz = [0 attach_z(1) NaN 0 attach_z(2) NaN 0 attach_z(3)];
        set(h_legs, 'XData', lx, 'YData', ly, 'ZData', lz);
        
        ball_z = H_base + state_z - state_x(1)*curr_alpha + state_y(1)*curr_beta + R_ball;
        set(h_ball_surf, 'XData', sx*R_ball + state_x(1), ...
                         'YData', sy*R_ball + state_y(1), ...
                         'ZData', sz*R_ball + ball_z);
                     
        drawnow limitrate;
        pause(dt / SPEED_FACTOR);
        
        if sqrt(state_x(1)^2 + state_y(1)^2) > R_plate
            set(h_title, 'String', 'PIŁKA SPADŁA Z PŁYTY!', 'Color', 'r');
            pause(1);
            break;
        end
    end
end

if ~isempty(hist_x)
    sim_time = time(1:length(hist_x))';
    v_x = hist_x';
    v_y = hist_y';
    v_alpha = hist_alpha';
    v_beta = hist_beta';
    v_h1 = hist_h(1,:)';
    v_h2 = hist_h(2,:)';
    v_h3 = hist_h(3,:)';
    
    data_table = table(sim_time, v_x, v_y, v_alpha, v_beta, v_h1, v_h2, v_h3, ...
        'VariableNames', {'Czas', 'Pozycja_X', 'Pozycja_Y', 'Kat_Alpha', 'Kat_Beta', 'Silownik_1', 'Silownik_2', 'Silownik_3'});
    
    writetable(data_table, fullfile(out_dir, 'symulacja_3dof_dane.csv'));
end

info_x = calc_metrics(hist_x, time(1:length(hist_x)));
info_y = calc_metrics(hist_y, time(1:length(hist_x)));

figure('Name', 'Analiza 3DoF', 'Color', 'white', 'Position', [900 100 900 600]);
sgtitle('Wyniki symulacji układu 3DoF (Ball & Plate)', 'FontSize', 14, 'FontWeight', 'bold');

subplot(2, 2, [1 3]);
plot(0, 0, 'rx', 'MarkerSize', 10, 'LineWidth', 2); hold on;
plot(hist_x(1), hist_y(1), 'go', 'MarkerSize', 6, 'LineWidth', 2);
plot(hist_x, hist_y, 'b-', 'LineWidth', 2);
theta_d = linspace(0, 2*pi, 100);
plot(R_plate*cos(theta_d), R_plate*sin(theta_d), 'k--', 'LineWidth', 1.2);
title('Trajektoria piłki na płycie XY'); xlabel('X [m]'); ylabel('Y [m]'); 
legend('Cel', 'Start', 'Trajektoria', 'Krawędź płyty', 'Location', 'best');
axis equal; grid on; xlim([-0.35 0.35]); ylim([-0.35 0.35]);

subplot(2, 2, 2);
plot(time(1:length(hist_x)), hist_x, 'b', 'LineWidth', 1.5); hold on; 
plot(time(1:length(hist_x)), hist_y, 'g', 'LineWidth', 1.5);
title('Współrzędne pozycji piłki'); xlabel('Czas [s]'); ylabel('Pozycja [m]'); 
legend('X(t)', 'Y(t)', 'Location', 'best'); grid on;

subplot(2, 2, 4);
plot(time(1:length(hist_x)), hist_h(1,:)*1000, 'r', 'LineWidth', 1.2); hold on;
plot(time(1:length(hist_x)), hist_h(2,:)*1000, 'g', 'LineWidth', 1.2); 
plot(time(1:length(hist_x)), hist_h(3,:)*1000, 'b', 'LineWidth', 1.2);
title('Skok pracy siłowników (3 ramiona)'); xlabel('Czas [s]'); ylabel('\Delta h [mm]'); 
legend('Siłownik 1', 'Siłownik 2', 'Siłownik 3', 'Location', 'best'); grid on;

function info = calc_metrics(signal, t)
    dt = t(2) - t(1);
    r0 = abs(signal(1));
    if r0 < 1e-4, r0 = 1e-4; end
    threshold = max(0.001, 0.05 * r0);
    idx = find(abs(signal) > threshold, 1, 'last');
    if isempty(idx), info.Ts = 0; else, info.Ts = t(idx); end
    info.Ess = abs(signal(end));
    info.IAE = sum(abs(signal)) * dt;
end