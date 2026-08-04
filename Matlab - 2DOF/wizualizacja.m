clear; clc; close all;

SPEED_FACTOR = 1;     
ANIMATION_ON = true; 
SAVE_GIF = true;              

% Utworzenie folderu na dane i wizualizacje, jeśli nie istnieje
output_folder = 'wizualizacja_dane';
if ~isfolder(output_folder)
    mkdir(output_folder);
end

gif_filename = fullfile(output_folder, '2dof_sim.gif');
frame_skip = 3;               
frame_count = 0;

if SAVE_GIF && isfile(gif_filename)
    delete(gif_filename);
end

g = 9.81;
C_g = (5/7) * g; 
L_x = 0.30; 
L_y = 0.30; 
H_base = 0.15;  
R_ball = 0.015; 

A = [0 1 0 0; 0 0 C_g 0; 0 0 0 1; 0 0 0 0];
B = [0; 0; 0; 1];

Q = diag([500, 20, 10, 0.1]); 
R = 1;
K = care(A, B, Q, R);

if ANIMATION_ON
    fig_anim = figure('Name', 'Wizualizacja 2DOF', 'Color', 'white', 'Position', [50 50 800 600]);
    view(35, 25); axis equal; grid on; hold on;
    xlim([-0.25 0.25]); ylim([-0.25 0.25]); zlim([0 0.35]);
    xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
    title(sprintf('Symulacja 2DoF (Prędkość: %.1fx)', SPEED_FACTOR));
    
    plot3([-0.2 0.2], [0 0], [0 0], 'k-'); 
    plot3([0 0], [-0.2 0.2], [0 0], 'k-'); 
    
    [sx, sy, sz] = cylinder(0.02);
    surf(sx, sy, sz * H_base, 'FaceColor', [0.2 0.2 0.2], 'EdgeColor', 'none');
    
    plate_local_x = [-L_x/2, L_x/2, L_x/2, -L_x/2];
    plate_local_y = [-L_y/2, -L_y/2, L_y/2, L_y/2];
    plate_local_z = zeros(1,4);
    
    h_plate = patch(plate_local_x, plate_local_y, plate_local_z + H_base, ...
                    'c', 'FaceAlpha', 0.8, 'EdgeColor', 'k', 'LineWidth', 2);
    
    [bx, by, bz] = sphere(12);
    h_ball_surf = surface(bx*R_ball, by*R_ball, bz*R_ball + H_base + R_ball, ...
                          'FaceColor', 'r', 'EdgeColor', 'none');
    
    light('Position', [1 -1 2], 'Style', 'infinite');
    lighting gouraud;
    material shiny;
end

dt = 0.01;
T_sim = 6.0;
time = 0:dt:T_sim;

state_x = [-0.14; 0.1; 0; 0]; 
state_y = [0.11; 0.1; 0; 0];
hist_x = []; hist_y = [];
hist_ux = []; hist_uy = [];
hist_alpha = []; hist_beta = [];

for t = time
    u_x = -K * state_x;
    u_y = -K * state_y;
    
    lim = deg2rad(15);
    alpha = max(min(state_x(3), lim), -lim);
    beta  = max(min(state_y(3), lim), -lim);
    
    hist_x = [hist_x, state_x(1)];
    hist_y = [hist_y, state_y(1)];
    hist_ux = [hist_ux, u_x];
    hist_uy = [hist_uy, u_y];
    hist_alpha = [hist_alpha, alpha];
    hist_beta = [hist_beta, beta];
    
    is_fallen = abs(state_x(1)) > L_x/2 || abs(state_y(1)) > L_y/2;
    if ANIMATION_ON
        if ~isvalid(fig_anim), break; end
        
        Z_verts = H_base - plate_local_x * alpha - plate_local_y * beta; 
        set(h_plate, 'ZData', Z_verts);
        
        ball_z = H_base - state_x(1)*alpha - state_y(1)*beta + R_ball;
        set(h_ball_surf, 'XData', bx*R_ball + state_x(1), ...
                         'YData', by*R_ball + state_y(1), ...
                         'ZData', bz*R_ball + ball_z);
        
        if is_fallen
            title('Piłka spadła!'); 
        end
                     
        drawnow
        
        if SAVE_GIF
            frame_count = frame_count + 1;
            if mod(frame_count, frame_skip) == 0 || is_fallen 
                frame = getframe(fig_anim); 
                im = frame2im(frame);
                
                [imind, cm] = rgb2ind(im, 256, 'nodither'); 
                
                delay = dt * frame_skip / SPEED_FACTOR; 
                
                if ~exist('gif_started', 'var')
                    imwrite(imind, cm, gif_filename, 'gif', 'Loopcount', inf, 'DelayTime', delay);
                    gif_started = true;
                else
                    imwrite(imind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', delay);
                end
            end
        end
    end
    
    if is_fallen
        disp('Przerwano: piłka spadła.');
        break;
    end
    
    dx = A * state_x + B .* u_x;
    dy = A * state_y + B .* u_y;
    
    state_x = state_x + dx * dt;
    state_y = state_y + dy * dt;
end

if ~isempty(hist_x)
    time_data = time(1:length(hist_x))';
    sim_data = table(time_data, hist_x', hist_y', hist_alpha', hist_beta', ...
        'VariableNames', {'Czas_s', 'Pozycja_X_m', 'Pozycja_Y_m', 'Kat_Alpha_rad', 'Kat_Beta_rad'});
    writetable(sim_data, fullfile(output_folder, 'symulacja_dane.csv'));
end

if isempty(hist_x), return; end
time = time(1:length(hist_x));
info_x = calc_metrics(hist_x, time);
info_y = calc_metrics(hist_y, time);

figure('Name', 'Analiza 2DOF', 'Color', 'white', 'Position', [900 100 900 600]);
sgtitle('Wyniki symulacji 2DOF', 'FontSize', 14, 'FontWeight', 'bold');

subplot(2, 2, [1 3]);
plot(0, 0, 'rx', 'MarkerSize', 10, 'LineWidth', 2); hold on;
plot(hist_x(1), hist_y(1), 'go', 'MarkerSize', 6, 'LineWidth', 2);
plot(hist_x, hist_y, 'b-', 'LineWidth', 2);
rectangle('Position', [-L_x/2, -L_y/2, L_x, L_y], 'EdgeColor', 'k', 'LineStyle', '--');
title('Trajektoria na płycie'); xlabel('X [m]'); ylabel('Y [m]'); axis equal; grid on;
legend('Cel', 'Start', 'Trajektoria', 'Krawędź płyty');

subplot(2, 2, 2);
plot(time, hist_x, 'b', 'LineWidth', 1.5); hold on;
plot(time, hist_y, 'g', 'LineWidth', 1.5);
yline(0, 'k--');
title('Pozycje w czasie'); xlabel('Czas [s]'); ylabel('Pozycja [m]');
legend('X', 'Y'); grid on;

subplot(2, 2, 4);
plot(time, rad2deg(hist_alpha), 'b', 'LineWidth', 1.5); hold on;
plot(time, rad2deg(hist_beta), 'g', 'LineWidth', 1.5);
title('Kąty nachylenia płyty');
legend('\alpha (oś X)', '\beta (oś Y)');
ylabel('Kąt [deg]'); xlabel('Czas [s]'); grid on;

function info = calc_metrics(signal, t)
    threshold = max(0.001, 0.02 * abs(signal(1))); 
    idx = find(abs(signal) > threshold, 1, 'last');
    if isempty(idx), info.Ts = 0; else, info.Ts = t(idx); end
    info.Ess = abs(signal(end));
end