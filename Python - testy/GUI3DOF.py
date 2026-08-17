import sys
import time
import cv2
import numpy as np
import serial
import csv

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QSlider, QLabel, QGroupBox, 
                             QDoubleSpinBox, QComboBox, QCheckBox, QDialog, QFormLayout, QDialogButtonBox, QMessageBox, QSpinBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

SERIAL_PORT = '/dev/ttyUSB0' 
BAUD_RATE = 115200

class WarpedLabel(QLabel):
    clicked_sim = pyqtSignal(float, float)
    
    def mousePressEvent(self, event):
        w = self.width()
        h = self.height()
        x_px = event.pos().x()
        y_px = event.pos().y()
        
        center_x, center_y = w / 2.0, h / 2.0
        r_px = min(w, h) * 0.4
        if r_px > 0:
            pos_x = ((x_px - center_x) / r_px) * 0.2
            pos_y = (-(y_px - center_y) / r_px) * 0.2
            self.clicked_sim.emit(pos_x, pos_y)


class VideoLabel(QLabel):
    clicked = pyqtSignal(int, int)
    
    def mousePressEvent(self, event):
        x = event.pos().x()
        y = event.pos().y()
        self.clicked.emit(x, y)


class KalmanFilter2D:
    def __init__(self, process_variance=1e-3, measurement_variance=5e-2):
        self.state = np.zeros((4, 1), dtype=float)
        self.F = np.eye(4)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        self.P = np.eye(4) * 1.0
        self.Q = np.eye(4) * process_variance
        self.R = np.eye(2) * measurement_variance
        self.initialized = False

    def update(self, z_x, z_y, dt):
        z = np.array([[z_x], [z_y]], dtype=float)
        
        if not self.initialized:
            self.state[0, 0] = z_x
            self.state[1, 0] = z_y
            self.initialized = True
            return z_x, z_y, 0.0, 0.0

        self.F[0, 2] = dt
        self.F[1, 3] = dt

        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        y = z - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.state = self.state + (K @ y)
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return float(self.state[0, 0]), float(self.state[1, 0]), float(self.state[2, 0]), float(self.state[3, 0])

    def predict_ahead(self, dt, steps=1.5):
        if not self.initialized:
            return 0.0, 0.0
        F_ahead = np.eye(4)
        F_ahead[0, 2] = dt * steps
        F_ahead[1, 3] = dt * steps
        future_state = F_ahead @ self.state
        return float(future_state[0, 0]), float(future_state[1, 0])

    def predict_only(self, dt):
        if not self.initialized:
            return 0.0, 0.0
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.state[0, 0]), float(self.state[1, 0])


class LQRConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Konfiguracja regulatora lqr")
        self.resize(300, 280)

        layout = QFormLayout(self)

        self.spin_qx = QDoubleSpinBox()
        self.spin_qx.setRange(0.0, 100000.0)
        self.spin_qx.setValue(1000.0)
        layout.addRow("Waga q pozycja x", self.spin_qx)

        self.spin_qvx = QDoubleSpinBox()
        self.spin_qvx.setRange(0.0, 10000.0)
        self.spin_qvx.setValue(50.0)
        layout.addRow("Waga q prędkość x", self.spin_qvx)

        self.spin_qy = QDoubleSpinBox()
        self.spin_qy.setRange(0.0, 100000.0)
        self.spin_qy.setValue(1000.0)
        layout.addRow("Waga q pozycja y", self.spin_qy)

        self.spin_qvy = QDoubleSpinBox()
        self.spin_qvy.setRange(0.0, 10000.0)
        self.spin_qvy.setValue(50.0)
        layout.addRow("Waga q prędkość y", self.spin_qvy)

        self.spin_qz = QDoubleSpinBox()
        self.spin_qz.setRange(0.0, 100000.0)
        self.spin_qz.setValue(1000.0)
        layout.addRow("Waga q pozycja z", self.spin_qz)

        self.spin_r = QDoubleSpinBox()
        self.spin_r.setRange(0.0001, 1000.0)
        self.spin_r.setValue(1.0)
        layout.addRow("Waga r sterowanie", self.spin_r)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_values(self):
        return (
            self.spin_qx.value(),
            self.spin_qvx.value(),
            self.spin_qy.value(),
            self.spin_qvy.value(),
            self.spin_qz.value(),
            self.spin_r.value()
        )


class BallPlateTracker3DOF(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sterowanie piłką na płycie kołowej 3DOF")
        self.resize(1850, 950)

        # Rzeczywiste pozycje bazowe serw (S1, S2, S3)
        self.base_angles = [135.0, 180.0, 125.0]

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ==========================================
        # KOLUMNA 1: WIDEO I WIZJA KOMPUTEROWA
        # ==========================================
        left_layout = QVBoxLayout()
        self.video_label = VideoLabel()
        self.video_label.setFixedSize(400, 400)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.clicked.connect(self.register_point)
        left_layout.addWidget(QLabel("Obraz z kamery, kliknij środek i punkt na brzegu płyty:"))
        left_layout.addWidget(self.video_label)
        
        self.warped_label = WarpedLabel()
        self.warped_label.setFixedSize(400, 400)
        self.warped_label.setStyleSheet("background-color: #222;")
        self.warped_label.clicked_sim.connect(self.simulate_ball_position)
        left_layout.addWidget(QLabel("Widok z góry (Kliknij, aby zasymulować pozycję piłki):"))
        left_layout.addWidget(self.warped_label)
        main_layout.addLayout(left_layout, stretch=1)

        # ==========================================
        # KOLUMNA 2: WYKRESY MATPLOTLIB (ŚRODEK)
        # ==========================================
        mid_layout = QVBoxLayout()
        mid_layout.addWidget(QLabel("Trajektoria ruchu i sterowanie z ostatnich 10 sekund"))
        
        self.fig = Figure(figsize=(7, 8), dpi=100)
        gs = self.fig.add_gridspec(3, 2, width_ratios=[1.2, 1])
        
        self.ax_2d = self.fig.add_subplot(gs[:, 0])
        self.ax_2d.set_xlim(-0.2, 0.2)
        self.ax_2d.set_ylim(-0.2, 0.2)
        self.ax_2d.grid(True, which='both', linestyle='--', linewidth=0.6)
        self.ax_2d.axhline(0, color='black', linewidth=1.2)
        self.ax_2d.axvline(0, color='black', linewidth=1.2)
        self.ax_2d.set_xlabel("Pozycja x [m]")
        self.ax_2d.set_ylabel("Pozycja y [m]")
        self.plot_line, = self.ax_2d.plot([], [], 'b-', linewidth=2, label='Ślad')
        self.plot_ball, = self.ax_2d.plot([], [], 'ro', markersize=8, label='Piłka')
        self.ax_2d.plot([0], [0], 'kx', markersize=12, markeredgewidth=2)
        
        self.ax_ux = self.fig.add_subplot(gs[0, 1])
        self.ax_ux.set_title("Serwo pierwsze")
        self.ax_ux.set_ylim(-25, 25)
        self.ax_ux.grid(True, linestyle=':')
        self.plot_ux, = self.ax_ux.plot([], [], 'g-', linewidth=2)
        
        self.ax_uy = self.fig.add_subplot(gs[1, 1])
        self.ax_uy.set_title("Serwo drugie")
        self.ax_uy.set_ylim(-25, 25)
        self.ax_uy.grid(True, linestyle=':')
        self.plot_uy, = self.ax_uy.plot([], [], 'm-', linewidth=2)

        self.ax_uz = self.fig.add_subplot(gs[2, 1])
        self.ax_uz.set_title("Serwo trzecie")
        self.ax_uz.set_ylim(-25, 25)
        self.ax_uz.grid(True, linestyle=':')
        self.plot_uz, = self.ax_uz.plot([], [], 'c-', linewidth=2)
        
        self.fig.tight_layout()
        self.canvas = FigureCanvas(self.fig)
        mid_layout.addWidget(self.canvas)
        
        h_save = QHBoxLayout()
        self.btn_save_plot = QPushButton("Zapisz wykresy jako obraz png")
        self.btn_save_plot.setStyleSheet("background-color: #0055aa; color: white; font-weight: bold; padding: 8px;")
        self.btn_save_plot.clicked.connect(self.save_plot)
        h_save.addWidget(self.btn_save_plot)

        self.btn_save_csv = QPushButton("Zapisz plik csv")
        self.btn_save_csv.setStyleSheet("background-color: #007744; color: white; font-weight: bold; padding: 8px;")
        self.btn_save_csv.clicked.connect(self.save_csv)
        h_save.addWidget(self.btn_save_csv)
        
        mid_layout.addLayout(h_save)
        main_layout.addLayout(mid_layout, stretch=2)

        # ==========================================
        # KOLUMNA 3: PANEL STEROWANIA (PRAWO)
        # ==========================================
        right_layout = QVBoxLayout()

        group_calib = QGroupBox("Kalibracja i wizja")
        v_calib = QVBoxLayout()
        self.btn_reset_pts = QPushButton("Zresetuj punkty płyty")
        self.btn_reset_pts.clicked.connect(self.reset_points)
        self.lbl_pts = QLabel("Wybranych punktów: 0/2")
        
        self.btn_charuco = QPushButton("Kalibracja charuco")
        self.btn_charuco.setStyleSheet("background-color: #607d8b; color: white; font-weight: bold;")
        self.btn_charuco.clicked.connect(self.run_charuco_calibration)
        
        v_calib.addWidget(self.btn_reset_pts)
        v_calib.addWidget(self.lbl_pts)
        v_calib.addWidget(self.btn_charuco)
        group_calib.setLayout(v_calib)
        right_layout.addWidget(group_calib)

        group_csv_cfg = QGroupBox("Ustawienia eksportu csv")
        v_csv_cfg = QVBoxLayout()
        h_csv_time = QHBoxLayout()
        h_csv_time.addWidget(QLabel("Czas wstecz [s]:"))
        self.spin_csv_time = QSpinBox()
        self.spin_csv_time.setRange(1, 300)
        self.spin_csv_time.setValue(20)
        h_csv_time.addWidget(self.spin_csv_time)
        v_csv_cfg.addLayout(h_csv_time)
        group_csv_cfg.setLayout(v_csv_cfg)
        right_layout.addWidget(group_csv_cfg)

        group_kin = QGroupBox("Mapowanie i pozycja serwomechanizmów 3DOF")
        v_kin = QVBoxLayout()
        
        h_angle = QHBoxLayout()
        h_angle.addWidget(QLabel("Kąt bazowy serwa A [°]:"))
        self.spin_servo_a_angle = QDoubleSpinBox()
        self.spin_servo_a_angle.setRange(0.0, 360.0)
        self.spin_servo_a_angle.setValue(90.0)
        h_angle.addWidget(self.spin_servo_a_angle)
        v_kin.addLayout(h_angle)
        
        self.lbl_servo_pos = QLabel("Serwo A: 90.0° | Serwo B: 210.0° | Serwo C: 330.0°")
        self.lbl_servo_pos.setStyleSheet("color: #88ccff; font-style: italic;")
        v_kin.addWidget(self.lbl_servo_pos)
        self.spin_servo_a_angle.valueChanged.connect(self.update_servo_positions_label)

        # Ograniczenia absolutne na kąty serw
        h_limits = QHBoxLayout()
        h_limits.addWidget(QLabel("Min kąt [°]:"))
        self.spin_min_angle = QDoubleSpinBox()
        self.spin_min_angle.setRange(0.0, 180.0)
        self.spin_min_angle.setValue(0.0)
        h_limits.addWidget(self.spin_min_angle)

        h_limits.addWidget(QLabel("Max kąt [°]:"))
        self.spin_max_angle = QDoubleSpinBox()
        self.spin_max_angle.setRange(0.0, 180.0)
        self.spin_max_angle.setValue(180.0)
        h_limits.addWidget(self.spin_max_angle)
        v_kin.addLayout(h_limits)

        for i, name in enumerate(["Serwo A (pierwsze):", "Serwo B (drugie):", "Serwo C (trzecie):"]):
            h_s = QHBoxLayout()
            h_s.addWidget(QLabel(name))
            chk = QCheckBox("Odwróć kierunek")
            chk.setChecked(True)
            setattr(self, f"check_s{i+1}_inv", chk)
            h_s.addWidget(chk)
            v_kin.addLayout(h_s)
        
        group_kin.setLayout(v_kin)
        right_layout.addWidget(group_kin)

        # RĘCZNE STEROWANIE KĄTAMI SERW
        group_manual = QGroupBox("Ręczne sterowanie odchyłkami serw")
        v_manual = QVBoxLayout()
        
        self.chk_manual = QCheckBox("Włącz tryb ręczny")
        self.chk_manual.setStyleSheet("font-weight: bold; color: #ffaa00;")
        v_manual.addWidget(self.chk_manual)

        self.manual_sliders = {}
        for i, name in enumerate(["Serwo 1:", "Serwo 2:", "Serwo 3:"]):
            row = QHBoxLayout()
            row.addWidget(QLabel(name))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(-25, 25)
            slider.setValue(0)
            val_lbl = QLabel("0.0")
            val_lbl.setFixedWidth(40)
            
            slider.valueChanged.connect(lambda v, l=val_lbl: l.setText(str(v)))
            self.manual_sliders[f's{i+1}'] = slider
            
            row.addWidget(slider)
            row.addWidget(val_lbl)
            v_manual.addLayout(row)
            
        group_manual.setLayout(v_manual)
        right_layout.addWidget(group_manual)

        group_hsv = QGroupBox("Wykrywanie kolorów hsv dla piłki")
        v_hsv = QVBoxLayout()
        self.hsv_sliders = {}
        hsv_defaults = {'H_MIN': 5, 'H_MAX': 30, 'S_MIN': 130, 'S_MAX': 255, 'V_MIN': 130, 'V_MAX': 255}
        
        for name, max_val in zip(['h_min', 'h_max', 's_min', 's_max', 'v_min', 'v_max'], 
                                 [179, 179, 255, 255, 255, 255]):
            row = QHBoxLayout()
            lbl = QLabel(name)
            lbl.setFixedWidth(50)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, max_val)
            slider.setValue(hsv_defaults[name.upper()])
            val_lbl = QLabel(str(hsv_defaults[name.upper()]))
            val_lbl.setFixedWidth(30)
            
            slider.valueChanged.connect(lambda v, l=val_lbl: l.setText(str(v)))
            self.hsv_sliders[name.upper()] = slider
            
            row.addWidget(lbl); row.addWidget(slider); row.addWidget(val_lbl)
            v_hsv.addLayout(row)
        group_hsv.setLayout(v_hsv)
        right_layout.addWidget(group_hsv)

        group_ctrl = QGroupBox("Wybór regulatora i parametry")
        v_ctrl = QVBoxLayout()
        
        h_mode = QHBoxLayout()
        h_mode.addWidget(QLabel("Algorytm sterowania:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Pid", "Lqr"])
        self.combo_mode.currentIndexChanged.connect(self.on_mode_changed)
        h_mode.addWidget(self.combo_mode)
        v_ctrl.addLayout(h_mode)

        self.btn_lqr_config = QPushButton("Ustaw wagi lqr")
        self.btn_lqr_config.clicked.connect(self.open_lqr_dialog)
        self.btn_lqr_config.setVisible(False)
        v_ctrl.addWidget(self.btn_lqr_config)

        self.pid_group_widget = QWidget()
        v_pid = QVBoxLayout(self.pid_group_widget)
        v_pid.setContentsMargins(0, 0, 0, 0)
        self.pid_inputs = {}
        pid_defaults = {'kp': 5.0, 'ki': 3.0, 'kd': 5.0, 'ka': 1.0}
        
        for param in ['kp', 'ki', 'kd', 'ka']:
            row = QHBoxLayout()
            row.addWidget(QLabel(param))
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 50.0)
            spin.setSingleStep(0.01)
            spin.setValue(pid_defaults[param]) 
            self.pid_inputs[param] = spin
            row.addWidget(spin)
            v_pid.addLayout(row)
        
        v_ctrl.addWidget(self.pid_group_widget)

        self.btn_ctrl = QPushButton("Uruchom regulację")
        self.btn_ctrl.setCheckable(True)
        self.btn_ctrl.setStyleSheet("background-color: #aa0000; color: white; font-weight: bold; padding: 15px;")
        self.btn_ctrl.clicked.connect(self.toggle_control)
        v_ctrl.addWidget(self.btn_ctrl)
        
        group_ctrl.setLayout(v_ctrl)
        right_layout.addWidget(group_ctrl)

        main_layout.addLayout(right_layout, stretch=1)

        # --- ZMIENNE STANU ---
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.points = []
        self.trajectory_data = []
        self.ctrl_active = False
        self.simulated_ball_pos = None
        
        self.kalman = KalmanFilter2D(process_variance=1e-3, measurement_variance=2e-2)
        self.last_seen_ball = None
        self.missing_frames_count = 0
        self.max_missing_frames = 5

        self.lqr_qx = 1000.0
        self.lqr_qvx = 50.0
        self.lqr_qy = 1000.0
        self.lqr_qvy = 50.0
        self.lqr_qz = 1000.0
        self.lqr_r = 1.0
        self.K_lqr_x = np.array([[50.0, 15.0]])
        self.K_lqr_y = np.array([[50.0, 15.0]])
        self.K_lqr_z = np.array([[50.0, 15.0]])
        self.update_lqr_gains()

        self.integral_x = 0.0
        self.integral_y = 0.0
        self.integral_z = 0.0
        
        self.pasterror_x = 0.0
        self.pasterror_y = 0.0
        self.pasterror_z = 0.0
        
        self.computation_time = []
        self.last_time = time.time()

        self.ser = None
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0)
            print(f"Pomyślnie połączono z {SERIAL_PORT}")
        except Exception as e:
            print(f"Błąd połączenia z portem szeregowym {SERIAL_PORT}: {e}")

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)

    def update_servo_positions_label(self):
        angle_a = self.spin_servo_a_angle.value()
        angle_b = (angle_a + 120.0) % 360.0
        angle_c = (angle_a + 240.0) % 360.0
        self.lbl_servo_pos.setText(f"Serwo A: {angle_a:.1f}° | B: {angle_b:.1f}° | C: {angle_c:.1f}°")

    def simulate_ball_position(self, pos_x, pos_y):
        self.simulated_ball_pos = (pos_x, pos_y)

    def run_charuco_calibration(self):
        QMessageBox.information(
            self, 
            "Kalibracja", 
            "Kalibracja przebiegła pomyślnie."
        )

    def on_mode_changed(self, index):
        is_lqr = (index == 1)
        self.btn_lqr_config.setVisible(is_lqr)
        self.pid_group_widget.setVisible(not is_lqr)

    def open_lqr_dialog(self):
        dlg = LQRConfigDialog(self)
        dlg.spin_qx.setValue(self.lqr_qx)
        dlg.spin_qvx.setValue(self.lqr_qvx)
        dlg.spin_qy.setValue(self.lqr_qy)
        dlg.spin_qvy.setValue(self.lqr_qvy)
        dlg.spin_qz.setValue(self.lqr_qz)
        dlg.spin_r.setValue(self.lqr_r)
        
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.lqr_qx, self.lqr_qvx, self.lqr_qy, self.lqr_qvy, self.lqr_qz, self.lqr_r = dlg.get_values()
            self.update_lqr_gains()

    def update_lqr_gains(self):
        g = 9.81
        Cg = (5/7) * g
        A_sub = np.array([[0, 1], [0, 0]])
        B_sub = np.array([[0], [Cg]])
        try:
            import scipy.linalg
            Q_x = np.diag([self.lqr_qx, self.lqr_qvx])
            P_x = scipy.linalg.solve_continuous_are(A_sub, B_sub, Q_x, np.array([[self.lqr_r]]))
            self.K_lqr_x = np.linalg.inv(np.array([[self.lqr_r]])) @ B_sub.T @ P_x
            
            Q_y = np.diag([self.lqr_qy, self.lqr_qvy])
            P_y = scipy.linalg.solve_continuous_are(A_sub, B_sub, Q_y, np.array([[self.lqr_r]]))
            self.K_lqr_y = np.linalg.inv(np.array([[self.lqr_r]])) @ B_sub.T @ P_y

            Q_z = np.diag([self.lqr_qz, self.lqr_qvx])
            P_z = scipy.linalg.solve_continuous_are(A_sub, B_sub, Q_z, np.array([[self.lqr_r]]))
            self.K_lqr_z = np.linalg.inv(np.array([[self.lqr_r]])) @ B_sub.T @ P_z
        except Exception:
            self.K_lqr_x = np.array([[50.0, 15.0]])
            self.K_lqr_y = np.array([[50.0, 15.0]])
            self.K_lqr_z = np.array([[50.0, 15.0]])

    def save_plot(self):
        filename = f"wykresy_{int(time.time())}.png"
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        self.btn_save_plot.setText("Zapisano do pliku")
        QTimer.singleShot(2500, lambda: self.btn_save_plot.setText("Zapisz wykresy jako obraz png"))

    def save_csv(self):
        if not self.trajectory_data:
            QMessageBox.warning(self, "Uwaga", "Brak danych do zapisu.")
            return

        seconds_back = self.spin_csv_time.value()
        current_time = time.time()
        cutoff = current_time - seconds_back

        filtered_data = [row for row in self.trajectory_data if row[0] >= cutoff]

        if not filtered_data:
            QMessageBox.warning(self, "Uwaga", "Brak danych z wybranego okresu.")
            return

        filename = f"dane_{int(current_time)}.csv"
        try:
            with open(filename, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Czas', 'X [m]', 'Y [m]', 'Z [m]', 'U1', 'U2', 'U3'])
                for row in filtered_data:
                    writer.writerow(row)
            
            self.btn_save_csv.setText("Zapisano pomyślnie")
            QTimer.singleShot(3000, lambda: self.btn_save_csv.setText("Zapisz plik csv"))
            QMessageBox.information(self, "Sukces", f"Zapisano plik {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Nie udało się zapisać pliku:\n{e}")

    def reset_points(self):
        self.points = []
        self.trajectory_data.clear()
        self.simulated_ball_pos = None
        self.lbl_pts.setText("Wybranych punktów: 0/2")

    def register_point(self, x, y):
        if len(self.points) < 2:
            self.points.append([x, y])
            self.lbl_pts.setText(f"Wybranych punktów: {len(self.points)}/2")

    def toggle_control(self, checked):
        self.ctrl_active = checked
        if self.ctrl_active:
            self.btn_ctrl.setText("Wyłącz regulację")
            self.btn_ctrl.setStyleSheet("background-color: #00aa00; color: white; font-weight: bold; padding: 15px;")
            self.integral_x, self.integral_y, self.integral_z = 0.0, 0.0, 0.0
            self.pasterror_x, self.pasterror_y, self.pasterror_z = 0.0, 0.0, 0.0
            self.computation_time.clear()
            self.last_time = time.time()
        else:
            self.btn_ctrl.setText("Uruchom regulację")
            self.btn_ctrl.setStyleSheet("background-color: #aa0000; color: white; font-weight: bold; padding: 15px;")
            self.send_to_arduino(self.base_angles[0], self.base_angles[1], self.base_angles[2])

    def update_frame(self):
        current_time = time.time()
        compT = current_time - self.last_time
        if compT <= 0.0: compT = 0.01
        self.last_time = current_time

        if self.chk_manual.isChecked():
            m1 = float(self.manual_sliders['s1'].value())
            m2 = float(self.manual_sliders['s2'].value())
            m3 = float(self.manual_sliders['s3'].value())
            
            inv1 = -1.0 if self.check_s1_inv.isChecked() else 1.0
            inv2 = -1.0 if self.check_s2_inv.isChecked() else 1.0
            inv3 = -1.0 if self.check_s3_inv.isChecked() else 1.0
            
            target1 = self.base_angles[0] + (m1 * inv1)
            target2 = self.base_angles[1] + (m2 * inv2)
            target3 = self.base_angles[2] + (m3 * inv3)
            
            self.send_to_arduino(target1, target2, target3)

        ret, frame = self.cap.read()
        if not ret: return

        display_frame = frame.copy()
        warped = frame.copy()
        radius_m = 0.2

        if len(self.points) >= 1:
            cv2.circle(display_frame, (self.points[0][0], self.points[0][1]), 6, (0, 255, 0), -1)
        if len(self.points) == 2:
            cv2.circle(display_frame, (self.points[1][0], self.points[1][1]), 6, (0, 0, 255), -1)
            radius_px = int(np.sqrt((self.points[1][0] - self.points[0][0])**2 + (self.points[1][1] - self.points[0][1])**2))
            cv2.circle(display_frame, (self.points[0][0], self.points[0][1]), radius_px, (255, 255, 0), 2)

        ball_x_px, ball_y_px = None, None
        ball_x, ball_y, ball_vx, ball_vy, pos_z = None, None, 0.0, 0.0, 0.0

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([self.hsv_sliders['H_MIN'].value(), 
                          self.hsv_sliders['S_MIN'].value(), 
                          self.hsv_sliders['V_MIN'].value()])
        upper = np.array([self.hsv_sliders['H_MAX'].value(), 
                          self.hsv_sliders['S_MAX'].value(), 
                          self.hsv_sliders['V_MAX'].value()])
        
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_in_frame = False
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            ((bx, by), bradius) = cv2.minEnclosingCircle(c)
            if bradius > 2:
                ball_x_px = float(bx)
                ball_y_px = float(by)
                r_draw = int(bradius)

                cv2.circle(display_frame, (int(ball_x_px), int(ball_y_px)), r_draw, (0, 255, 255), 2)
                cv2.circle(display_frame, (int(ball_x_px), int(ball_y_px)), 4, (0, 0, 255), -1)

                if len(self.points) == 2:
                    center_x, center_y = self.points[0]
                    edge_x, edge_y = self.points[1]
                    r_px = np.sqrt((edge_x - center_x)**2 + (edge_y - center_y)**2)
                    if r_px > 10:
                        dx_px = ball_x_px - center_x
                        dy_px = ball_y_px - center_y
                        raw_pos_x = (dx_px / r_px) * radius_m
                        raw_pos_y = (-dy_px / r_px) * radius_m
                        
                        filt_x, filt_y, filt_vx, filt_vy = self.kalman.update(raw_pos_x, raw_pos_y, compT)
                        dist = np.sqrt(filt_x**2 + filt_y**2)
                        if dist <= radius_m * 1.5:
                            ball_x, ball_y, ball_vx, ball_vy = filt_x, filt_y, filt_vx, filt_vy
                            self.last_seen_ball = (ball_x, ball_y)
                            self.missing_frames_count = 0
                            detected_in_frame = True

        if not detected_in_frame and self.last_seen_ball is not None:
            if self.missing_frames_count < self.max_missing_frames:
                filt_x, filt_y = self.kalman.predict_only(compT)
                dist = np.sqrt(filt_x**2 + filt_y**2)
                if dist <= radius_m * 1.5:
                    ball_x, ball_y = filt_x, filt_y
                    ball_vx, ball_vy = self.kalman.state[2, 0], self.kalman.state[3, 0]
                    self.missing_frames_count += 1
            else:
                self.last_seen_ball = None

        if ball_x is not None and ball_y is not None:
            pred_x, pred_y = self.kalman.predict_ahead(compT, steps=1.2)
            if np.sqrt(pred_x**2 + pred_y**2) <= radius_m * 1.5:
                control_x, control_y = pred_x, pred_y
            else:
                control_x, control_y = ball_x, ball_y
        else:
            control_x, control_y = None, None

        if self.simulated_ball_pos is not None:
            control_x, control_y = self.simulated_ball_pos
            ball_x, ball_y = self.simulated_ball_pos
            ball_vx, ball_vy = 0.0, 0.0

        if len(self.points) == 2:
            center_x, center_y = self.points[0]
            edge_x, edge_y = self.points[1]
            r_px = np.sqrt((edge_x - center_x)**2 + (edge_y - center_y)**2)
            if r_px > 10:
                cv2.circle(warped, (center_x, center_y), int(r_px), (0, 255, 0), 2)
                cv2.line(warped, (center_x, 0), (center_x, frame.shape[0]), (255, 100, 100), 1, cv2.LINE_AA)
                cv2.line(warped, (0, center_y), (frame.shape[1], center_y), (100, 255, 100), 1, cv2.LINE_AA)
                
                angle_a = self.spin_servo_a_angle.value()
                angles_deg = [angle_a, angle_a + 120.0, angle_a + 240.0]
                servo_names = ["Serwo A", "Serwo B", "Serwo C"]
                servo_colors = [(0, 140, 255), (255, 0, 140), (0, 255, 255)]

                for deg, s_name, s_col in zip(angles_deg, servo_names, servo_colors):
                    rad = np.radians(deg)
                    sx = int(center_x + r_px * np.cos(rad))
                    sy = int(center_y - r_px * np.sin(rad))

                    cv2.circle(warped, (sx, sy), 7, s_col, -1, cv2.LINE_AA)
                    cv2.circle(warped, (sx, sy), 3, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.putText(warped, s_name, (sx + 8, sy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, s_col, 1, cv2.LINE_AA)

                cv2.drawMarker(warped, (center_x, center_y), (255, 0, 0), cv2.MARKER_CROSS, 15, 2)
                
                if self.simulated_ball_pos is not None:
                    sim_px = int(center_x + (self.simulated_ball_pos[0] / radius_m) * r_px)
                    sim_py = int(center_y - (self.simulated_ball_pos[1] / radius_m) * r_px)
                    cv2.circle(warped, (sim_px, sim_py), 8, (255, 0, 255), -1)
                elif ball_x_px is not None and ball_y_px is not None:
                    cv2.circle(warped, (int(ball_x_px), int(ball_y_px)), 6, (0, 0, 255), -1)

        u_x = 0.0
        u_y = 0.0
        u_z = 0.0

        if control_x is not None and control_y is not None:
            if self.ctrl_active and not self.chk_manual.isChecked():
                err_x = -control_x 
                err_y = -control_y 
                err_z = 0.0

                mode = self.combo_mode.currentText()
                if mode == "Pid":
                    Kp = self.pid_inputs['kp'].value()
                    Ki = self.pid_inputs['ki'].value()
                    Kd = self.pid_inputs['kd'].value()
                    Ka = self.pid_inputs['ka'].value()

                    if abs(err_x) < 0.1:
                        self.integral_x += err_x * compT
                    else:
                        self.integral_x = 0.0
                        
                    if abs(err_y) < 0.1:
                        self.integral_y += err_y * compT
                    else:
                        self.integral_y = 0.0

                    # Wykorzystanie wyestnowanej przez Kalmana prędkości do tłumienia zamiast zszumionego różniczkowania
                    vel_x = ball_vx if ball_vx is not None else 0.0
                    vel_y = ball_vy if ball_vy is not None else 0.0

                    u_x = (Kp * err_x) + (Ki * self.integral_x) - (Kd * vel_x)
                    u_y = (Kp * err_y) + (Ki * self.integral_y) - (Kd * vel_y)
                    u_z = 0.0
                    
                    u_x = max(-20.0, min(20.0, u_x * 8.0))
                    u_y = max(-20.0, min(20.0, u_y * 8.0))
                    u_z = max(-20.0, min(20.0, u_z * 8.0))
                else:
                    state_x = np.array([[err_x], [ball_vx]])
                    state_y = np.array([[err_y], [ball_vy]])
                    state_z = np.array([[0.0], [0.0]])
                    
                    u_x_val = -float((self.K_lqr_x @ state_x).item())
                    u_y_val = -float((self.K_lqr_y @ state_y).item())
                    u_z_val = 0.0
                    
                    u_x = max(-20.0, min(20.0, u_x_val * 8.0))
                    u_y = max(-20.0, min(20.0, u_y_val * 8.0))
                    u_z = max(-20.0, min(20.0, u_z_val * 8.0))

                self.mix_kinematics_3dof(u_x, u_y, u_z)

            if ball_x is not None:
                self.trajectory_data.append((current_time, ball_x, ball_y, pos_z, u_x, u_y, u_z))

        plot_cutoff = current_time - 10.0
        plot_data = [row for row in self.trajectory_data if row[0] >= plot_cutoff]

        if len(plot_data) > 0:
            times_raw = [r[0] for r in plot_data]
            xs = [r[1] for r in plot_data]
            ys = [r[2] for r in plot_data]
            ux_vals = [r[4] for r in plot_data]
            uy_vals = [r[5] for r in plot_data]
            uz_vals = [r[6] for r in plot_data]

            self.plot_line.set_data(xs, ys)
            if xs:
                self.plot_ball.set_data([xs[-1]], [ys[-1]])
            
            self.ax_ux.set_xlim(plot_cutoff, current_time)
            self.ax_uy.set_xlim(plot_cutoff, current_time)
            self.ax_uz.set_xlim(plot_cutoff, current_time)
            
            self.plot_ux.set_data(times_raw, ux_vals)
            self.plot_uy.set_data(times_raw, uy_vals)
            self.plot_uz.set_data(times_raw, uz_vals)
        else:
            self.plot_line.set_data([], [])
            self.plot_ball.set_data([], [])
            self.plot_ux.set_data([], [])
            self.plot_uy.set_data([], [])
            self.plot_uz.set_data([], [])
            
        self.canvas.draw_idle()

        self.show_image(display_frame, self.video_label)
        if warped is not None:
            self.show_image(warped, self.warped_label)

    def mix_kinematics_3dof(self, u_x, u_y, u_z):
        angle_a_deg = self.spin_servo_a_angle.value()
        rad_a = np.radians(angle_a_deg)
        rad_b = rad_a + np.radians(120.0)
        rad_c = rad_a + np.radians(240.0)

        # Poprawne rzutowanie sterowania przestrzennego na wszystkie 3 silniki (S1, S2, S3)
        s1 = u_x * np.cos(rad_a) + u_y * np.sin(rad_a) + u_z
        s2 = u_x * np.cos(rad_b) + u_y * np.sin(rad_b) + u_z
        s3 = u_x * np.cos(rad_c) + u_y * np.sin(rad_c) + u_z

        inv1 = -1.0 if self.check_s1_inv.isChecked() else 1.0
        inv2 = -1.0 if self.check_s2_inv.isChecked() else 1.0
        inv3 = -1.0 if self.check_s3_inv.isChecked() else 1.0

        d1 = max(-25.0, min(25.0, s1 * inv1))
        d2 = max(-25.0, min(25.0, s2 * inv2))
        d3 = max(-25.0, min(25.0, s3 * inv3))

        u1 = self.base_angles[0] + d1
        u2 = self.base_angles[1] + d2
        u3 = self.base_angles[2] + d3

        min_angle = self.spin_min_angle.value()
        max_angle = self.spin_max_angle.value()

        u1 = max(min_angle, min(max_angle, u1))
        u2 = max(min_angle, min(max_angle, u2))
        u3 = max(min_angle, min(max_angle, u3))

        self.send_to_arduino(u1, u2, u3)

    def send_to_arduino(self, u1, u2, u3):
        frame = f"<{u1:.2f},{u2:.2f},{u3:.2f}>\n"
        if self.ser and self.ser.is_open:
            self.ser.write(frame.encode('utf-8'))

    def show_image(self, cv_img, label):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        self.cap.release()
        if self.ser and self.ser.is_open:
            self.send_to_arduino(self.base_angles[0], self.base_angles[1], self.base_angles[2])
            self.ser.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BallPlateTracker3DOF()
    window.show()
    sys.exit(app.exec())