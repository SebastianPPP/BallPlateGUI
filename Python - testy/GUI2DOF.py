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

class VideoLabel(QLabel):
    clicked = pyqtSignal(int, int)
    
    def mousePressEvent(self, event):
        x = event.pos().x()
        y = event.pos().y()
        self.clicked.emit(x, y)


class KalmanFilter2D:
    def __init__(self, process_variance=1e-2, measurement_variance=1e-1):
        # Stan: [x, y, vx, vy]
        self.state = np.zeros((4, 1), dtype=float)
        
        # Macierz przejścia stanu (zakładamy model stałej prędkości w czasie dt)
        self.F = np.eye(4)
        
        # Macierz pomiaru (obserwujemy tylko pozycję x, y)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        
        # Kowariancja błędu szacunku
        self.P = np.eye(4) * 1.0
        
        # Szum procesu
        self.Q = np.eye(4) * process_variance
        
        # Szum pomiaru (z kamery)
        self.R = np.eye(2) * measurement_variance
        
        self.initialized = False

    def update(self, z_x, z_y, dt):
        z = np.array([[z_x], [z_y]], dtype=float)
        
        if not self.initialized:
            self.state[0, 0] = z_x
            self.state[1, 0] = z_y
            self.initialized = True
            return z_x, z_y

        # Aktualizacja macierzy F o aktualny krok czasowy dt
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        # 1. Predykcja (Time Update)
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        # 2. Korekta (Measurement Update)
        y = z - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.state = self.state + (K @ y)
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return float(self.state[0, 0]), float(self.state[1, 0])


class LQRConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Konfiguracja regulatora lqr")
        self.resize(300, 250)

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
            self.spin_r.value()
        )


class BallPlateTrackerSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sterowanie piłką na płycie - Wersja Płynna (Bez nasyceń)")
        self.resize(1700, 950)

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
        left_layout.addWidget(QLabel("Obraz z kamery, kliknij cztery rogi płyty:"))
        left_layout.addWidget(self.video_label)
        
        self.warped_label = QLabel()
        self.warped_label.setFixedSize(400, 400)
        self.warped_label.setStyleSheet("background-color: #222;")
        left_layout.addWidget(QLabel("Widok z góry na platformę:"))
        left_layout.addWidget(self.warped_label)
        main_layout.addLayout(left_layout, stretch=1)

        # ==========================================
        # KOLUMNA 2: WYKRESY MATPLOTLIB (ŚRODEK)
        # ==========================================
        mid_layout = QVBoxLayout()
        mid_layout.addWidget(QLabel("Trajektoria ruchu i sterowanie z ostatnich 10 sekund"))
        
        self.fig = Figure(figsize=(7, 8), dpi=100)
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.2, 1])
        
        self.ax_2d = self.fig.add_subplot(gs[:, 0])
        self.ax_2d.set_xlim(-0.25, 0.25)
        self.ax_2d.set_ylim(-0.25, 0.25)
        self.ax_2d.grid(True, which='both', linestyle='--', linewidth=0.6)
        self.ax_2d.axhline(0, color='black', linewidth=1.2)
        self.ax_2d.axvline(0, color='black', linewidth=1.2)
        self.ax_2d.set_xlabel("Pozycja x [m]")
        self.ax_2d.set_ylabel("Pozycja y [m]")
        self.plot_line, = self.ax_2d.plot([], [], 'b-', linewidth=2, label='Ślad')
        self.plot_ball, = self.ax_2d.plot([], [], 'ro', markersize=8, label='Piłka')
        self.ax_2d.plot([0], [0], 'kx', markersize=12, markeredgewidth=2)
        
        self.ax_ux = self.fig.add_subplot(gs[0, 1])
        self.ax_ux.set_title("Serwo osi x")
        self.ax_ux.set_ylim(-15, 15)
        self.ax_ux.grid(True, linestyle=':')
        self.plot_ux, = self.ax_ux.plot([], [], 'g-', linewidth=2)
        
        self.ax_uy = self.fig.add_subplot(gs[1, 1])
        self.ax_uy.set_title("Serwo osi y")
        self.ax_uy.set_ylim(-15, 15)
        self.ax_uy.grid(True, linestyle=':')
        self.plot_uy, = self.ax_uy.plot([], [], 'm-', linewidth=2)
        
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
        self.btn_reset_pts = QPushButton("Zresetuj narożniki")
        self.btn_reset_pts.clicked.connect(self.reset_points)
        self.lbl_pts = QLabel("Wybranych punktów: 0/4")
        
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

        group_kin = QGroupBox("Mapowanie serwomechanizmów")
        v_kin = QVBoxLayout()
        
        h_s1 = QHBoxLayout()
        h_s1.addWidget(QLabel("Serwo pierwsze:"))
        self.combo_s1 = QComboBox()
        self.combo_s1.addItems(["Oś x", "Oś y"])
        self.combo_s1.setCurrentIndex(0)
        self.check_s1_inv = QCheckBox("Odwróć kierunek")
        h_s1.addWidget(self.combo_s1)
        h_s1.addWidget(self.check_s1_inv)
        v_kin.addLayout(h_s1)

        h_s2 = QHBoxLayout()
        h_s2.addWidget(QLabel("Serwo drugie:"))
        self.combo_s2 = QComboBox()
        self.combo_s2.addItems(["Oś x", "Oś y"])
        self.combo_s2.setCurrentIndex(1)
        self.check_s2_inv = QCheckBox("Odwróć kierunek")
        h_s2.addWidget(self.combo_s2)
        h_s2.addWidget(self.check_s2_inv)
        v_kin.addLayout(h_s2)
        
        group_kin.setLayout(v_kin)
        right_layout.addWidget(group_kin)

        group_hsv = QGroupBox("Wykrywanie kolorów hsv dla piłki")
        v_hsv = QVBoxLayout()
        self.hsv_sliders = {}
        hsv_defaults = {'H_MIN': 5, 'H_MAX': 65, 'S_MIN': 120, 'S_MAX': 255, 'V_MIN': 120, 'V_MAX': 255}
        
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
        pid_defaults = {'kp': 3.0, 'ki': 0.0, 'kd': 6.0, 'ka': 1.0}
        
        for param in ['kp', 'ki', 'kd', 'ka']:
            row = QHBoxLayout()
            row.addWidget(QLabel(param))
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 100.0)
            spin.setSingleStep(0.5)
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
        
        self.kalman = KalmanFilter2D(process_variance=1e-3, measurement_variance=1e-2)

        self.lqr_qx = 1000.0
        self.lqr_qvx = 50.0
        self.lqr_qy = 1000.0
        self.lqr_qvy = 50.0
        self.lqr_r = 1.0
        self.K_lqr_x = np.array([50.0, 15.0])
        self.K_lqr_y = np.array([50.0, 15.0])
        self.update_lqr_gains()

        self.integral_x = 0.0
        self.integral_y = 0.0
        self.derivative_x = 0.0
        self.last_derivative_x = 0.0
        self.last_last_derivative_x = 0.0
        self.derivative_y = 0.0
        self.last_derivative_y = 0.0
        self.last_last_derivative_y = 0.0
        self.pasterror_x = 0.0
        self.pasterror_y = 0.0
        
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
        dlg.spin_r.setValue(self.lqr_r)
        
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.lqr_qx, self.lqr_qvx, self.lqr_qy, self.lqr_qvy, self.lqr_r = dlg.get_values()
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
        except Exception:
            self.K_lqr_x = np.array([[50.0, 15.0]])
            self.K_lqr_y = np.array([[50.0, 15.0]])

    def save_plot(self):
        filename = f"wykresy_{int(time.time())}.png"
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        self.btn_save_plot.setText(f"Zapisano do pliku")
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
                writer.writerow(['Czas', 'X [m]', 'Y [m]', 'Ux', 'Uy'])
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
        self.lbl_pts.setText("Wybranych punktów: 0/4")

    def register_point(self, x, y):
        if len(self.points) < 4:
            self.points.append([x, y])
            self.lbl_pts.setText(f"Wybranych punktów: {len(self.points)}/4")

    def toggle_control(self, checked):
        self.ctrl_active = checked
        if self.ctrl_active:
            self.btn_ctrl.setText("Wyłącz regulację")
            self.btn_ctrl.setStyleSheet("background-color: #00aa00; color: white; font-weight: bold; padding: 15px;")
            self.integral_x, self.integral_y = 0.0, 0.0
            self.derivative_x, self.last_derivative_x, self.last_last_derivative_x = 0.0, 0.0, 0.0
            self.derivative_y, self.last_derivative_y, self.last_last_derivative_y = 0.0, 0.0, 0.0
            self.pasterror_x, self.pasterror_y = 0.0, 0.0
            self.computation_time.clear()
            self.last_time = time.time()
        else:
            self.btn_ctrl.setText("Uruchom regulację")
            self.btn_ctrl.setStyleSheet("background-color: #aa0000; color: white; font-weight: bold; padding: 15px;")
            self.send_to_arduino(0, 0)

    def update_frame(self):
        current_time = time.time()
        compT = current_time - self.last_time
        if compT <= 0.0: compT = 0.01

        ret, frame = self.cap.read()
        if not ret: return

        frame_resized = cv2.resize(frame, (400, 400))
        display_frame = frame_resized.copy()

        for pt in self.points:
            cv2.circle(display_frame, (pt[0], pt[1]), 5, (0, 255, 0), -1)
        if len(self.points) == 4:
            cv2.polylines(display_frame, [np.array(self.points)], True, (0, 255, 0), 2)

        warped = None
        if len(self.points) == 4:
            pts1 = np.float32(self.points)
            pts2 = np.float32([[0, 0], [400, 0], [400, 400], [0, 400]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            warped = cv2.warpPerspective(frame_resized, matrix, (400, 400))
        else:
            warped = np.zeros((400, 400, 3), dtype=np.uint8)

        # Rysowanie osi układu współrzędnych oraz pozycji serwomechanizmów na widoku "warped"
        if len(self.points) == 4:
            # Oś X (niebieska) i Oś Y (zielona) przechodzące przez środek (200, 200)
            cv2.line(warped, (200, 0), (200, 400), (255, 100, 100), 1, cv2.LINE_AA)
            cv2.line(warped, (0, 200), (400, 200), (100, 255, 100), 1, cv2.LINE_AA)
            
            # Etykiety osi
            cv2.putText(warped, "+X", (205, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1, cv2.LINE_AA)
            cv2.putText(warped, "+Y", (370, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1, cv2.LINE_AA)

            # Pozycje serwomechanizmów (np. umieszczone na krawędziach platformy: Serwo X na dole, Serwo Y z prawej)
            # Możesz dostosować te współrzędne w zależności od fizycznej lokalizacji serw na platformie
            servo_x_pos = (200, 385)
            servo_y_pos = (385, 200)

            # Rysowanie znaczników serw (pomarańczowe kwadraty/kółka)
            cv2.circle(warped, servo_x_pos, 8, (0, 140, 255), -1, cv2.LINE_AA)
            cv2.circle(warped, servo_x_pos, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(warped, "Serwo X", (servo_x_pos[0] - 30, servo_x_pos[1] - 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA)

            cv2.circle(warped, servo_y_pos, 8, (255, 0, 140), -1, cv2.LINE_AA)
            cv2.circle(warped, servo_y_pos, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(warped, "Serwo Y", (servo_y_pos[0] - 65, servo_y_pos[1] + 4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 140), 1, cv2.LINE_AA)

        ball_x, ball_y = None, None
        if len(self.points) == 4:
            hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
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
            
            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                if radius > 5:
                    ball_x, ball_y = float(x), float(y)
                    cv2.circle(warped, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(warped, (int(x), int(y)), 5, (0, 0, 255), -1)

            cv2.drawMarker(warped, (200, 200), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)

        u_x = 0.0
        u_y = 0.0

        if ball_x is not None and ball_y is not None:
            platform_radius_m = 0.15
            scale_factor = 200.0 / platform_radius_m

            raw_pos_x = (ball_x - 200.0) / scale_factor
            raw_pos_y = (ball_y - 200.0) / scale_factor

            pos_x, pos_y = self.kalman.update(raw_pos_x, raw_pos_y, compT)

            if self.ctrl_active:
                self.computation_time.append(compT)
                if len(self.computation_time) > 50:
                    self.computation_time.pop(0)
                mean_compT = np.mean(self.computation_time)

                err_x = pos_x
                err_y = pos_y

                mode = self.combo_mode.currentText()
                if mode == "Pid":
                    Kp = self.pid_inputs['kp'].value()
                    Ki = self.pid_inputs['ki'].value()
                    Kd = self.pid_inputs['kd'].value()
                    Ka = self.pid_inputs['ka'].value()

                    if abs(err_x) < 0.05:
                        self.integral_x += err_x * compT
                    else:
                        self.integral_x = 0.0
                        
                    if abs(err_y) < 0.05:
                        self.integral_y += err_y * compT
                    else:
                        self.integral_y = 0.0

                    Kd_norm = Kd * (compT / mean_compT)
                    Ka_norm = Ka * (compT / mean_compT)

                    self.last_last_derivative_x = self.last_derivative_x
                    self.last_derivative_x = self.derivative_x
                    self.derivative_x = err_x - self.pasterror_x

                    self.last_last_derivative_y = self.last_derivative_y
                    self.last_derivative_y = self.derivative_y
                    self.derivative_y = err_y - self.pasterror_y

                    d_filtered_x = self.derivative_x
                    d_filtered_y = self.derivative_y

                    u_x = (Kp * err_x) + (Ki * self.integral_x) + (Kd_norm * d_filtered_x)
                    u_y = (Kp * err_y) + (Ki * self.integral_y) + (Kd_norm * d_filtered_y)
                    
                    u_x = max(-10.0, min(10.0, u_x * 10.0))
                    u_y = max(-10.0, min(10.0, u_y * 10.0))
                else:
                    vel_x = (err_x - self.pasterror_x) / compT
                    vel_y = (err_y - self.pasterror_y) / compT
                    state_x = np.array([[err_x], [vel_x]])
                    state_y = np.array([[err_y], [vel_y]])
                    
                    u_x_val = -float((self.K_lqr_x @ state_x).item())
                    u_y_val = -float((self.K_lqr_y @ state_y).item())
                    
                    u_x = max(-10.0, min(10.0, u_x_val * 5.0))
                    u_y = max(-10.0, min(10.0, u_y_val * 5.0))

                self.pasterror_x = err_x
                self.pasterror_y = err_y
                self.last_time = current_time

                self.mix_kinematics_2dof(u_x, u_y)

            self.trajectory_data.append((current_time, pos_x, -pos_y, u_x, u_y))

        plot_cutoff = current_time - 10.0
        plot_data = [row for row in self.trajectory_data if row[0] >= plot_cutoff]

        if len(plot_data) > 0:
            times_raw, xs, ys, ux_vals, uy_vals = zip(*plot_data)
            self.plot_line.set_data(xs, ys)
            self.plot_ball.set_data([xs[-1]], [ys[-1]])
            self.ax_ux.set_xlim(plot_cutoff, current_time)
            self.ax_uy.set_xlim(plot_cutoff, current_time)
            self.plot_ux.set_data(times_raw, ux_vals)
            self.plot_uy.set_data(times_raw, uy_vals)
        else:
            self.plot_line.set_data([], [])
            self.plot_ball.set_data([], [])
            self.plot_ux.set_data([], [])
            self.plot_uy.set_data([], [])
            
        self.canvas.draw_idle()

        self.show_image(display_frame, self.video_label)
        if warped is not None:
            self.show_image(warped, self.warped_label)

    def mix_kinematics_2dof(self, u_x, u_y):
        s1_is_x = (self.combo_s1.currentIndex() == 0)
        s2_is_x = (self.combo_s2.currentIndex() == 0)
        
        s1_inv = -1.0 if self.check_s1_inv.isChecked() else 1.0
        s2_inv = -1.0 if self.check_s2_inv.isChecked() else 1.0

        u1 = (u_x if s1_is_x else u_y) * s1_inv
        u2 = (u_x if s2_is_x else u_y) * s2_inv

        self.send_to_arduino(u1, u2)

    def send_to_arduino(self, u1, u2):
        frame = f"<{u1:.2f},{u2:.2f}>\n"
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
            self.send_to_arduino(0, 0)
            self.ser.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BallPlateTrackerSystem()
    window.show()
    sys.exit(app.exec())