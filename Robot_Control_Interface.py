import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QSlider, QLabel
from PyQt5.QtCore import Qt, QTimer
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class RobotController(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.connectToSim()
        self.speed = 1

    def initUI(self):
        self.setWindowTitle('4-Wheel Robot Controller')
        layout = QVBoxLayout()

        # Управление движением робота
        self.forwardButton = QPushButton('Move Forward', self)
        self.forwardButton.clicked.connect(lambda: self.move_robot(self.speed, 0))
        self.backwardButton = QPushButton('Move Backward', self)
        self.backwardButton.clicked.connect(lambda: self.move_robot(-self.speed, 0))
        self.leftButton = QPushButton('Turn Left', self)
        self.leftButton.clicked.connect(lambda: self.move_robot(0, self.speed * 0.5))
        self.rightButton = QPushButton('Turn Right', self)
        self.rightButton.clicked.connect(lambda: self.move_robot(0, -self.speed * 0.5))

        layout.addWidget(self.forwardButton)
        layout.addWidget(self.backwardButton)
        layout.addWidget(self.leftButton)
        layout.addWidget(self.rightButton)

        # Управление симуляцией
        self.startSimButton = QPushButton('Start Simulation', self)
        self.startSimButton.clicked.connect(self.start_simulation)
        layout.addWidget(self.startSimButton)

        self.stopSimButton = QPushButton('Stop Simulation', self)
        self.stopSimButton.clicked.connect(self.stop_simulation)
        layout.addWidget(self.stopSimButton)

        # Слайдер для регулировки скорости
        self.speedLabel = QLabel('Speed: 1', self)
        layout.addWidget(self.speedLabel)

        self.speedSlider = QSlider(Qt.Horizontal, self)
        self.speedSlider.setRange(1, 10)
        self.speedSlider.setValue(1)
        self.speedSlider.valueChanged.connect(self.update_speed)
        layout.addWidget(self.speedSlider)

        self.setLayout(layout)
        self.setGeometry(100, 100, 300, 400)

        # Таймер для обновления изображений с камер
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_images)
        self.timer.start(100)  # Обновление каждые 100 мс

    def connectToSim(self):
        try:
            self.client = RemoteAPIClient()
            self.sim = self.client.getObject('sim')

            # Получаем дескрипторы для всех четырех моторов
            self.left_front_motor = self.sim.getObject('/leftFrontMotor')
            self.left_rear_motor = self.sim.getObject('/leftRearMotor')
            self.right_front_motor = self.sim.getObject('/rightFrontMotor')
            self.right_rear_motor = self.sim.getObject('/rightRearMotor')

            # Подключение к RGB и Depth камерам
            self.rgb_camera_handle = self.sim.getObject('/Kamera')
            self.depth_camera_handle = self.sim.getObject('/KameraGlubina')

            if self.rgb_camera_handle == -1 or self.depth_camera_handle == -1:
                print("Error: One or both camera handles not found.")
            else:
                print("Connected to both cameras in CoppeliaSim.")

            # Получаем разрешение камеры
            _, self.rgb_resolution = self.sim.getVisionSensorImg(self.rgb_camera_handle)
            _, self.depth_resolution = self.sim.getVisionSensorImg(self.depth_camera_handle)

            print(f"RGB camera resolution: {self.rgb_resolution}")
            print(f"Depth camera resolution: {self.depth_resolution}")

            # Установка пошагового режима симуляции
            self.sim.setStepping(True)

        except Exception as e:
            print(f"Connection error: {e}")

    def move_robot(self, linear_velocity, angular_velocity):
        try:
            left_velocity = linear_velocity - angular_velocity
            right_velocity = linear_velocity + angular_velocity

            self.sim.setJointTargetVelocity(self.left_front_motor, left_velocity)
            self.sim.setJointTargetVelocity(self.left_rear_motor, left_velocity)
            self.sim.setJointTargetVelocity(self.right_front_motor, right_velocity)
            self.sim.setJointTargetVelocity(self.right_rear_motor, right_velocity)
            self.sim.setJointTargetVelocity(self.right_rear_motor, right_velocity)

        except Exception as e:
            print(f"Error moving robot: {e}")

    def start_simulation(self):
        try:
            self.sim.startSimulation()
            print("Simulation started.")
        except Exception as e:
            print(f"Error starting simulation: {e}")

    def stop_simulation(self):
        try:
            self.sim.stopSimulation()
            print("Simulation stopped.")
        except Exception as e:
            print(f"Error stopping simulation: {e}")

    def update_speed(self, value):
        self.speed = value
        self.speedLabel.setText(f'Speed: {value}')

    def update_camera_images(self):
        try:
            # Проверка на существование камеры перед обращением к ней
            if not hasattr(self, 'rgb_camera_handle') or not hasattr(self, 'depth_camera_handle'):
                print("Error: Camera handles are not set. Please check camera connection.")
                return

            # Получаем изображение с RGB-камеры
            rgb_img, resX, resY = self.sim.getVisionSensorCharImage(self.rgb_camera_handle)
            rgb_img = np.frombuffer(rgb_img, dtype=np.uint8).reshape(self.rgb_resolution[1], self.rgb_resolution[0], 3)
            rgb_img = cv2.flip(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB), 0)

            # Получаем изображение с Depth-камеры
            depth_img, depthX, depthY = self.sim.getVisionSensorCharImage(self.depth_camera_handle)
            depth_img = np.frombuffer(depth_img, dtype=np.uint8).reshape(self.depth_resolution[1], self.depth_resolution[0])
            depth_img = cv2.flip(depth_img, 0)

            # Отображаем изображения с камер
            cv2.imshow('RGB Camera', rgb_img)
            cv2.imshow('Depth Camera', depth_img)
            cv2.waitKey(1)

            # Переход к следующему шагу симуляции
            self.sim.step()

        except Exception as e:
            print(f"Error updating camera images: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    controller = RobotController()
    controller.show()
    sys.exit(app.exec_())
