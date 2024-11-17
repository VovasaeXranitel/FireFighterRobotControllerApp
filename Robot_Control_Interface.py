import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QSlider, QLabel, QGraphicsView, \
    QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import cv2


class RobotController(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.connectToSim()
        self.speed = 1

    def initUI(self):
        self.setWindowTitle('4-Wheel Robot Controller')
        main_layout = QVBoxLayout()

        # Кнопки управления движением робота
        movement_layout = QVBoxLayout()
        self.forwardButton = QPushButton('Move Forward', self)
        self.forwardButton.clicked.connect(lambda: self.move_robot(self.speed, 0))
        self.backwardButton = QPushButton('Move Backward', self)
        self.backwardButton.clicked.connect(lambda: self.move_robot(-self.speed, 0))
        self.leftButton = QPushButton('Turn Left', self)
        self.leftButton.clicked.connect(lambda: self.move_robot(0, self.speed * 0.5))
        self.rightButton = QPushButton('Turn Right', self)
        self.rightButton.clicked.connect(lambda: self.move_robot(0, -self.speed * 0.5))

        movement_layout.addWidget(self.forwardButton)
        movement_layout.addWidget(self.backwardButton)
        movement_layout.addWidget(self.leftButton)
        movement_layout.addWidget(self.rightButton)

        # Кнопки для включения и выключения моторов
        self.motorOnButton = QPushButton('Turn Motors On', self)
        self.motorOnButton.clicked.connect(self.turn_motors_on)
        self.motorOffButton = QPushButton('Turn Motors Off', self)
        self.motorOffButton.clicked.connect(self.turn_motors_off)
        movement_layout.addWidget(self.motorOnButton)
        movement_layout.addWidget(self.motorOffButton)

        # Кнопки для запуска и остановки симуляции
        self.startSimButton = QPushButton('Start Simulation', self)
        self.startSimButton.clicked.connect(self.start_simulation)
        movement_layout.addWidget(self.startSimButton)
        self.stopSimButton = QPushButton('Stop Simulation', self)
        self.stopSimButton.clicked.connect(self.stop_simulation)
        movement_layout.addWidget(self.stopSimButton)

        # Слайдер для регулировки скорости
        self.speedLabel = QLabel('Speed: 1', self)
        movement_layout.addWidget(self.speedLabel)
        self.speedSlider = QSlider(Qt.Horizontal, self)
        self.speedSlider.setRange(1, 10)
        self.speedSlider.setValue(1)
        self.speedSlider.valueChanged.connect(self.update_speed)
        movement_layout.addWidget(self.speedSlider)

        # Графические виджеты для отображения изображений с камер
        self.rgb_view = QGraphicsView()
        self.rgb_scene = QGraphicsScene()
        self.rgb_view.setScene(self.rgb_scene)
        self.rgb_image_item = QGraphicsPixmapItem()
        self.rgb_scene.addItem(self.rgb_image_item)

        self.depth_view = QGraphicsView()
        self.depth_scene = QGraphicsScene()
        self.depth_view.setScene(self.depth_scene)
        self.depth_image_item = QGraphicsPixmapItem()
        self.depth_scene.addItem(self.depth_image_item)

        # Добавляем виджеты для отображения изображений
        main_layout.addLayout(movement_layout)
        main_layout.addWidget(QLabel("RGB Camera"))
        main_layout.addWidget(self.rgb_view)
        main_layout.addWidget(QLabel("Depth Camera"))
        main_layout.addWidget(self.depth_view)

        # Устанавливаем основной слой
        self.setLayout(main_layout)
        self.setGeometry(100, 100, 300, 600)

        # Таймер для обновления изображений с камер
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_images)
        self.timer.start(100)  # Обновление каждые 100 мс

    def connectToSim(self):
        try:
            self.client = RemoteAPIClient()
            self.sim = self.client.getObject('sim')

            # Получаем дескрипторы для моторов
            self.left_front_motor = self.sim.getObject('/Joint_Koleso_PL')
            self.left_rear_motor = self.sim.getObject('/Joint_Koleso_ZL')
            self.right_front_motor = self.sim.getObject('/Joint_Koleso_PP')
            self.right_rear_motor = self.sim.getObject('/Joint_Koleso_ZP')

            # Подключение к RGB и Depth камерам Kinect
            self.rgb_camera_handle = self.sim.getObjectHandle('/kinect/rgb')
            self.depth_camera_handle = self.sim.getObjectHandle('/kinect/depth')

            if self.rgb_camera_handle == -1:
                print("Error: RGB camera handle not found.")
            else:
                print("RGB camera connected.")

            if self.depth_camera_handle == -1:
                print("Error: Depth camera handle not found.")
            else:
                print("Depth camera connected.")

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

        except Exception as e:
            print(f"Error moving robot: {e}")

    def turn_motors_on(self):
        self.move_robot(self.speed, 0)

    def turn_motors_off(self):
        self.move_robot(0, 0)

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
            if not hasattr(self, 'rgb_camera_handle') or not hasattr(self, 'depth_camera_handle'):
                print("Error: Camera handles are not set.")
                return

            # Получаем изображение с RGB-камеры
            rgb_img, resX, resY = self.sim.getVisionSensorCharImage(self.rgb_camera_handle)
            rgb_img = np.frombuffer(rgb_img, dtype=np.uint8).reshape(resY, resX, 3)
            rgb_img = cv2.flip(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB), 0)
            rgb_qimg = QImage(rgb_img.data, resX, resY, QImage.Format_RGB888)
            self.rgb_image_item.setPixmap(QPixmap.fromImage(rgb_qimg))

            # Получаем изображение с Depth-камеры
            depth_img, depthX, depthY = self.sim.getVisionSensorCharImage(self.depth_camera_handle)
            depth_img = np.frombuffer(depth_img, dtype=np.uint8).reshape(depthY, depthX, 3)
            depth_img = cv2.flip(depth_img, 0)
            depth_qimg = QImage(depth_img.data, depthX, depthY, QImage.Format_RGB888)
            self.depth_image_item.setPixmap(QPixmap.fromImage(depth_qimg))

            self.sim.step()

        except Exception as e:
            print(f"Error updating camera images: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    controller = RobotController()
    controller.show()
    sys.exit(app.exec_())
