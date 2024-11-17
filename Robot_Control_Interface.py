import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, \
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


class RobotController(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.connectToSim()
        self.speed = 1

    def initUI(self):
        self.setWindowTitle('4-Wheel Robot Controller')

        # Основной горизонтальный слой
        main_layout = QHBoxLayout()

        # Вертикальный слой для кнопок управления и ползунка скорости
        control_layout = QVBoxLayout()

        # Кнопки управления движением робота
        self.forwardButton = QPushButton('Move Forward', self)
        self.forwardButton.clicked.connect(lambda: self.move_robot(self.speed, 0))
        self.backwardButton = QPushButton('Move Backward', self)
        self.backwardButton.clicked.connect(lambda: self.move_robot(-self.speed, 0))
        self.leftButton = QPushButton('Turn Left', self)
        self.leftButton.clicked.connect(lambda: self.move_robot(0, self.speed * 0.5))
        self.rightButton = QPushButton('Turn Right', self)
        self.rightButton.clicked.connect(lambda: self.move_robot(0, -self.speed * 0.5))

        control_layout.addWidget(self.forwardButton)
        control_layout.addWidget(self.backwardButton)
        control_layout.addWidget(self.leftButton)
        control_layout.addWidget(self.rightButton)

        # Кнопки для включения и выключения моторов
        self.motorOnButton = QPushButton('Turn Motors On', self)
        self.motorOnButton.clicked.connect(self.turn_motors_on)
        self.motorOffButton = QPushButton('Turn Motors Off', self)
        self.motorOffButton.clicked.connect(self.turn_motors_off)
        control_layout.addWidget(self.motorOnButton)
        control_layout.addWidget(self.motorOffButton)

        # Кнопки для запуска и остановки симуляции
        self.startSimButton = QPushButton('Start Simulation', self)
        self.startSimButton.clicked.connect(self.start_simulation)
        control_layout.addWidget(self.startSimButton)
        self.stopSimButton = QPushButton('Stop Simulation', self)
        self.stopSimButton.clicked.connect(self.stop_simulation)
        control_layout.addWidget(self.stopSimButton)

        # Слайдер для регулировки скорости
        self.speedLabel = QLabel('Speed: 1', self)
        control_layout.addWidget(self.speedLabel)
        self.speedSlider = QSlider(Qt.Horizontal, self)
        self.speedSlider.setRange(1, 10)
        self.speedSlider.setValue(1)
        self.speedSlider.valueChanged.connect(self.update_speed)
        control_layout.addWidget(self.speedSlider)

        # Добавляем вертикальный слой с кнопками на основной горизонтальный слой
        main_layout.addLayout(control_layout)

        # Вертикальный слой для изображений с этапами обработки
        image_layout = QVBoxLayout()

        # Добавляем виджеты для RGB, пороговой фильтрации, детекции и Depth камеры
        self.rgb_view = QGraphicsView()
        self.rgb_scene = QGraphicsScene()
        self.rgb_view.setScene(self.rgb_scene)
        self.rgb_image_item = QGraphicsPixmapItem()
        self.rgb_scene.addItem(self.rgb_image_item)
        image_layout.addWidget(QLabel("Original RGB Image"))
        image_layout.addWidget(self.rgb_view)

        self.filtered_view = QGraphicsView()
        self.filtered_scene = QGraphicsScene()
        self.filtered_view.setScene(self.filtered_scene)
        self.filtered_image_item = QGraphicsPixmapItem()
        self.filtered_scene.addItem(self.filtered_image_item)
        image_layout.addWidget(QLabel("Threshold Filtered Image"))
        image_layout.addWidget(self.filtered_view)

        self.detection_view = QGraphicsView()
        self.detection_scene = QGraphicsScene()
        self.detection_view.setScene(self.detection_scene)
        self.detection_image_item = QGraphicsPixmapItem()
        self.detection_scene.addItem(self.detection_image_item)
        image_layout.addWidget(QLabel("Fire Detection Image"))
        image_layout.addWidget(self.detection_view)

        self.depth_view = QGraphicsView()
        self.depth_scene = QGraphicsScene()
        self.depth_view.setScene(self.depth_scene)
        self.depth_image_item = QGraphicsPixmapItem()
        self.depth_scene.addItem(self.depth_image_item)
        image_layout.addWidget(QLabel("Depth Camera Image"))
        image_layout.addWidget(self.depth_view)

        # Добавляем вертикальный слой с изображениями на основной горизонтальный слой
        main_layout.addLayout(image_layout)

        # Устанавливаем основной слой в окне
        self.setLayout(main_layout)
        self.setGeometry(100, 100, 800, 600)

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
            rgb_img = rgb_img[::-1]  # Отражение изображения по оси Y

            # Применение пороговой фильтрации для выделения огня
            hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

            # Настройка диапазонов пороговой фильтрации для огня (оранжево-жёлтый цвет)
            lower_bound = np.array([15, 100, 100])  # Нижняя граница HSV
            upper_bound = np.array([35, 255, 255])  # Верхняя граница HSV
            threshold_img = cv2.inRange(hsv_img, lower_bound, upper_bound)

            # Оценка координат очага возгорания
            contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detection_img = rgb_img.copy()
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                cv2.circle(detection_img, (int(x), int(y)), int(radius), (255, 0, 0), 2)
                print(
                    f"Detected fire at coordinates: X={x}, Y={y}, Z=?")  # Z можно рассчитать с использованием данных глубины

            # Отображение оригинального изображения
            rgb_qimg = QImage(rgb_img.data.tobytes(), resX, resY, QImage.Format_RGB888)
            self.rgb_image_item.setPixmap(QPixmap.fromImage(rgb_qimg))

            # Отображение пороговой фильтрации
            threshold_qimg = QImage(threshold_img.data.tobytes(), resX, resY, QImage.Format_Grayscale8)
            self.filtered_image_item.setPixmap(QPixmap.fromImage(threshold_qimg))

            # Отображение детекции очага
            detection_qimg = QImage(detection_img.data.tobytes(), resX, resY, QImage.Format_RGB888)
            self.detection_image_item.setPixmap(QPixmap.fromImage(detection_qimg))

            # Получаем изображение с Depth-камеры и отображаем его
            depth_img, depthX, depthY = self.sim.getVisionSensorCharImage(self.depth_camera_handle)
            depth_img = np.frombuffer(depth_img, dtype=np.uint8).reshape(depthY, depthX, 3)
            depth_img = cv2.flip(depth_img, 0)
            depth_qimg = QImage(depth_img.data, depthX, depthY, QImage.Format_RGB888)
            self.depth_image_item.setPixmap(QPixmap.fromImage(depth_qimg))

            # Выполняем шаг симуляции
            self.sim.step()

        except Exception as e:
            print(f"Error updating camera images: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    controller = RobotController()
    controller.show()
    sys.exit(app.exec_())
