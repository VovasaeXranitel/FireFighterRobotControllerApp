import sys
import numpy as np
import cv2
import pandas as pd
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

        # Виджеты для RGB, пороговой фильтрации, детекции и Depth камеры
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

        # Добавляем виджет для изображения с наложенным облаком точек
        overlay_layout = QVBoxLayout()
        self.overlay_view = QGraphicsView()
        self.overlay_scene = QGraphicsScene()
        self.overlay_view.setScene(self.overlay_scene)
        self.overlay_image_item = QGraphicsPixmapItem()
        self.overlay_scene.addItem(self.overlay_image_item)
        overlay_layout.addWidget(QLabel("Overlay Point Cloud Image"))
        overlay_layout.addWidget(self.overlay_view)

        # Добавляем вертикальный слой с наложенным изображением на основной горизонтальный слой
        main_layout.addLayout(overlay_layout)

        # Устанавливаем основной слой в окне
        self.setLayout(main_layout)
        self.setGeometry(100, 100, 1600, 800)  # Увеличим размер окна для размещения новых виджетов

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
            # Проверяем, запущена ли симуляция
            sim_state = self.sim.getSimulationState()
            if sim_state != self.sim.simulation_advancing_running:
                return  # Если симуляция не запущена, выходим из метода

            if not hasattr(self, 'rgb_camera_handle') or not hasattr(self, 'depth_camera_handle'):
                print("Error: Camera handles are not set.")
                return

            # Получаем изображение с RGB-камеры
            rgb_img_data, resX, resY = self.sim.getVisionSensorCharImage(self.rgb_camera_handle)
            rgb_img = np.frombuffer(rgb_img_data, dtype=np.uint8).reshape(resY, resX, 3)
            rgb_img = cv2.flip(rgb_img, 0)  # Отражение изображения по оси Y

            # Конвертируем в BGR для OpenCV
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

            # Применение пороговой фильтрации для выделения огня
            hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

            # Настройка диапазонов пороговой фильтрации для огня (оранжево-жёлтый цвет)
            lower_bound = np.array([15, 100, 100])  # Нижняя граница HSV
            upper_bound = np.array([35, 255, 255])  # Верхняя граница HSV
            threshold_img = cv2.inRange(hsv_img, lower_bound, upper_bound)

            # Оценка координат очага возгорания
            contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detection_img = bgr_img.copy()
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                cv2.circle(detection_img, (int(x), int(y)), int(radius), (0, 0, 255), 2)  # Красный круг

            # Конвертируем изображения обратно в RGB для отображения
            detection_img_rgb = cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB)

            # Получаем изображение с Depth-камеры
            depth_img_data, depthX, depthY = self.sim.getVisionSensorCharImage(self.depth_camera_handle)
            depth_img = np.frombuffer(depth_img_data, dtype=np.uint8).reshape(depthY, depthX, 3)
            depth_img = cv2.flip(depth_img, 0)

            # Отображение изображений этапов
            rgb_qimg = QImage(rgb_img.data, resX, resY, QImage.Format_RGB888)
            self.rgb_image_item.setPixmap(QPixmap.fromImage(rgb_qimg))

            threshold_qimg = QImage(threshold_img.data, resX, resY, QImage.Format_Grayscale8)
            self.filtered_image_item.setPixmap(QPixmap.fromImage(threshold_qimg))

            detection_qimg = QImage(detection_img_rgb.data, resX, resY, QImage.Format_RGB888)
            self.detection_image_item.setPixmap(QPixmap.fromImage(detection_qimg))

            depth_qimg = QImage(depth_img.data, depthX, depthY, QImage.Format_RGB888)
            self.depth_image_item.setPixmap(QPixmap.fromImage(depth_qimg))

            # Обновляем облако точек и накладываем на RGB изображение
            overlay_img = self.display_point_cloud(rgb_img, depth_img, threshold_img)

            # Отображаем изображение с наложенным облаком точек
            overlay_qimg = QImage(overlay_img.data, resX, resY, QImage.Format_RGB888)
            self.overlay_image_item.setPixmap(QPixmap.fromImage(overlay_qimg))

            # Выполняем шаг симуляции
            self.sim.step()

        except Exception as e:
            print(f"Error updating camera images: {e}")

    def display_point_cloud(self, rgb_img, depth_img, threshold_img):
        # Конвертируем изображения в BGR для OpenCV
        overlay_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        h, w = depth_img.shape[:2]
        fx, fy = 500, 500  # Примерные значения фокусного расстояния
        cx, cy = w / 2, h / 2
        points_3d = []
        colors = []

        for y in range(0, h, 20):  # Шаг 20 пикселей
            for x in range(0, w, 20):  # Шаг 20 пикселей
                depth = depth_img[y, x, 0]
                if depth > 0:
                    z = depth / 255.0 * 5.0
                    px = (x - cx) * z / fx
                    py = (y - cy) * z / fy
                    points_3d.append([px, py, z])

                    # Проверяем, является ли пиксель огнем
                    if threshold_img[y, x] != 0:
                        # Если пиксель соответствует огню, красим точку в красный
                        color = (0, 0, 255)  # Красный цвет в BGR
                    else:
                        # Иначе используем исходный цвет пикселя
                        b, g, r = overlay_img[y, x]
                        color = (255, 0, 0)
                    colors.append(color)

        # Проекция 3D точек обратно на 2D плоскость изображения
        points_2d = []
        for point in points_3d:
            x, y, z = point
            u = int(fx * x / z + cx)
            v = int(fy * y / z + cy)
            points_2d.append([u, v])

        # Наложение точек на изображение
        for (u, v), color in zip(points_2d, colors):
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(overlay_img, (u, v), 5, color, -1)  # Точки с радиусом 5

        # Конвертируем обратно в RGB для отображения
        overlay_img_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

        return overlay_img_rgb

    def closeEvent(self, event):
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    controller = RobotController()
    controller.show()
    sys.exit(app.exec_())
