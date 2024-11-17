import sys
import time
import math
import numpy as np
import cv2
import traceback
import random
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, \
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from zmqRemoteApi import RemoteAPIClient

class RobotController(QWidget):
    extinguishingFinished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.initUI()
        self.connectToSim()
        self.speed = 1

        # Флаги и таймеры для тушения огня
        self.fire_extinguishing = False
        self.fire_detected_once = False

        # Таймер для тушения огня
        self.extinguishing_timer = QTimer()
        self.extinguishing_timer.setSingleShot(True)
        self.extinguishing_timer.timeout.connect(self.on_extinguishing_finished)

        # Флаг для отслеживания положения манипулятора
        self.manipulator_default_position = True

    def initUI(self):
        self.setWindowTitle('Управление 4-колёсным роботом')

        # Основной горизонтальный слой
        main_layout = QHBoxLayout()

        # Вертикальный слой для кнопок управления и ползунка скорости
        control_layout = QVBoxLayout()

        # Кнопки для запуска и остановки симуляции
        self.startSimButton = QPushButton('Запустить симуляцию', self)
        self.startSimButton.clicked.connect(self.start_simulation)
        control_layout.addWidget(self.startSimButton)
        self.stopSimButton = QPushButton('Остановить симуляцию', self)
        self.stopSimButton.clicked.connect(self.stop_simulation)
        control_layout.addWidget(self.stopSimButton)

        # Слайдер для регулировки скорости
        self.speedLabel = QLabel('Скорость: 1', self)
        control_layout.addWidget(self.speedLabel)
        self.speedSlider = QSlider(Qt.Horizontal, self)
        self.speedSlider.setRange(1, 10)
        self.speedSlider.setValue(1)
        self.speedSlider.valueChanged.connect(self.update_speed)
        control_layout.addWidget(self.speedSlider)

        # Кнопки для включения и выключения моторов
        self.motorOnButton = QPushButton('Включить моторы', self)
        self.motorOnButton.clicked.connect(self.turn_motors_on)
        self.motorOffButton = QPushButton('Выключить моторы', self)
        self.motorOffButton.clicked.connect(self.turn_motors_off)
        control_layout.addWidget(self.motorOnButton)
        control_layout.addWidget(self.motorOffButton)

        # Добавляем вертикальный слой с кнопками на основной горизонтальный слой
        main_layout.addLayout(control_layout)

        # Вертикальный слой для изображений с этапами обработки
        image_layout = QVBoxLayout()

        # Виджеты для RGB, пороговой фильтрации, детекции и глубинной камеры
        self.rgb_view = QGraphicsView()
        self.rgb_scene = QGraphicsScene()
        self.rgb_view.setScene(self.rgb_scene)
        self.rgb_image_item = QGraphicsPixmapItem()
        self.rgb_scene.addItem(self.rgb_image_item)
        image_layout.addWidget(QLabel("Оригинальное RGB изображение"))
        image_layout.addWidget(self.rgb_view)

        self.filtered_view = QGraphicsView()
        self.filtered_scene = QGraphicsScene()
        self.filtered_view.setScene(self.filtered_scene)
        self.filtered_image_item = QGraphicsPixmapItem()
        self.filtered_scene.addItem(self.filtered_image_item)
        image_layout.addWidget(QLabel("Пороговая фильтрация"))
        image_layout.addWidget(self.filtered_view)

        self.detection_view = QGraphicsView()
        self.detection_scene = QGraphicsScene()
        self.detection_view.setScene(self.detection_scene)
        self.detection_image_item = QGraphicsPixmapItem()
        self.detection_scene.addItem(self.detection_image_item)
        image_layout.addWidget(QLabel("Детекция огня"))
        image_layout.addWidget(self.detection_view)

        self.depth_view = QGraphicsView()
        self.depth_scene = QGraphicsScene()
        self.depth_view.setScene(self.depth_scene)
        self.depth_image_item = QGraphicsPixmapItem()
        self.depth_scene.addItem(self.depth_image_item)
        image_layout.addWidget(QLabel("Изображение глубины"))
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
        overlay_layout.addWidget(QLabel("Облако точек"))
        overlay_layout.addWidget(self.overlay_view)

        # Добавляем вертикальный слой с наложенным изображением на основной горизонтальный слой
        main_layout.addLayout(overlay_layout)

        # Устанавливаем основной слой в окне
        self.setLayout(main_layout)
        self.setGeometry(100, 100, 1600, 800)

        # Таймер для обновления изображений с камер
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_images)
        self.timer.start(50)  # Обновление каждые 50 мс

    def connectToSim(self):
        try:
            self.client = RemoteAPIClient()
            self.sim = self.client.getObject('sim')

            # Получаем дескрипторы для моторов
            self.left_front_motor = self.sim.getObject('/Joint_Koleso_PL')
            self.left_rear_motor = self.sim.getObject('/Joint_Koleso_ZL')
            self.right_front_motor = self.sim.getObject('/Joint_Koleso_PP')
            self.right_rear_motor = self.sim.getObject('/Joint_Koleso_ZP')

            # Подключение к RGB и глубинной камерам Kinect
            self.rgb_camera_handle = self.sim.getObject('/kinect/rgb')
            self.depth_camera_handle = self.sim.getObject('/kinect/depth')

            # Получаем дескрипторы для манипулятора
            self.manipulator_joint_x = self.sim.getObject('/Joint_Osi_X_Polivalka')
            self.manipulator_joint_y = self.sim.getObject('/Joint_Osi_Y_Polivalka')

            # Убираем пошаговый режим симуляции
            # self.sim.setStepping(True)  # Закомментировано

        except Exception as e:
            print(f"Ошибка подключения: {e}")
            traceback.print_exc()

    def move_robot(self, linear_velocity, angular_velocity):
        # Ограничиваем скорость
        max_speed = 5.0
        linear_velocity = np.clip(linear_velocity, -max_speed, max_speed)
        angular_velocity = np.clip(angular_velocity, -max_speed, max_speed)

        try:
            left_velocity = linear_velocity - angular_velocity
            right_velocity = linear_velocity + angular_velocity

            self.sim.setJointTargetVelocity(self.left_front_motor, left_velocity)
            self.sim.setJointTargetVelocity(self.left_rear_motor, left_velocity)
            self.sim.setJointTargetVelocity(self.right_front_motor, right_velocity)
            self.sim.setJointTargetVelocity(self.right_rear_motor, right_velocity)

        except Exception as e:
            print(f"Ошибка при движении робота: {e}")
            traceback.print_exc()

    def turn_motors_on(self):
        self.move_robot(self.speed, 0)

    def turn_motors_off(self):
        self.move_robot(0, 0)

    def start_simulation(self):
        try:
            self.sim.startSimulation()
            print("Симуляция запущена.")
        except Exception as e:
            print(f"Ошибка при запуске симуляции: {e}")
            traceback.print_exc()

    def stop_simulation(self):
        try:
            self.sim.stopSimulation()
            print("Симуляция остановлена.")
        except Exception as e:
            print(f"Ошибка при остановке симуляции: {e}")
            traceback.print_exc()

    def update_speed(self, value):
        self.speed = value
        self.speedLabel.setText(f'Скорость: {value}')

    def update_camera_images(self):
        try:
            # Проверяем, запущена ли симуляция
            sim_state = self.sim.getSimulationState()
            if sim_state != self.sim.simulation_advancing_running:
                return  # Если симуляция не запущена, выходим из метода

            if not hasattr(self, 'rgb_camera_handle') or not hasattr(self, 'depth_camera_handle'):
                print("Ошибка: Дескрипторы камер не установлены.")
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
            fire_detected = False
            fire_coordinates = None
            fire_distance = None  # Расстояние до огня
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)

                # Вычисляем моменты контура
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    fire_coordinates = (cx, cy)
                    fire_detected = True  # Устанавливаем в True только если удалось вычислить центр масс

                    # Рисуем центр очага на изображении
                    cv2.circle(detection_img, fire_coordinates, 5, (0, 0, 255), -1)  # Красная точка

                    # Вычисляем расстояние до огня
                    depth = self.get_depth_at_point(fire_coordinates[0], fire_coordinates[1])
                    if depth is not None and depth > 0:
                        fire_distance = depth * 5.0  # Масштабируем глубину
                        print(f"fire_distance = {fire_distance} meters")
                    else:
                        print("Не удалось получить глубину в точке огня.")
                else:
                    fire_detected = False  # Если не удалось вычислить центр масс, считаем, что огонь не обнаружен

            else:
                fire_detected = False

            # Конвертируем изображения обратно в RGB для отображения
            detection_img_rgb = cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB)

            # Получаем разрешение глубинного сенсора
            resolution = self.sim.getVisionSensorResolution(self.depth_camera_handle)
            depthX, depthY = resolution

            # Получаем изображение с глубинной камеры
            depth_buffer = self.sim.getVisionSensorDepth(self.depth_camera_handle)[0]
            depth_img = np.frombuffer(depth_buffer, dtype=np.float32).reshape((depthY, depthX))
            depth_img = cv2.flip(depth_img, 0)

            # Нормализация глубинного изображения для отображения
            depth_display_img = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
            depth_display_img = depth_display_img.astype(np.uint8)
            depth_display_img = cv2.applyColorMap(depth_display_img, cv2.COLORMAP_JET)

            # Параметры камеры
            h, w = depth_img.shape
            fx, fy = 500, 500  # Настройте эти значения в соответствии с параметрами вашей камеры
            cx_cam, cy_cam = w / 2, h / 2

            # Генерация облака точек и анализ препятствий
            points_3d, colors = self.generate_point_cloud(depth_img, threshold_img, fx, fy, cx_cam, cy_cam)
            obstacle_detected = self.analyze_point_cloud(points_3d)

            # Автономное управление
            if fire_detected:
                # Огонь обнаружен и центр масс вычислен
                self.aim_manipulator(fire_coordinates, depth_img, fx, fy, cx_cam, cy_cam)
                self.manipulator_default_position = False

                if self.fire_extinguishing:
                    # Робот тушит огонь
                    self.move_robot(0, 0)
                elif fire_distance is not None and fire_distance < 0.5:
                    # Достаточно близко к огню, начинаем тушение
                    self.fire_extinguishing = True
                    self.fire_detected_once = True
                    self.extinguishing_timer.start(30000)  # 30 секунд
                    print("Начинаем тушение огня.")
                    self.move_robot(0, 0)
                else:
                    # Продолжаем движение к огню
                    self.move_robot(self.speed, 0)
            else:
                # Огонь не обнаружен или центр масс не вычислен
                if not self.manipulator_default_position:
                    # Возвращаем манипулятор в исходное положение
                    self.reset_manipulator()
                    self.manipulator_default_position = True

                if obstacle_detected:
                    # Обход препятствия: робот поворачивает на месте
                    turn_direction = random.choice([-1, 1])  # Случайно выбираем направление поворота
                    self.move_robot(0, turn_direction * self.speed)
                else:
                    # Движемся вперёд
                    self.move_robot(self.speed, 0)

            # Проекция 3D точек на 2D плоскость
            points_2d = []
            for point in points_3d:
                x, y, z = point
                u = int(fx * x / z + cx_cam)
                v = int(fy * y / z + cy_cam)
                points_2d.append([u, v])

            # Отображаем облако точек
            overlay_img = self.display_point_cloud(rgb_img, points_2d, colors)

            # Отображение изображений этапов
            rgb_qimg = QImage(rgb_img.data, rgb_img.shape[1], rgb_img.shape[0], rgb_img.strides[0], QImage.Format_RGB888)
            self.rgb_image_item.setPixmap(QPixmap.fromImage(rgb_qimg))

            threshold_qimg = QImage(threshold_img.data, threshold_img.shape[1], threshold_img.shape[0], threshold_img.strides[0], QImage.Format_Grayscale8)
            self.filtered_image_item.setPixmap(QPixmap.fromImage(threshold_qimg))

            detection_qimg = QImage(detection_img_rgb.data, detection_img_rgb.shape[1], detection_img_rgb.shape[0], detection_img_rgb.strides[0], QImage.Format_RGB888)
            self.detection_image_item.setPixmap(QPixmap.fromImage(detection_qimg))

            depth_qimg = QImage(depth_display_img.data, depth_display_img.shape[1], depth_display_img.shape[0], depth_display_img.strides[0], QImage.Format_RGB888)
            self.depth_image_item.setPixmap(QPixmap.fromImage(depth_qimg))

            overlay_qimg = QImage(overlay_img.data, overlay_img.shape[1], overlay_img.shape[0], overlay_img.strides[0], QImage.Format_RGB888)
            self.overlay_image_item.setPixmap(QPixmap.fromImage(overlay_qimg))

            # Симуляция идёт асинхронно, нет необходимости вызывать self.sim.step()

        except Exception as e:
            print(f"Ошибка при обновлении изображений с камер: {e}")
            traceback.print_exc()

    def on_extinguishing_finished(self):
        self.fire_extinguishing = False
        self.move_robot(self.speed, 0)
        print("Тушение огня завершено, продолжаем движение.")

    def get_depth_at_point(self, x, y):
        # Получаем глубину в точке (x, y)
        try:
            depth_buffer = self.sim.getVisionSensorDepth(self.depth_camera_handle)[0]
            depthX, depthY = self.sim.getVisionSensorResolution(self.depth_camera_handle)
            depth_img = np.frombuffer(depth_buffer, dtype=np.float32).reshape((depthY, depthX))
            depth_img = cv2.flip(depth_img, 0)

            # Проверка границ
            if 0 <= x < depthX and 0 <= y < depthY:
                depth = depth_img[y, x]
                return depth
            else:
                print(f"Координаты ({x}, {y}) выходят за границы изображения глубины ({depthX}, {depthY})")
                return None
        except Exception as e:
            print(f"Ошибка при получении глубины в точке: {e}")
            return None

    def generate_point_cloud(self, depth_img, threshold_img, fx, fy, cx, cy):
        h, w = depth_img.shape
        points_3d = []
        colors = []

        for y in range(0, h, 5):  # Шаг 5 пикселей для более подробного облака
            for x in range(0, w, 5):
                depth = depth_img[y, x]
                if depth > 0:
                    z = depth * 5.0  # Масштабируем глубину
                    px = (x - cx) * z / fx
                    py = (y - cy) * z / fy
                    points_3d.append([px, py, z])

                    # Проверяем, является ли пиксель огнём
                    if threshold_img[y, x] != 0:
                        color = (0, 0, 255)  # Красный
                    else:
                        color = (255, 0, 0)  # Синий
                    colors.append(color)

        return points_3d, colors

    def analyze_point_cloud(self, points_3d):
        obstacle_threshold = 0.3  # Пороговое расстояние для обнаружения препятствия (30 см)
        obstacle_detected = False

        for point in points_3d:
            x, y, z = point
            # Проверяем точки перед роботом в пределах obstacle_threshold
            if 0.1 < z < obstacle_threshold and abs(x) < 0.7:
                obstacle_detected = True
                break

        return obstacle_detected

    def display_point_cloud(self, rgb_img, points_2d, colors):
        overlay_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        h, w = overlay_img.shape[:2]

        # Наложение точек на изображение
        for (u, v), color in zip(points_2d, colors):
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(overlay_img, (u, v), 2, color, -1)

        # Конвертируем обратно в RGB для отображения
        overlay_img_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
        return overlay_img_rgb

    def aim_manipulator(self, fire_coordinates, depth_img, fx, fy, cx, cy):
        x_img, y_img = fire_coordinates
        depth = depth_img[y_img, x_img]
        if depth > 0:
            # Вычисляем реальные координаты
            z = depth * 5.0  # Масштабируем глубину
            x = (x_img - cx) * z / fx
            y = (y_img - cy) * z / fy

            # Командуем манипулятору навестись на (x, y, z)
            self.point_manipulator_at(x, y, z)
        else:
            print("Не удалось получить глубину в точке наведения манипулятора.")

    def point_manipulator_at(self, x, y, z):
        try:
            # Вычисляем углы для суставов манипулятора
            angle_x = float(np.arctan2(y, z))
            angle_y = float(np.arctan2(x, z))

            # Ограничиваем углы в диапазоне [-pi/2, pi/2]
            max_angle = math.pi / 2
            min_angle = -math.pi / 2
            angle_x = np.clip(angle_x, min_angle, max_angle)
            angle_y = np.clip(angle_y, min_angle, max_angle)

            print(f"Устанавливаемые углы манипулятора: angle_x={angle_x}, angle_y={angle_y}")

            # Меняем местами управление суставами
            self.sim.setJointTargetPosition(self.manipulator_joint_x, angle_y)
            self.sim.setJointTargetPosition(self.manipulator_joint_y, angle_x)

        except Exception as e:
            print(f"Ошибка при наведении манипулятора: {e}")
            traceback.print_exc()

    def reset_manipulator(self):
        try:
            # Возвращаем суставы манипулятора в изначальные положения (например, 0 радиан)
            self.sim.setJointTargetPosition(self.manipulator_joint_x, 0.0)
            self.sim.setJointTargetPosition(self.manipulator_joint_y, 0.0)
            print("Возвращаем манипулятор в исходное положение.")
        except Exception as e:
            print(f"Ошибка при возврате манипулятора: {e}")
            traceback.print_exc()

    def closeEvent(self, event):
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    controller = RobotController()
    controller.show()
    sys.exit(app.exec_())
