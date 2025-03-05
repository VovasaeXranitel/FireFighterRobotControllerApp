import sys
import math
import numpy as np
import cv2
import traceback
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from zmqRemoteApi import RemoteAPIClient


################################################################
# Глобальный перехватчик необработанных исключений
################################################################
def my_excepthook(exctype, value, tb):
    print("Необработанное исключение!")
    traceback.print_exception(exctype, value, tb)
    input("Нажмите Enter, чтобы выйти.")
    sys.exit(1)


sys.excepthook = my_excepthook


################################################################


class RobotController(QWidget):
    extinguishingFinished = pyqtSignal()

    def __init__(self):
        super().__init__()
        try:
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

            # Параметры Shi-Tomasi и Лукаса-Канаде
            self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

            self.prev_gray_frame = None
            self.p0 = None

            # Храним «хорошие» точки оптического потока, чтобы analyze_environment
            # мог объединять облако точек и оптический поток
            self.flow_good_old = None
            self.flow_good_new = None

            # Таймеры
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_camera_images)
            self.timer.start(50)

            self.timer_flow = QTimer()
            self.timer_flow.timeout.connect(self.update_optical_flow)
            self.timer_flow.start(50)

            # Для движения змейкой
            self.snake_step = 0
            self.snake_direction = 1
            self.snake_timer = None

        except Exception as e:
            print("Ошибка в __init__:")
            traceback.print_exc()

    def initUI(self):
        try:
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
            control_layout.addWidget(self.motorOnButton)

            self.motorOffButton = QPushButton('Выключить моторы', self)
            self.motorOffButton.clicked.connect(self.turn_motors_off)
            control_layout.addWidget(self.motorOffButton)

            # Кнопка для запуска змейки
            self.snakeButton = QPushButton('Двигаться по змейке', self)
            self.snakeButton.clicked.connect(self.move_snake_pattern)
            control_layout.addWidget(self.snakeButton)

            # Добавляем вертикальный слой с кнопками на основной горизонтальный слой
            main_layout.addLayout(control_layout)

            # Вертикальный слой для изображений с этапами обработки
            image_layout = QVBoxLayout()

            # RGB изображение
            self.rgb_view = QGraphicsView()
            self.rgb_scene = QGraphicsScene()
            self.rgb_view.setScene(self.rgb_scene)
            self.rgb_image_item = QGraphicsPixmapItem()
            self.rgb_scene.addItem(self.rgb_image_item)
            image_layout.addWidget(QLabel("Оригинальное RGB изображение"))
            image_layout.addWidget(self.rgb_view)

            # Пороговая фильтрация
            self.filtered_view = QGraphicsView()
            self.filtered_scene = QGraphicsScene()
            self.filtered_view.setScene(self.filtered_scene)
            self.filtered_image_item = QGraphicsPixmapItem()
            self.filtered_scene.addItem(self.filtered_image_item)
            image_layout.addWidget(QLabel("Пороговая фильтрация"))
            image_layout.addWidget(self.filtered_view)

            # Детекция огня
            self.detection_view = QGraphicsView()
            self.detection_scene = QGraphicsScene()
            self.detection_view.setScene(self.detection_scene)
            self.detection_image_item = QGraphicsPixmapItem()
            self.detection_scene.addItem(self.detection_image_item)
            image_layout.addWidget(QLabel("Детекция огня"))
            image_layout.addWidget(self.detection_view)

            # Изображение глубины
            self.depth_view = QGraphicsView()
            self.depth_scene = QGraphicsScene()
            self.depth_view.setScene(self.depth_scene)
            self.depth_image_item = QGraphicsPixmapItem()
            self.depth_scene.addItem(self.depth_image_item)
            image_layout.addWidget(QLabel("Изображение глубины"))
            image_layout.addWidget(self.depth_view)

            # Добавляем вертикальный слой с изображениями на основной горизонтальный слой
            main_layout.addLayout(image_layout)

            # Виджет для изображения с наложенным облаком точек
            overlay_layout = QVBoxLayout()
            self.overlay_view = QGraphicsView()
            self.overlay_scene = QGraphicsScene()
            self.overlay_view.setScene(self.overlay_scene)
            self.overlay_image_item = QGraphicsPixmapItem()
            self.overlay_scene.addItem(self.overlay_image_item)
            overlay_layout.addWidget(QLabel("Облако точек"))
            overlay_layout.addWidget(self.overlay_view)

            main_layout.addLayout(overlay_layout)

            # Виджет для отображения оптического потока
            optical_flow_layout = QVBoxLayout()
            self.optical_flow_view = QGraphicsView()
            self.optical_flow_scene = QGraphicsScene()
            self.optical_flow_view.setScene(self.optical_flow_scene)
            self.optical_flow_image_item = QGraphicsPixmapItem()
            self.optical_flow_scene.addItem(self.optical_flow_image_item)
            optical_flow_layout.addWidget(QLabel("Оптический поток"))
            optical_flow_layout.addWidget(self.optical_flow_view)

            main_layout.addLayout(optical_flow_layout)

            # Устанавливаем основной слой в окне
            self.setLayout(main_layout)
            self.setGeometry(100, 100, 1600, 800)

        except Exception as e:
            print("Ошибка в initUI:")
            traceback.print_exc()

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
            try:
                self.rgb_camera_handle = self.sim.getObject('/kinect/rgb')
            except Exception as e:
                print("Ошибка: Камера RGB не найдена. Проверьте путь '/kinect/rgb'.")
                raise e

            try:
                self.depth_camera_handle = self.sim.getObject('/kinect/depth')
            except Exception as e:
                print("Ошибка: Камера глубины не найдена. Проверьте путь '/kinect/depth'.")
                raise e

            # Параметры камеры
            resolution = self.sim.getVisionSensorResolution(self.rgb_camera_handle)
            resX, resY = resolution
            self.fx, self.fy = resX / 2.0, resY / 2.0
            self.cx, self.cy = resX / 2.0, resY / 2.0

            # Получаем дескрипторы для манипулятора
            self.manipulator_joint_x = self.sim.getObject('/Joint_Osi_X_Polivalka')
            self.manipulator_joint_y = self.sim.getObject('/Joint_Osi_Y_Polivalka')

        except Exception as e:
            print("Ошибка подключения к CoppeliaSim:")
            traceback.print_exc()

    #################################################################
    #   ОБНОВЛЁННЫЙ UPDATE_OPTICAL_FLOW: Храним good_old, good_new
    #################################################################
    def update_optical_flow(self):
        try:
            rgb_img_data, resX, resY = self.sim.getVisionSensorCharImage(self.rgb_camera_handle)
            frame = np.frombuffer(rgb_img_data, dtype=np.uint8).reshape(resY, resX, 3)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            if self.prev_gray_frame is None:
                self.prev_gray_frame = gray_frame
                self.p0 = cv2.goodFeaturesToTrack(self.prev_gray_frame, mask=None, **self.feature_params)
                return

            if self.p0 is None or len(self.p0) == 0:
                self.p0 = cv2.goodFeaturesToTrack(gray_frame, mask=None, **self.feature_params)
                self.prev_gray_frame = gray_frame
                return

            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray_frame, gray_frame, self.p0, None, **self.lk_params)

            if p1 is not None:
                good_new = p1[st == 1]
                good_old = self.p0[st == 1]

                # Сохраняем для объединённого анализа
                self.flow_good_old = good_old
                self.flow_good_new = good_new

                # Визуализация оптического потока
                flow_frame = frame.copy()
                for new, old in zip(good_new, good_old):
                    x_new, y_new = new.ravel()
                    x_old, y_old = old.ravel()
                    cv2.arrowedLine(flow_frame, (int(x_old), int(y_old)), (int(x_new), int(y_new)),
                                    (0, 255, 0), 2, tipLength=0.5)

                flow_frame_flipped = cv2.flip(flow_frame, 0)
                qimg = QImage(flow_frame_flipped.data, flow_frame_flipped.shape[1], flow_frame_flipped.shape[0],
                              flow_frame_flipped.strides[0], QImage.Format_RGB888)
                self.optical_flow_image_item.setPixmap(QPixmap.fromImage(qimg))

                self.prev_gray_frame = gray_frame.copy()
                self.p0 = good_new.reshape(-1, 1, 2)
            else:
                self.p0 = cv2.goodFeaturesToTrack(gray_frame, mask=None, **self.feature_params)
                self.prev_gray_frame = gray_frame

        except Exception as e:
            print("Ошибка при обновлении оптического потока:")
            traceback.print_exc()

    def move_robot(self, linear_velocity, angular_velocity):
        try:
            max_speed = 5.0
            linear_velocity = np.clip(linear_velocity, -max_speed, max_speed)
            angular_velocity = np.clip(angular_velocity, -max_speed, max_speed)

            left_velocity = linear_velocity - angular_velocity
            right_velocity = linear_velocity + angular_velocity

            self.sim.setJointTargetVelocity(self.left_front_motor, left_velocity)
            self.sim.setJointTargetVelocity(self.left_rear_motor, left_velocity)
            self.sim.setJointTargetVelocity(self.right_front_motor, right_velocity)
            self.sim.setJointTargetVelocity(self.right_rear_motor, right_velocity)

        except Exception as e:
            print("Ошибка при движении робота:")
            traceback.print_exc()

    def turn_motors_on(self):
        try:
            self.move_robot(self.speed, 0)
        except Exception as e:
            print("Ошибка при включении моторов:")
            traceback.print_exc()

    def turn_motors_off(self):
        try:
            self.move_robot(0, 0)
        except Exception as e:
            print("Ошибка при выключении моторов:")
            traceback.print_exc()

    def start_simulation(self):
        try:
            self.sim.startSimulation()
            print("Симуляция запущена.")
        except Exception as e:
            print("Ошибка при запуске симуляции:")
            traceback.print_exc()

    def stop_simulation(self):
        try:
            self.sim.stopSimulation()
            print("Симуляция остановлена.")
        except Exception as e:
            print("Ошибка при остановке симуляции:")
            traceback.print_exc()

    def update_speed(self, value):
        try:
            self.speed = value
            self.speedLabel.setText(f'Скорость: {value}')
        except Exception as e:
            print("Ошибка при обновлении скорости:")
            traceback.print_exc()

    ####################################################################
    # ЛОГИКА ДВИЖЕНИЯ ЗМЕЙКОЙ (с учётом препятствий, обнаруженных
    # как по облаку точек, так и по оптическому потоку)
    ####################################################################
    def move_snake_pattern(self):
        """
        Запускает движение робота по «змейке».
        """
        try:
            self.snake_step = 0
            self.snake_direction = 1
            if self.snake_timer is None:
                self.snake_timer = QTimer()
                self.snake_timer.timeout.connect(self.execute_snake_step)

            self.snake_timer.start(3000)
            print("Движение по змейке запущено.")
        except Exception as e:
            print("Ошибка при запуске движения змейкой:")
            traceback.print_exc()

    def stop_snake_timer(self):
        """Останавливаем таймер змейки, если активен."""
        if self.snake_timer and self.snake_timer.isActive():
            self.snake_timer.stop()
            print("Таймер змейки остановлен (из-за препятствия).")

    def resume_snake_timer(self):
        """Возобновляем таймер змейки."""
        if self.snake_timer:
            self.snake_timer.start(3000)
            print("Таймер змейки возобновлён.")

    def execute_snake_step(self):
        """
        Выполняет один шаг змейки:
         - Проверяет, есть ли препятствие (analyze_environment)
         - Если есть, объезжает
         - Если нет, двигается по схеме (прямо -> поворот -> прямо -> ...)
        """
        try:
            obstacle_detected, turn_direction = self.analyze_environment()

            if obstacle_detected:
                print("Обнаружено препятствие по объединённым данным. Выполняем объезд.")
                # Останавливаем змейку
                self.stop_snake_timer()
                # Пример «откатиться назад, затем повернуть»
                self.move_robot(-self.speed, 0)
                QTimer.singleShot(1500, lambda: self.finish_backing_up(turn_direction))
            else:
                # Стандартная «змейка»
                if self.snake_step % 2 == 0:
                    print("Робот двигается прямо (змейка).")
                    self.move_robot(self.speed, 0)
                else:
                    dir_text = 'вправо' if self.snake_direction == 1 else 'влево'
                    print(f"Робот поворачивает {dir_text} (змейка).")
                    self.move_robot(0, self.snake_direction * self.speed * 0.5)
                    self.snake_direction *= -1

                self.snake_step += 1
        except Exception as e:
            print("Ошибка при выполнении шага змейки:")
            traceback.print_exc()

    def finish_backing_up(self, turn_direction):
        """Вызывается после того, как робот 1.5 сек сдавал назад."""
        try:
            print("Откат завершён, поворачиваем для объезда...")
            self.move_robot(0, turn_direction * self.speed * 0.7)
            QTimer.singleShot(1500, self.finish_turning)
        except Exception as e:
            print("Ошибка в finish_backing_up:")
            traceback.print_exc()

    def finish_turning(self):
        """Заканчиваем поворот, возобновляем змейку."""
        try:
            print("Поворот завершён, продолжаем движение змейкой.")
            self.move_robot(self.speed, 0)
            self.resume_snake_timer()
        except Exception as e:
            print("Ошибка в finish_turning:")
            traceback.print_exc()

    ####################################################################
    # ОБЪЕДИНЁННЫЙ АНАЛИЗ: облако точек + оптический поток
    ####################################################################
    def analyze_environment(self):
        """
        Слияние облака точек (depth) и оптического потока (flow)
        для решения об объезде препятствия.
        Возвращает: (obstacle_detected, turn_direction).
        """

        try:
            # 1) Проверяем облако точек
            depth_buffer = self.sim.getVisionSensorDepth(self.depth_camera_handle)[0]
            depthX, depthY = self.sim.getVisionSensorResolution(self.depth_camera_handle)
            depth_img = np.frombuffer(depth_buffer, dtype=np.float32).reshape((depthY, depthX))

            points_3d, _ = self.generate_point_cloud(
                depth_img,
                np.zeros_like(depth_img),  # неиспользуем threshold для огня
                self.fx, self.fy, self.cx, self.cy
            )

            # analyze_point_cloud => (bool_obstacle, direction)
            cloud_obstacle, cloud_dir = self.analyze_point_cloud(points_3d)

            # 2) Проверяем оптический поток
            flow_obstacle = False
            if (self.flow_good_old is not None) and (self.flow_good_new is not None):
                flow_obstacle = self.detect_obstacle_by_flow(
                    self.flow_good_old, self.flow_good_new, depthX, depthY
                )

            # 3) Сливаем
            # Логика: если хотя бы один говорит «объект», считаем, что препятствие
            obstacle_detected = cloud_obstacle or flow_obstacle

            if obstacle_detected:
                if cloud_obstacle:
                    turn_direction = cloud_dir
                else:
                    # Если «только поток» видит препятствие, поворачиваем вправо (или влево)
                    turn_direction = 1
                return True, turn_direction

            return False, 0

        except Exception as e:
            print("Ошибка в analyze_environment:")
            traceback.print_exc()
            return False, 0

    def detect_obstacle_by_flow(self, old_pts, new_pts, width, height):
        """
        Пример упрощённой детекции препятствия по оптическому потоку:
        - Смотрим центральную область кадра
        - Считаем среднюю длину векторов
        - Если > 2.0 пикселей/кадр, считаем, что есть препятствие
        """
        try:
            roi_left = width * 1 / 3
            roi_right = width * 2 / 3
            roi_top = height * 1 / 3
            roi_bottom = height * 2 / 3

            vectors_in_roi = []
            for (ox, oy), (nx, ny) in zip(old_pts, new_pts):
                if roi_left < ox < roi_right and roi_top < oy < roi_bottom:
                    dx = nx - ox
                    dy = ny - oy
                    vectors_in_roi.append((dx, dy))

            if len(vectors_in_roi) == 0:
                return False

            magnitudes = [math.hypot(dx, dy) for (dx, dy) in vectors_in_roi]
            avg_mag = np.mean(magnitudes)

            # Порог, подбирайте под свою сцену:
            if avg_mag > 2.0:
                return True
            return False
        except Exception as e:
            print("Ошибка в detect_obstacle_by_flow:")
            traceback.print_exc()
            return False

    ####################################################################
    # ЛОГИКА РАБОТЫ С КАМЕРОЙ, ОГНЁМ, ГЛУБИНОЙ
    ####################################################################
    def update_camera_images(self):
        try:
            # Проверяем, запущена ли симуляция
            sim_state = self.sim.getSimulationState()
            if sim_state != self.sim.simulation_advancing_running:
                return

            if not hasattr(self, 'rgb_camera_handle') or not hasattr(self, 'depth_camera_handle'):
                print("Ошибка: Дескрипторы камер не установлены.")
                return

            # Получаем изображение с RGB-камеры
            rgb_img_data, resX, resY = self.sim.getVisionSensorCharImage(self.rgb_camera_handle)
            rgb_img = np.frombuffer(rgb_img_data, dtype=np.uint8).reshape(resY, resX, 3)
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

            # Пороговая фильтрация для выделения огня
            hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

            # Границы для красного
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([179, 255, 255])

            mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
            threshold_img = cv2.bitwise_or(mask1, mask2)

            # Оценка координат очага возгорания
            contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detection_img = bgr_img.copy()
            fire_detected = False
            fire_coordinates = None
            fire_distance = None

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    fire_coordinates = (cx, cy)
                    fire_detected = True

                    cv2.circle(detection_img, fire_coordinates, 5, (0, 0, 255), -1)

                    depth_val = self.get_depth_at_point(cx, cy)
                    if depth_val is not None and depth_val > 0:
                        fire_distance = depth_val * 5.0
                        print(f"fire_distance = {fire_distance} meters")
                    else:
                        print("Не удалось получить глубину в точке огня.")

            # Получаем изображение с глубинной камеры
            resolution = self.sim.getVisionSensorResolution(self.depth_camera_handle)
            depthX, depthY = resolution
            depth_buffer = self.sim.getVisionSensorDepth(self.depth_camera_handle)[0]
            depth_img = np.frombuffer(depth_buffer, dtype=np.float32).reshape((depthY, depthX))

            # Нормализация глубинного изображения
            depth_display_img = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
            depth_display_img = depth_display_img.astype(np.uint8)
            depth_display_img = cv2.applyColorMap(depth_display_img, cv2.COLORMAP_JET)

            # Генерация облака точек (для отображения)
            points_3d, colors = self.generate_point_cloud(
                depth_img, threshold_img,
                self.fx, self.fy, self.cx, self.cy
            )
            # analyze_point_cloud + detect_obstacle_by_flow => объединены в analyze_environment

            # Логика «огонь» + «препятствие»
            if fire_detected:
                self.aim_manipulator(fire_coordinates, depth_img, self.fx, self.fy, self.cx, self.cy)
                self.manipulator_default_position = False

                if self.fire_extinguishing:
                    self.move_robot(0, 0)  # тушим
                elif fire_distance is not None and fire_distance < 0.5:
                    self.fire_extinguishing = True
                    self.fire_detected_once = True
                    self.extinguishing_timer.start(30000)
                    print("Начинаем тушение огня.")
                    self.move_robot(0, 0)
                else:
                    self.move_robot(self.speed, 0)
            else:
                if not self.manipulator_default_position:
                    self.reset_manipulator()
                    self.manipulator_default_position = True
                # Ниже — код обычного движения, но у нас уже есть логика змейки,
                # поэтому можно оставить move_robot(...) = self.speed, 0
                # или ничего не делать (змейка сама рулит)
                # self.move_robot(self.speed, 0)

            # Отображение (перевороты)
            rgb_display_img = cv2.flip(rgb_img, 0)
            threshold_display_img = cv2.flip(threshold_img, 0)
            detection_display_img = cv2.flip(detection_img, 0)
            depth_display_img_flipped = cv2.flip(depth_display_img, 0)

            rgb_qimg = QImage(rgb_display_img.data, rgb_display_img.shape[1], rgb_display_img.shape[0],
                              rgb_display_img.strides[0], QImage.Format_RGB888)
            self.rgb_image_item.setPixmap(QPixmap.fromImage(rgb_qimg))

            threshold_qimg = QImage(threshold_display_img.data, threshold_display_img.shape[1],
                                    threshold_display_img.shape[0], threshold_display_img.strides[0],
                                    QImage.Format_Grayscale8)
            self.filtered_image_item.setPixmap(QPixmap.fromImage(threshold_qimg))

            detection_qimg = QImage(detection_display_img.data, detection_display_img.shape[1],
                                    detection_display_img.shape[0], detection_display_img.strides[0],
                                    QImage.Format_RGB888)
            self.detection_image_item.setPixmap(QPixmap.fromImage(detection_qimg))

            depth_qimg = QImage(depth_display_img_flipped.data, depth_display_img_flipped.shape[1],
                                depth_display_img_flipped.shape[0], depth_display_img_flipped.strides[0],
                                QImage.Format_RGB888)
            self.depth_image_item.setPixmap(QPixmap.fromImage(depth_qimg))

            # Отображение облака точек
            points_2d = self.project_to_2d(points_3d, self.fx, self.fy, self.cx, self.cy)
            overlay_img = self.display_point_cloud(rgb_img, points_2d, colors)
            overlay_display_img = cv2.flip(overlay_img, 0)
            overlay_qimg = QImage(overlay_display_img.data, overlay_display_img.shape[1], overlay_display_img.shape[0],
                                  overlay_display_img.strides[0], QImage.Format_RGB888)
            self.overlay_image_item.setPixmap(QPixmap.fromImage(overlay_qimg))

        except Exception as e:
            print("Ошибка при обновлении изображений с камер:")
            traceback.print_exc()

    def on_extinguishing_finished(self):
        try:
            self.fire_extinguishing = False
            self.move_robot(self.speed, 0)
            print("Тушение огня завершено, продолжаем движение.")
        except Exception as e:
            print("Ошибка в on_extinguishing_finished:")
            traceback.print_exc()

    def get_depth_at_point(self, x, y):
        try:
            depth_buffer = self.sim.getVisionSensorDepth(self.depth_camera_handle)[0]
            depthX, depthY = self.sim.getVisionSensorResolution(self.depth_camera_handle)
            depth_img = np.frombuffer(depth_buffer, dtype=np.float32).reshape((depthY, depthX))
            if 0 <= x < depthX and 0 <= y < depthY:
                return depth_img[y, x]
            else:
                print(f"Координаты ({x}, {y}) выходят за границы ({depthX}, {depthY})")
                return None
        except Exception as e:
            print("Ошибка в get_depth_at_point:")
            traceback.print_exc()
            return None

    def generate_point_cloud(self, depth_img, threshold_img, fx, fy, cx, cy):
        try:
            h, w = depth_img.shape
            points_3d = []
            colors = []
            for y in range(0, h, 5):
                for x in range(0, w, 5):
                    depth = depth_img[y, x]
                    if depth > 0:
                        z = depth * 5.0
                        px = (x - cx) * z / fx
                        py = (y - cy) * z / fy
                        points_3d.append([px, py, z])

                        # цвет - красный если threshold_img[y,x]!=0, иначе синий
                        if threshold_img[y, x] != 0:
                            color = (0, 0, 255)
                        else:
                            color = (255, 0, 0)
                        colors.append(color)
            return points_3d, colors
        except Exception as e:
            print("Ошибка в generate_point_cloud:")
            traceback.print_exc()
            return [], []

    def analyze_point_cloud(self, points_3d):
        try:
            obstacle_threshold = 0.3
            obstacle_detected = False
            left_points = 0
            right_points = 0

            for (x, y, z) in points_3d:
                if 0.1 < z < obstacle_threshold and abs(x) < 0.7:
                    obstacle_detected = True
                    if x < 0:
                        left_points += 1
                    else:
                        right_points += 1

            if obstacle_detected:
                if left_points > right_points:
                    return True, 1  # вправо
                else:
                    return True, -1  # влево
            return False, 0
        except Exception as e:
            print("Ошибка в analyze_point_cloud:")
            traceback.print_exc()
            return False, 0

    def detect_obstacle_by_flow(self, old_pts, new_pts, width, height):
        try:
            roi_left = width * 1 / 3
            roi_right = width * 2 / 3
            roi_top = height * 1 / 3
            roi_bottom = height * 2 / 3

            vectors_in_roi = []
            for (ox, oy), (nx, ny) in zip(old_pts, new_pts):
                if roi_left < ox < roi_right and roi_top < oy < roi_bottom:
                    dx = nx - ox
                    dy = ny - oy
                    vectors_in_roi.append((dx, dy))

            if len(vectors_in_roi) == 0:
                return False

            magnitudes = [math.hypot(dx, dy) for (dx, dy) in vectors_in_roi]
            avg_mag = np.mean(magnitudes)
            return (avg_mag > 2.0)
        except Exception as e:
            print("Ошибка в detect_obstacle_by_flow:")
            traceback.print_exc()
            return False

    def project_to_2d(self, points_3d, fx, fy, cx, cy):
        try:
            points_2d = []
            for point in points_3d:
                if len(point) == 3:
                    x, y, z = point
                    if z > 0:
                        u = fx * x / z + cx
                        v = fy * y / z + cy
                        points_2d.append((u, v))
            return points_2d
        except Exception as e:
            print("Ошибка в project_to_2d:")
            traceback.print_exc()
            return []

    def display_point_cloud(self, rgb_img, points_2d, colors):
        try:
            overlay_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            h, w = overlay_img.shape[:2]
            for (u, v), color in zip(points_2d, colors):
                if 0 <= u < w and 0 <= v < h:
                    cv2.circle(overlay_img, (int(u), int(v)), 2, color, -1)
            overlay_img_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
            return overlay_img_rgb
        except Exception as e:
            print("Ошибка в display_point_cloud:")
            traceback.print_exc()
            return rgb_img

    def aim_manipulator(self, fire_coordinates, depth_img, fx, fy, cx, cy):
        try:
            x_img, y_img = fire_coordinates
            depth = depth_img[y_img, x_img]
            if depth > 0:
                x_img_inverted = depth_img.shape[1] - x_img
                z = depth * 5.0
                x = (x_img_inverted - cx) * z / fx
                y = (y_img - cy) * z / fy
                self.point_manipulator_at(x, y, z)
            else:
                print("Не удалось получить глубину в точке наведения манипулятора.")
        except Exception as e:
            print("Ошибка в aim_manipulator:")
            traceback.print_exc()

    def point_manipulator_at(self, x, y, z):
        try:
            angle_x = float(np.arctan2(y, z))
            angle_y = float(np.arctan2(x, z))

            max_angle = math.pi / 2
            min_angle = -math.pi / 2
            angle_x = np.clip(angle_x, min_angle, max_angle)
            angle_y = np.clip(angle_y, min_angle, max_angle)

            print(f"Углы манипулятора: angle_x={angle_x:.2f}, angle_y={angle_y:.2f}")

            self.sim.setJointTargetPosition(self.manipulator_joint_x, angle_y)
            self.sim.setJointTargetPosition(self.manipulator_joint_y, angle_x)
        except Exception as e:
            print("Ошибка в point_manipulator_at:")
            traceback.print_exc()

    def reset_manipulator(self):
        try:
            self.sim.setJointTargetPosition(self.manipulator_joint_x, 0.0)
            self.sim.setJointTargetPosition(self.manipulator_joint_y, 0.0)
            print("Возвращаем манипулятор в исходное положение.")
        except Exception as e:
            print("Ошибка в reset_manipulator:")
            traceback.print_exc()

    def closeEvent(self, event):
        try:
            event.accept()
        except Exception as e:
            print("Ошибка при закрытии приложения:")
            traceback.print_exc()


if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        controller = RobotController()
        controller.show()
        sys.exit(app.exec_())
    except Exception as e:
        print("Необработанное исключение при запуске приложения:")
        traceback.print_exc()
        input("Нажмите Enter для выхода.")
        sys.exit(1)
