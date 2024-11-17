import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from zmqRemoteApi import RemoteAPIClient


class RobotController(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.connectToSim()

    def initUI(self):
        self.setWindowTitle('4-Wheel Robot Controller')

        layout = QVBoxLayout()

        self.forwardButton = QPushButton('Move Forward', self)
        self.forwardButton.clicked.connect(lambda: self.move_robot(1, 0))

        self.backwardButton = QPushButton('Move Backward', self)
        self.backwardButton.clicked.connect(lambda: self.move_robot(-1, 0))

        self.leftButton = QPushButton('Turn Left', self)
        self.leftButton.clicked.connect(lambda: self.move_robot(0, 0.5))

        self.rightButton = QPushButton('Turn Right', self)
        self.rightButton.clicked.connect(lambda: self.move_robot(0, -0.5))

        layout.addWidget(self.forwardButton)
        layout.addWidget(self.backwardButton)
        layout.addWidget(self.leftButton)
        layout.addWidget(self.rightButton)

        self.setLayout(layout)
        self.setGeometry(100, 100, 200, 200)

    def connectToSim(self):
        try:
            self.client = RemoteAPIClient()
            self.sim = self.client.getObject('sim')

            # Получаем дескрипторы для всех четырех моторов
            self.left_front_motor = self.sim.getObject('/Joint_Koleso_PL')
            self.left_rear_motor = self.sim.getObject('/Joint_Koleso_ZL')
            self.right_front_motor = self.sim.getObject('/Joint_Koleso_PP')
            self.right_rear_motor = self.sim.getObject('/Joint_Koleso_ZP')

            if -1 in [self.left_front_motor, self.left_rear_motor, self.right_front_motor, self.right_rear_motor]:
                raise Exception("Error: One or more motor handles not found in the scene.")
            print("Connected to CoppeliaSim successfully.")
        except Exception as e:
            print(f"Connection error: {e}")

    def move_robot(self, linear_velocity, angular_velocity):
        try:
            # Расчет скорости для левых и правых колес
            left_velocity = linear_velocity - angular_velocity
            right_velocity = linear_velocity + angular_velocity

            # Установка скоростей для левых колес
            self.sim.setJointTargetVelocity(self.left_front_motor, left_velocity)
            self.sim.setJointTargetVelocity(self.left_rear_motor, left_velocity)

            # Установка скоростей для правых колес
            self.sim.setJointTargetVelocity(self.right_front_motor, right_velocity)
            self.sim.setJointTargetVelocity(self.right_rear_motor, right_velocity)

        except Exception as e:
            print(f"Error moving robot: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    controller = RobotController()
    controller.show()
    sys.exit(app.exec_())
