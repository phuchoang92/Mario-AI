import cv2
import sys
import pyqtgraph as pg

from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot

from Agent import DQNAgent
from ProjectAI.Policy import *
from ProjectAI.Wrappers import make_env

save_directory = "mario_dqn_model"
strategy = EGreedyStategy()
discount_factor = 0.0
algo_name = "DQN"
saved_model = ""
dueling = False


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    changePlot = pyqtSignal(int)
    changePlot1 = pyqtSignal(int)

    def __init__(self, parent=None):
        super(Thread, self).__init__(parent=parent)
        self.n_episodes = 5000
        self.model_name = algo_name
        self.env = make_env('SuperMarioBros-1-1-v0')
        state_dim = self.env.observation_space.shape
        action_dim = self.env.action_space.n
        self.agent = DQNAgent(state_dim, action_dim, strategy, save_directory, dueling)

    def image_change(self):
        img = self.env.render(mode='rgb_array')
        rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        qImg = QImage(rgb, rgb.shape[1], rgb.shape[0], QImage.Format_Grayscale8)
        p = qImg.scaled(640, 480, Qt.KeepAspectRatio)
        self.changePixmap.emit(p)

    def run(self):

        try:
            if saved_model != "":
                self.agent.load_checkpoint(save_directory + "/" + saved_model)
        except:
            print("Maybe you input wrong model or it doesn't exist")

        self.agent.gamma = discount_factor
        self.agent.update_target()
        for episode in range(self.n_episodes):

            state, done = self.env.reset(), False
            self.agent.episode_reward.append(0.0)
            self.agent.episode_timestep.append(0.0)

            while not done:
                # time.sleep(0.03)
                state, reward, done = self.agent.act(state, self.env)
                self.agent.experience_replay(self.model_name)
                if np.sum(self.agent.episode_timestep) % self.agent.sync_period == 0:
                    self.agent.update_target()
                self.changePlot1.emit(reward)
                self.image_change()
            self.changePlot.emit(self.agent.episode_reward[-1])


class App(QMainWindow):
    def __init__(self, parent=None):
        super(App, self).__init__(parent)

        self.label = QLabel(self)
        frame = QFrame()
        self.run_button = QPushButton("Run")
        self.wait_button = QPushButton("Stop")
        self.stop_button = QPushButton("Terminate")

        self.stop_button.clicked.connect(self.stop_thread)
        self.run_button.clicked.connect(self.runMario)

        self.graphWidget = pg.PlotWidget()
        self.graphWidget2 = pg.PlotWidget()

        self.algo_button = QButtonGroup()
        self.policy_button = QButtonGroup()

        self.plot1 = QWidget()
        self.plot2 = QWidget()
        self.buttonBox = QGroupBox()
        self.bottom_box = QGroupBox()
        self.gamma_choice = QGroupBox()
        self.policy_layout = QGroupBox()
        self.algorithm_choice = QGroupBox()

        plot1_layout = QHBoxLayout()
        plot2_layout = QHBoxLayout()
        self.plot_layout = QHBoxLayout()
        self.bottom_layout = QHBoxLayout()
        self.display_layout = QHBoxLayout()
        self.button_layout = QVBoxLayout()

        self.gamma_choice.setMaximumWidth(300)
        self.display_layout.addWidget(self.label, alignment=QtCore.Qt.AlignRight)

        self.plot1.setMaximumWidth(500)
        self.plot2.setMaximumWidth(150)
        self.plot2.setMaximumHeight(180)

        plot1_layout.addWidget(self.graphWidget)
        plot2_layout.addWidget(self.graphWidget2)

        self.plot1.setLayout(plot1_layout)
        self.plot2.setLayout(plot2_layout)

        self.x = [0]
        self.y = [0]
        pen = pg.mkPen(color=(255, 0, 0))
        self.graphWidget.setBackground('w')
        self.data_line = self.graphWidget.plot(self.x, self.y, pen=pen)

        self.m = list(range(100))
        self.n = list(range(100))
        pen1 = pg.mkPen(color=(0, 255, 0))
        self.graphWidget2.setBackground('w')
        self.data_line2 = self.graphWidget2.plot(self.m, self.n, pen=pen1)

        self.plot_layout.setSpacing(0)
        self.plot_layout.addWidget(self.plot1)
        self.plot_layout.addWidget(self.plot2, alignment=QtCore.Qt.AlignBottom)
        self.plot_layout.setContentsMargins(0, 5, 0, 5)

        self.button_layout.addWidget(self.run_button)
        self.button_layout.addWidget(self.stop_button)
        self.button_layout.setContentsMargins(50, 2, 20, 5)

        self.buttonBox.setLayout(self.button_layout)

        self.policy_choice()
        self.gamma_choices()
        self.algorithms_choice()
        self.bottom_layout.addWidget(self.buttonBox)
        self.bottom_layout.addWidget(self.algorithm_choice)
        self.bottom_layout.addWidget(self.policy_layout)
        self.bottom_layout.addWidget(self.gamma_choice)

        self.bottom_box.setLayout(self.bottom_layout)
        self.bottom_box.setContentsMargins(2, 2, 2, 2)

        mainLayout = QGridLayout()
        mainLayout.addLayout(self.display_layout, 0, 5, 1, -1)
        mainLayout.addLayout(self.plot_layout, 0, 0, 1, 5)
        mainLayout.addWidget(self.bottom_box, 2, 0, 1, 10)

        self.left = 350
        self.top = 100
        self.width = 1200
        self.height = 800

        self.set_style_sheet()
        frame.setLayout(mainLayout)
        self.setCentralWidget(frame)
        self.setWindowTitle('Reinforcement Learning')
        self.setGeometry(self.left, self.top, self.width, self.height)

    def set_style_sheet(self):
        self.setStyleSheet("QLabel{font-size: 20px;} QRadioButton {font : 16px Arial;} "
                           "QPushButton{font : 18px Arial;border-radius : 10px;; border : 2px solid black;"
                           "background:qradialgradient(cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,radius: 1.35, stop: 0 #fff, stop: 1 #bbb)}")
        self.label1.setStyleSheet("QLabel{font-size: 15px;}")
        self.label2.setStyleSheet("QLabel{font-size: 15px;}")
        self.buttonBox.setStyleSheet("QGroupBox {border: none;}")

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def policy_choice(self):
        label = QLabel("Strategies")

        layout = QVBoxLayout()

        self.plc_radio1 = QRadioButton('Epsilon Greedy')
        self.plc_radio2 = QRadioButton('Epsilon Linear Greedy')
        self.plc_radio3 = QRadioButton('Epsilon Exponential Greedy')
        self.plc_radio4 = QRadioButton('Softmax')

        self.plc_radio1.toggled.connect(self.gamma_choices)
        self.plc_radio2.toggled.connect(self.gamma_choices)
        self.plc_radio3.toggled.connect(self.gamma_choices)
        self.plc_radio4.toggled.connect(self.gamma_choices)

        self.policy_button.addButton(self.plc_radio1)
        self.policy_button.addButton(self.plc_radio2)
        self.policy_button.addButton(self.plc_radio3)
        self.policy_button.addButton(self.plc_radio4)

        layout.setContentsMargins(50, 50, 50, 50)

        layout.addWidget(label, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.plc_radio1)
        layout.addWidget(self.plc_radio2)
        layout.addWidget(self.plc_radio3)
        layout.addWidget(self.plc_radio4)

        self.policy_layout.setLayout(layout)

    def algorithms_choice(self):
        label = QLabel("Algorithms")

        layout = QVBoxLayout()

        self.alg_radio1 = QRadioButton('Deep Q-Network')
        self.alg_radio2 = QRadioButton('Double DQN')
        self.alg_radio3 = QRadioButton('Dueling DQN')

        self.algo_button.addButton(self.alg_radio1)
        self.algo_button.addButton(self.alg_radio2)
        self.algo_button.addButton(self.alg_radio3)

        self.alg_radio1.toggled.connect(self.algo_check)
        self.alg_radio2.toggled.connect(self.algo_check)
        self.alg_radio3.toggled.connect(self.algo_check)

        layout.setContentsMargins(50, 50, 50, 50)
        layout.addWidget(label, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.alg_radio1)
        layout.addWidget(self.alg_radio2)
        layout.addWidget(self.alg_radio3)

        self.algorithm_choice.setLayout(layout)

    def gamma_choices(self):
        label = QLabel("")
        self.label1 = QLabel("Discount Factors:")
        self.label2 = QLabel("Load Model:")

        self.gamma_input = QLineEdit()
        self.load_model = QLineEdit()

        layout = QFormLayout()

        layout.addRow(label)
        layout.addRow(self.label1, self.gamma_input)
        layout.addRow(self.label2, self.load_model)
        layout.setVerticalSpacing(30)

        self.gamma_choice.setLayout(layout)

    def algo_check(self):
        global algo_name
        global dueling
        radioBtn = self.sender()
        if radioBtn.isChecked():
            if radioBtn.text() == "Deep Q-Network":
                algo_name = "DQN"
            if radioBtn.text() == "Double DQN":
                algo_name = "DDQN"
            if radioBtn.text() == "Dueling DQN":
                print("Dueliing")
                algo_name = "DDQN"
                dueling = True

    def strategy_check(self):
        global strategy
        radioBtn = self.sender()
        if radioBtn.isChecked():
            if radioBtn.text() == "Epsilon Greedy":
                strategy = EGreedyStategy()
            if radioBtn.text() == "Epsilon Linear Greedy":
                strategy = EGreedyLinearStrategy()
            if radioBtn.text() == "Epsilon Exponential Greedy":
                strategy = EGreedyExpStrategy()
            else:
                strategy = SoftMaxStrategy()

    def runMario(self):
        global discount_factor
        global saved_model
        if self.gamma_input.text() != "":
            discount_factor = float(self.gamma_input.text())
        if self.load_model.text() != "":
            saved_model = self.load_model.text()

        self.th = Thread(self)
        self.th.changePixmap.connect(self.setImage)
        self.th.changePlot.connect(self.update_plot_data)
        self.th.changePlot1.connect(self.update_plot_data1)
        self.th.start()

    def stop_thread(self):
        self.th.terminate()

    def update_plot_data(self, new_data):
        self.y.append(new_data)
        self.x.append(self.x[-1] + 1)
        self.data_line.setData(self.x, self.y)

    def update_plot_data1(self, new_data):
        self.m = self.m[1:]
        self.m.append(self.m[-1] + 1)

        self.n = self.n[1:]
        self.n.append(new_data)

        self.data_line2.setData(self.m, self.n)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
