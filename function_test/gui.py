import sys

import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from gui_screen import Ui_MainWindow
import cv2


class MainWindow:
    def __init__(self):
        self.main_win = QMainWindow()

        self.uic = Ui_MainWindow()

        # self.main
        self.uic.setupUi(self.main_win)
        self.uic.exit_button.clicked.connect(self.main_win.close)
        self.uic.switch_button.clicked.connect(self.start_capture_video)

        # multi-thread
        self.thread = {}

    def start_capture_video(self):
        self.thread[1] = capture_video(index=1)
        self.thread[1].signal.connect(self.show_cam)
        self.thread[1].start()
        print("ireowo")


    def show_cam(self, image):
        """Updates the image_label with a new opencv image"""
        print("viaedk")
        print(type(image))
        self.uic.video_screen.setPixmap(QPixmap.fromImage(image))


    def show(self):
        # command to run
        self.main_win.show()


class capture_video(QThread):
    signal = pyqtSignal(QImage)
    def __init__(self, index):
        self.index = index
        print("start threading", self.index)
        self.__thread_active = True
        self.fps = 0
        self.__thread_pause = False
        super(capture_video, self).__init__()

    def run(self):
        cap = cv2.VideoCapture("../video_test/resize-market-square.mp4")  # 'D:/8.Record video/My Video.mp4'
        if cap.isOpened():
            while self.__thread_active:
                if not self.__thread_pause:
                    ret, cv_img = cap.read()
                    if ret:
                        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        height, width, channel = cv_img.shape
                        bytes_per_line = 3 * width
                        q_image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
                        self.signal.emit(q_image)

        # When everything done, release the video capture object.
        cap.release()
        # Tells the thread's event loop to exit with return code 0 (success).
        self.quit()

    def stop(self) -> None:
        self.__thread_active = False

    def pause(self) -> None:
        self.__thread_pause = True

    def unpause(self) -> None:
        self.__thread_pause = False

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
