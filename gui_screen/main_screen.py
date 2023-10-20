from PyQt5 import QtCore, QtGui, QtWidgets
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint)

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(741, 413)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(741, 413))
        MainWindow.setMaximumSize(QtCore.QSize(741, 413))
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("\n"
f"background-image: url(./pics/dust.jpg);\n"
"\n"
"")
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setStyleSheet("")
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame = QtWidgets.QFrame(parent=self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame.setObjectName("frame")
        self.tracking_button = QtWidgets.QPushButton(parent=self.frame)
        self.tracking_button.setGeometry(QtCore.QRect(280, 360, 40, 40))
        self.tracking_button.setAutoFillBackground(False)
        self.tracking_button.setStyleSheet("background-color: transparent; border: 0px \n"
"")
        self.tracking_button.setText("")
        icon = QtGui.QIcon()
        print(ROOT_DIR)
        icon.addPixmap(QtGui.QPixmap(f"{ROOT_DIR}/../pics/track.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.tracking_button.setIcon(icon)
        self.tracking_button.setIconSize(QtCore.QSize(40, 40))
        self.tracking_button.setObjectName("tracking_button")
        self.screenshot_button = QtWidgets.QPushButton(parent=self.frame)
        self.screenshot_button.setGeometry(QtCore.QRect(360, 360, 40, 40))
        self.screenshot_button.setAutoFillBackground(False)
        self.screenshot_button.setStyleSheet("background-color: transparent; border: 0px \n"
"")
        self.screenshot_button.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(f"{ROOT_DIR}/../pics/screenshot.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.screenshot_button.setIcon(icon1)
        self.screenshot_button.setIconSize(QtCore.QSize(40, 40))
        self.screenshot_button.setObjectName("screenshot_button")
        self.switch_button = QtWidgets.QPushButton(parent=self.frame)
        self.switch_button.setGeometry(QtCore.QRect(200, 360, 40, 40))
        self.switch_button.setAutoFillBackground(False)
        self.switch_button.setStyleSheet("background-color: transparent; border: 0px \n"
"")
        self.switch_button.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(f"{ROOT_DIR}/../pics/power.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.switch_button.setIcon(icon2)
        self.switch_button.setIconSize(QtCore.QSize(40, 40))
        self.switch_button.setObjectName("switch_button")
        self.video_screen = QtWidgets.QLabel(parent=self.frame)
        self.video_screen.setGeometry(QtCore.QRect(30, 20, 511, 321))
        self.video_screen.setStyleSheet("border: 12px solid transparent; /* Set border width and make it transparent */\n"
"border-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #eb3349, stop:1 #f45c43);\n"
"border-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #eb3349,\n"
"stop:1 #f45c43)  ;\n"
"border-radius: 20px")
        self.video_screen.setText("")
        self.video_screen.setPixmap(QtGui.QPixmap(f"{ROOT_DIR}/../pics/video.jpg"))
        self.video_screen.setScaledContents(True)
        self.video_screen.setObjectName("video_screen")
        self.initialization_button = QtWidgets.QPushButton(parent=self.frame)
        self.initialization_button.setGeometry(QtCore.QRect(565, 70, 121, 41))

        self.initialization_button.setLayoutDirection(QtCore.Qt.LayoutDirection.RightToLeft)
        self.initialization_button.setStyleSheet("background:qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgb(71,71,71), stop:1 rgb(33,34,33));\n"
"border:2px solid transparent;\n"
"border-color:rgb(18,19,18);\n"
"border-radius: 8px;\n"
"color: rgb(22,145,162);\n"
"font-weight:bold;\n"
"font-family: Verdana\n"
)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(f"{ROOT_DIR}/../pics/down-list.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.initialization_button.setIcon(icon3)
        self.initialization_button.setIconSize(QtCore.QSize(8, 8))
        self.initialization_button.setObjectName("initialization_button")
        self.listWidget = QtWidgets.QListWidget(parent=self.frame)
        self.listWidget.setGeometry(QtCore.QRect(580, 110, 87, 71))

        self.listWidget.setSpacing(2)
        self.listWidget.setStyleSheet("border-bottom-left-radius: 3px;\n"
"border-bottom-right-radius: 3px; \n"
"border: 2px solid transparent;\n"
"border-color: rgb(18,19,18);\n"
"background: rgb(19, 20,19);\n"
"color: rgb(125,126,125);\n"
"font-weight:bold;\n"
"font-family: Segoe UI;\n"
"font-size: 12px;\n"

)
        self.listWidget.setObjectName("listWidget")
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)

        self.Eraser_button = QtWidgets.QPushButton(parent=self.frame)
        self.Eraser_button.setGeometry(QtCore.QRect(565, 210, 121, 41))
        self.Eraser_button.setLayoutDirection(QtCore.Qt.LayoutDirection.RightToLeft)
        self.Eraser_button.setStyleSheet("background:qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgb(71,71,71), stop:1 rgb(33,34,33));\n"
"border:2px solid transparent;\n"
"border-color:rgb(18,19,18);\n"
"border-radius: 8px;\n"
"color: rgb(22,145,162);\n"
"font-weight:bold;\n"
"font-family: Verdana\n"
 )
        self.Eraser_button.setIcon(icon3)
        self.Eraser_button.setIconSize(QtCore.QSize(8, 8))
        self.Eraser_button.setObjectName("Eraser_button")

        # Drawn Region
        self.listWidget_2 = QtWidgets.QListWidget(parent=self.frame)
        self.listWidget_2.setGeometry(QtCore.QRect(580, 250, 87, 71))
        self.listWidget_2.setStyleSheet("border-bottom-left-radius: 3px;\n"
"border-bottom-right-radius: 3px; \n"
"border: 2px solid transparent;\n"
"border-color: rgb(18,19,18);\n"
"background: rgb(19, 20,19);\n"
"color: rgb(125,126,125);\n"
"font-weight:bold;\n"
"font-family: Verdana\n"
)
        self.listWidget_2.setObjectName("listWidget_2")
        self.listWidget_2.setSpacing(2)
        item = QtWidgets.QListWidgetItem()
        self.listWidget_2.addItem(item)
        self.listWidget.raise_()
        self.tracking_button.raise_()
        self.screenshot_button.raise_()
        self.switch_button.raise_()
        self.video_screen.raise_()
        self.initialization_button.raise_()
        self.Eraser_button.raise_()
        self.listWidget_2.raise_()
        self.horizontalLayout.addWidget(self.frame)
        self.frame_2 = QtWidgets.QFrame(parent=self.centralwidget)
        self.frame_2.setMinimumSize(QtCore.QSize(0, 0))
        self.frame_2.setMaximumSize(QtCore.QSize(40, 16777215))
        self.frame_2.setStyleSheet("background: #000000;\n"
"")
        self.frame_2.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_2.setObjectName("frame_2")
        self.camera_switch_button = QtWidgets.QPushButton(parent=self.frame_2)
        self.camera_switch_button.setGeometry(QtCore.QRect(5, 10, 30, 30))
        self.camera_switch_button.setAutoFillBackground(False)
        self.camera_switch_button.setStyleSheet("background-color: transparent; border: 0px \n"
"")
        self.camera_switch_button.setText("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(f"{ROOT_DIR}/../pics/target.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.camera_switch_button.setIcon(icon4)
        self.camera_switch_button.setIconSize(QtCore.QSize(30, 30))
        self.camera_switch_button.setObjectName("camera_switch_button")
        self.download_button = QtWidgets.QPushButton(parent=self.frame_2)
        self.download_button.setGeometry(QtCore.QRect(9, 250, 26, 26))
        self.download_button.setAutoFillBackground(False)
        self.download_button.setStyleSheet("background-color: transparent; border: 0px \n"
"")
        self.download_button.setText("")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(f"{ROOT_DIR}/../pics/downloading.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.download_button.setIcon(icon5)
        self.download_button.setIconSize(QtCore.QSize(25, 25))
        self.download_button.setObjectName("download_button")
        self.analysis_file = QtWidgets.QPushButton(parent=self.frame_2)
        self.analysis_file.setGeometry(QtCore.QRect(5, 180, 30, 30))
        self.analysis_file.setAutoFillBackground(False)
        self.analysis_file.setStyleSheet("background-color: transparent; border: 0px \n"
"")
        self.analysis_file.setText("")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(f"{ROOT_DIR}/../pics/csv.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.analysis_file.setIcon(icon6)
        self.analysis_file.setIconSize(QtCore.QSize(30, 30))
        self.analysis_file.setObjectName("analysis_file")
        self.search_button = QtWidgets.QPushButton(parent=self.frame_2)
        self.search_button.setGeometry(QtCore.QRect(-30, 120, 61, 30))
        self.search_button.setLayoutDirection(QtCore.Qt.LayoutDirection.RightToLeft)
        self.search_button.setAutoFillBackground(False)
        self.search_button.setStyleSheet("background-color: transparent; border: 0px \n"
"")
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(f"{ROOT_DIR}/..\\pics/search.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.search_button.setIcon(icon7)
        self.search_button.setIconSize(QtCore.QSize(30, 30))
        self.search_button.setObjectName("search_button")
        self.exit_button = QtWidgets.QPushButton(parent=self.frame_2)
        self.exit_button.setGeometry(QtCore.QRect(3, 380, 60, 30))
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(f"{ROOT_DIR}/..\\pics/exit.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.exit_button.setIcon(icon8)
        self.exit_button.setIconSize(QtCore.QSize(30, 30))
        self.exit_button.setObjectName("exit_button")
        self.horizontalLayout.addWidget(self.frame_2)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.initialization_button.setText(_translate("MainWindow", "Create region  "))
        __sortingEnabled = self.listWidget.isSortingEnabled()
        self.listWidget.setSortingEnabled(False)
        item = self.listWidget.item(0)
        item.setText(_translate("MainWindow", "Add point "))
        item = self.listWidget.item(1)
        item.setText(_translate("MainWindow", "Delete point"))
        item = self.listWidget.item(2)
        item.setText(_translate("MainWindow", "Area name"))
        self.listWidget.setSortingEnabled(__sortingEnabled)
        self.Eraser_button.setText(_translate("MainWindow", "  Eraser           "))
        __sortingEnabled = self.listWidget_2.isSortingEnabled()
        self.listWidget_2.setSortingEnabled(False)
        item = self.listWidget_2.item(0)
        item.setText(_translate("MainWindow", "Region 0"))
        self.listWidget_2.setSortingEnabled(__sortingEnabled)
        self.search_button.setText(_translate("MainWindow", "Search "))
        self.exit_button.setText(_translate("MainWindow", "hvgj"))
