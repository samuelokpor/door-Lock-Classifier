# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'deployable_classifier.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
import sys
import resources_rc
import cv2
import os
import warnings
import os
import random
from shutil import copyfile
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib.image import imread
import numpy as np
from keras.models import load_model
from keras.utils import load_img, img_to_array
import pathlib
import io
import time
import gxipy as gx

class Ui_MainWindow(QMainWindow):
    def __init__(self, parent=None) -> None:
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        self.show()
        self.timer_camera = QTimer(self)
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.slot_init()
        
        self.model = None
        self.image_labels = ['ES2T_left_lagan', 'ES2T_left_laxian', 'ES2T_right_lagan', 'ES2T_right_laxian', 'ES2V_left_laxian', 'ES2V_right_laxian']
        


        #CONNECTING BUTTONs
        self.opencameraButton.clicked.connect(self.collect_images_fuction)
        self.dataAugButton.clicked.connect(self.data_Augmentation)
        self.trainModelButton.clicked.connect(self.visualize_data)
        self.classifyReal_timeButton.clicked.connect(self.classifier_loop)
        self.pushButton.clicked.connect(self.stop_inference)


    def slot_init(self):
        pass
        self.timer_camera.timeout.connect(self.show_camera) 

    def show_camera(self):
            
        flag, self.imageopened = self.cap.read()
        show = cv2.resize(self.imageopened, (640, 480))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        self.showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.image1.setPixmap(QPixmap.fromImage(self.showImage))
        self.image1.setScaledContents(True)

    def collect_images_fuction(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM, cv2.CAP_DSHOW)
            if flag == False:
                msg = QMessageBox.warning(self, u"Warning", u"Connect-Camera-Source",
                                                    buttons=QMessageBox.Ok,
                                                    defaultButton=QMessageBox.Ok)
            # if msg==QtGui.QMessageBox.Cancel:
            #                     pass
            else:
                self.timer_camera.start(30)

                self.opencameraButton.setText(u'Camera Opened')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.image1.clear()
            self.opencameraButton.setText(u'Camera Closed') 



    
    base_dir = 'Images_dir'

    def data_Augmentation(self):
        warnings.filterwarnings('ignore')
        #get all the paths
        data_dir_list = os.listdir('inteva_fenlei_Images')
        print(data_dir_list)
        path, dirs, files = next(os.walk("inteva_fenlei_Images","data_dir_list"))
        print(f'{path}, {dirs}, {files}')
        original_dataset_dir = 'inteva_fenlei_Images'
        base_dir = 'Images_dir'
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        
        train_dir = os.path.join(base_dir, 'train')
        if not os.path.exists(train_dir):
            pass
            os.mkdir(train_dir)

        validation_dir = os.path.join(base_dir, 'validation')
        if not os.path.exists(validation_dir):
            os.mkdir(validation_dir)

        train_ES2T_left_lagan_dir = os.path.join(train_dir, 'ES2T_left_lagan')
        if not os.path.exists(train_ES2T_left_lagan_dir):
            pass
            os.mkdir(train_ES2T_left_lagan_dir)
        train_ES2T_left_laxian_dir = os.path.join(train_dir, 'ES2T_left_laxian')
        if not os.path.exists(train_ES2T_left_laxian_dir):
            os.mkdir(train_ES2T_left_laxian_dir)
        train_ES2T_right_lagan_dir = os.path.join(train_dir, 'ES2T_right_lagan')
        if not os.path.exists(train_ES2T_right_lagan_dir):
            os.mkdir(train_ES2T_right_lagan_dir)
        train_ES2T_right_laxian_dir = os.path.join(train_dir, 'ES2T_right_laxian')
        if not os.path.exists(train_ES2T_right_laxian_dir):
            os.mkdir(train_ES2T_right_laxian_dir)
        train_ES2V_left_laxian_dir = os.path.join(train_dir, 'ES2V_left_laxian')
        if not os.path.exists(train_ES2V_left_laxian_dir):
            os.mkdir(train_ES2V_left_laxian_dir)
        train_ES2V_right_laxian_dir = os.path.join(train_dir, 'ES2V_right_laxian')
        if not os.path.exists(train_ES2V_right_laxian_dir):
            os.mkdir(train_ES2V_right_laxian_dir)

        validation_ES2T_left_lagan_dir = os.path.join(validation_dir, 'ES2T_left_lagan')
        if not os.path.exists(validation_ES2T_left_lagan_dir):
            pass
            os.mkdir(validation_ES2T_left_lagan_dir)

        validation_ES2T_left_laxian_dir = os.path.join(validation_dir, 'ES2T_left_laxian')
        if not os.path.exists(validation_ES2T_left_laxian_dir):
            os.mkdir(validation_ES2T_left_laxian_dir)

        validation_ES2T_right_lagan_dir = os.path.join(validation_dir, 'ES2T_right_lagan')
        if not os.path.exists(validation_ES2T_right_lagan_dir):
            os.mkdir(validation_ES2T_right_lagan_dir)

        validation_ES2V_left_laxian_dir = os.path.join(validation_dir, 'ES2V_left_laxian')
        if not os.path.exists(validation_ES2V_left_laxian_dir):
            os.mkdir(validation_ES2V_left_laxian_dir)

        validation_ES2V_right_laxian_dir = os.path.join(validation_dir, 'ES2V_right_laxian')
        if not os.path.exists(validation_ES2V_right_laxian_dir):
            os.mkdir(validation_ES2V_right_laxian_dir)
        validation_ES2T_right_laxian_dir = os.path.join(validation_dir, 'ES2T_right_laxian')
        if not os.path.exists(validation_ES2T_right_laxian_dir):
            os.mkdir(validation_ES2T_right_laxian_dir)

    
    global image_labels
    global  split_size
    ES2T_left_lagan_SOURCE_dir = 'inteva_fenlei_Images/ES2T_left_lagan/'
    TRAINING_ES2T_left_lagan_dir = 'Images_dir/train/ES2T_left_lagan/'
    VALID_ES2T_left_lagan_dir = 'Images_dir/validation/ES2T_left_lagan/'
    ES2T_left_laxian_SOURCE_dir = 'inteva_fenlei_Images/ES2T_left_laxian/'
    TRAINING_ES2T_left_laxian_dir = 'Images_dir/train/ES2T_left_laxian/'
    VALID_ES2T_left_laxian_dir = 'Images_dir/validation/ES2T_left_laxian/'
    ES2T_right_lagan_SOURCE_dir = 'inteva_fenlei_Images/ES2T_right_lagan/'
    TRAINING_ES2T_right_lagan_dir = 'Images_dir/train/ES2T_right_lagan/'
    VALID_ES2T_right_lagan_dir = 'Images_dir/validation/ES2T_right_lagan/'
    ES2T_right_laxian_SOURCE_dir = 'inteva_fenlei_Images/ES2T_right_laxian/'
    TRAINING_ES2T_right_laxian_dir = 'Images_dir/train/ES2T_right_laxian/'
    VALID_ES2T_right_laxian_dir = 'Images_dir/validation/ES2T_right_laxian/'
    ES2V_left_laxian_SOURCE_dir = 'inteva_fenlei_Images/ES2V_left_laxian/'
    TRAINING_ES2V_left_laxian_dir = 'Images_dir/train/ES2V_left_laxian/'
    VALID_ES2V_left_laxian_dir = 'Images_dir/validation/ES2V_left_laxian/'
    ES2V_right_laxian_SOURCE_dir = 'inteva_fenlei_Images/ES2V_right_laxian/'
    TRAINING_ES2V_right_laxian_dir = 'Images_dir/train/ES2V_right_laxian/'
    VALID_ES2V_right_laxian_dir = 'Images_dir/validation/ES2V_right_laxian/'
    split_size = 0.85
    image_labels = ['ES2T_left_lagan','ES2T_left_laxian','ES2T_right_lagan','ES2T_right_laxian','ES2V_left_laxian','ES2V_right_laxian']
    def split_data(SOURCE, TRAINING, VALIDATION, SPLIT_SIZE):       
        files = []
        for filename in os.listdir(SOURCE):
                file = SOURCE + filename
                if os.path.getsize(file)> 0:
                    pass
                    
                    files.append(filename)
                else:
                    pass
                    print(filename + "Ignored")
                
        training_length = int(len(files)* SPLIT_SIZE)
        valid_length = int(len(files) - training_length)
        shuffled_set = random.sample(files, len(files))
        training_set = shuffled_set[0:training_length]
        valid_set = shuffled_set[training_length:]
        
        for filename in training_set:

                this_file = SOURCE + filename
                destination = TRAINING + filename
                copyfile(this_file, destination)
                
        for filename in valid_set:
                this_file = SOURCE + filename
                destination = VALIDATION + filename
                copyfile(this_file, destination)

    
    split_data(ES2T_left_lagan_SOURCE_dir,TRAINING_ES2T_left_lagan_dir,VALID_ES2T_left_lagan_dir,split_size )
    split_data(ES2T_left_laxian_SOURCE_dir,TRAINING_ES2T_left_laxian_dir,VALID_ES2T_left_laxian_dir,split_size)
    split_data(ES2T_right_lagan_SOURCE_dir,TRAINING_ES2T_right_lagan_dir,VALID_ES2T_right_lagan_dir,split_size)
    split_data(ES2T_right_laxian_SOURCE_dir,TRAINING_ES2T_right_laxian_dir,VALID_ES2T_right_laxian_dir,split_size)
    split_data(ES2V_left_laxian_SOURCE_dir,TRAINING_ES2V_left_laxian_dir,VALID_ES2V_left_laxian_dir,split_size)
    split_data(ES2V_right_laxian_SOURCE_dir,TRAINING_ES2V_right_laxian_dir,VALID_ES2V_right_laxian_dir,split_size)
    
    def visualize_data(self):
        image_folder = ['ES2T_left_lagan','ES2T_left_laxian','ES2T_right_lagan','ES2T_right_laxian','ES2V_left_laxian','ES2V_right_laxian']
        nimgs={}
        num_images = ''
        for i in image_folder:
            nimages = len(os.listdir('Images_dir/train/'+i+'/'))
            nimgs[i]=nimages
            num_images += 'Training {} images are: '.format(i) + str(nimages) + '\n'
        fig, ax = plt.subplots(dpi=75)
        ax.bar(range(len(nimgs)), list(nimgs.values()), align='center')
        ax.set_xticks(range(len(nimgs)))
        ax.set_xticklabels(list(nimgs.keys()))
        ax.set_title('Distribution of different classes in Training Dataset')
        canvas = FigureCanvasQTAgg(fig)
        canvas.draw()

        img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(canvas.get_width_height()[::-1] + (3,))

        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img, width, height, bytes_per_line, QImage.Format_RGB888)

        self.resultsLabel.setPixmap(QPixmap.fromImage(q_img))
        palette = QPalette()
        palette.setColor(QPalette.WindowText, Qt.black)
        self.image2.setPalette(palette)
        self.image2.setFont(QFont("Roman times", 10, QFont.Bold))
        self.image2.setText(num_images)
        img_width=256; img_height=256
        batch_size=16 

        TRAINING_DIR = 'Images_dir/train/'
        train_datagen = ImageDataGenerator(rescale = 1/255.0,
                                  rotation_range=30,
                                  zoom_range=0.4,
                                  horizontal_flip=True)
        train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                   batch_size=batch_size,
                                                   class_mode='categorical',
                                                   target_size=(img_height, img_width))
        
        VALIDATION_DIR = 'Images_dir/validation/'

        validation_datagen = ImageDataGenerator(rescale = 1/255.0)

        validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                                batch_size=batch_size,
                                                                class_mode='categorical',
                                                                target_size=(img_height, img_width))
        callbacks  = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
        #auto save best Model
        best_model_file = 'best_model/best_weights.h5'
        best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose=1, save_best_only= True)

        model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(img_height, img_width, 3)), MaxPooling2D(2,2),
        Conv2D(32, (3,3), activation='relu'), MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(256, (3,3), activation='relu'),
        Conv2D(256, (3,3), activation='relu'),
        Conv2D(256, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(512, activation='relu'),
        Dense(6, activation='softmax')
        ])
        model.summary()
        model.compile(optimizer='Adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        history = model.fit_generator(train_generator, epochs=10, verbose=1, validation_data=validation_generator, callbacks=[best_model])
        model.save(best_model_file)

        acc = history.history['accuracy'][::5]
        val_acc = history.history['val_accuracy'][::5]
        loss = history.history['loss'][::5]
        val_loss = history.history['val_loss'][::5]
        epochs = range(0, 10, 5)

        fig = plt.figure(figsize=(7,7))
        plt.plot(history.history['accuracy'], 'r', label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], 'b', label='Validation Accuracy')
        plt.xticks(np.arange(0, 11, 5))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and validation accuracy')
        plt.legend(loc='lower right')
        plt.savefig('graphs/accuracy.png')

        pixmap = QPixmap('accuracy.png')
        self.image1.setPixmap(pixmap)
        self.image1.setScaledContents(True)
        self.image1.show()
        # model.summary()
        # model.compile(optimizer='Adam',
        #      loss='categorical_crossentropy',
        #      metrics=['accuracy'])
        # history = model.fit_generator(train_generator,epochs=10,verbose=1,validation_data=validation_generator, callbacks = [best_model])
        # model.save(best_model_file)
        # acc = history.history['accuracy'][::5]
        # val_acc = history.history['val_accuracy'][::5]
        # loss = history.history['loss'][::5]
        # val_loss = history.history['val_loss'][::5]
        # epochs = range(0, 10, 5) # increment of 5
        # fig = plt.figure(figsize=(7,7))
        # plt.plot(epochs, acc, 'r', label="Training Accuracy")
        # plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.title('Training and validation accuracy')
        # plt.legend(loc='lower right')
        # plt.savefig('graphs/accuracy.png')

        # pixmap = QPixmap('accuracy.png')
        # self.image1.setPixmap(pixmap)
        # self.image1.setScaledContents(True)
        # self.image1.show()


    def pre_process(self, frame):
            
            pass
            np.expand_dims(frame, 0)
            imgH, imgW = 256,256
            resized_image = cv2.resize(frame, (imgW, imgH))
            
            if len(resized_image.shape)==2:
                resized_image = resized_image[:,:,np.newaxis]
            resized_image = resized_image.astype('float32')
            resized_image = resized_image.transpose((0, 1, 2)) / 255
        
            return resized_image[np.newaxis,:,:,:]
    def softmax(self, x):
            
            pass
            """ softmax function """
            x -= np.max(x, axis = 1, keepdims = True)
            x = np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)
            return x
    


    def stop_inference(self):

        self.stop_camera = True

   
    def classifier_loop(self):
        print("")
        print("-------------------------------------------------------------")
        print("Load model")
        st_ = time.time()
        best_model_file = 'best_model/best_weights.h5'
        self.model = load_model(best_model_file)
        self.image_labels = ['ES2T_left_lagan', 'ES2T_left_laxian', 'ES2T_right_lagan', 'ES2T_right_laxian', 'ES2V_left_laxian', 'ES2V_right_laxian']
        print("Load model cost time: ", time.time() - st_)
        # print the demo information
        print("")
        print("-------------------------------------------------------------")
        print("Sample to show how to acquire mono image continuously and show acquired image.")
        print("-------------------------------------------------------------")
        print("")
        print("Initializing......")
        print("")

        device_manager = gx.DeviceManager()
        dev_num, dev_info_list = device_manager.update_device_list()

        if dev_num == 0:
            print("No device found")
            return

        cam = device_manager.open_device_by_index(1)
      
        if cam is None:
            print("No U3V device found")
            return

        cam.stream_on()
        self.stop_camera = False

        print("Camera is open")
        total = 0
        ES2T_left_lagan = 0
        ES2T_left_laxian = 0
        ES2T_right_lagan = 0
        ES2T_right_laxian = 0
        ES2V_left_laxian = 0
        ES2V_right_laxian = 0
        self.image_labels = ['ES2T_left_lagan','ES2T_left_laxian','ES2T_right_lagan','ES2T_right_laxian','ES2V_left_laxian','ES2V_right_laxian']
        while not self.stop_camera:
            try:
                #get raw Image
                raw_image = cam.data_stream[0].get_image()
                if raw_image is None:
                    print("Getting Image Failed")
                    continue
                
                #convert Image to numpy array
                image = raw_image.convert("RGB")
                image_numpy = image.get_numpy_array()
                image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)

                print("---------------------------------------------")
                st_ = time.time()
            

                input_batch = self.pre_process(frame=image_numpy)
                output = self.model.predict(input_batch)
                conf = np.max(output, axis=1)[0]
                label = self.image_labels[np.argmax(output, axis=1)[0]]
                cost = time.time() - st_
                print("Inference time: ", cost)
                st_ = time.time()
                total += 1
                if label == "ES2T_left_lagan":
                    ES2T_left_lagan += 1
                    time.sleep(1) 
                    display_label = "ES2T_left_lagan"
                    display_count = ES2T_left_lagan
                elif label == "ES2T_left_laxian":
                    ES2T_left_laxian += 1
                    time.sleep(1) 
                    display_label = "ES2T_left_laxian"
                    display_count = ES2T_left_laxian
                elif label == "ES2T_right_lagan":
                    ES2T_right_lagan += 1
                    time.sleep(1) 
                    display_label = "ES2T_right_lagan"
                    display_count = ES2T_right_lagan
                elif label == "ES2T_right_laxian":
                    ES2T_right_laxian += 1
                    time.sleep(1) 
                    display_label = "ES2T_right_laxian"
                    display_count = ES2T_right_laxian
                elif label == "ES2V_left_laxian":
                    ES2V_left_laxian += 1
                    time.sleep(1) 
                    display_label = "ES2V_left_laxian"
                    display_count = ES2V_left_laxian
                elif label == "ES2V_right_laxian":
                    ES2V_right_laxian += 1
                    time.sleep(1) 
                    display_label = "ES2V_right_laxian"
                    display_count = ES2V_right_laxian

                numpy_image = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2GRAY)
                numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_GRAY2BGR)
                cv2.putText(numpy_image, "Result: {}".format(display_label), (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
                cv2.putText(numpy_image, "Confidence: {:.3f}".format(conf), (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
                cv2.putText(numpy_image, "Time: {:.3f}".format(cost), (50, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
                cv2.putText(numpy_image, "Count: {}".format(display_count), (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
                cv2.putText(numpy_image, "Press 'q' to quit!", (50, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)

                # Save the predicted label as a string into a txt file
                with open("Predictions/predicted_labels.txt", "a") as file:

                    file.write("{} Confidence: {:.3f} Time: {:.3f} Count: {}\n".format(display_label, conf, cost, display_count))

                # Set the color of the text in the QLabel using a QPalette
                palette = QPalette()
                if display_label == "Positive":
                    palette.setColor(QPalette.WindowText, Qt.darkGreen)
                elif display_label == "Negative":
                    palette.setColor(QPalette.WindowText, Qt.darkRed)
                else:
                    palette.setColor(QPalette.WindowText, Qt.black)
                self.resultsLabel.setPalette(palette)

                # Set the text of the QLabel to the display label
                self.resultsLabel.setText(display_label)

                height, width, channel = numpy_image.shape
                bytesPerLine = 3 * width
                qImg = QImage(numpy_image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                self.image2.setPixmap(QPixmap.fromImage(qImg))
                self.image2.setScaledContents(True)

                # Wait for 1 millisecond to allow the GUI to update
                cv2.waitKey(1)

            except Exception as e:
                #print the exception and break out of the loop if error occurs
                print(f"An error occured {e}")
                break

        cam.close_device()
        cv2.destroyAllWindows()


            


    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(842, 803)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMaximumSize(QSize(16777215, 16777215))
        MainWindow.setStyleSheet(u"background-color:#1f232a;")
        self.actionNew = QAction(MainWindow)
        self.actionNew.setObjectName(u"actionNew")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.leftMenuContainer = QWidget(self.centralwidget)
        self.leftMenuContainer.setObjectName(u"leftMenuContainer")
        self.leftMenuContainer.setStyleSheet(u"background-color:#d5e4ff;")
        self.horizontalLayout_3 = QHBoxLayout(self.leftMenuContainer)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.cameraHolderFrame = QFrame(self.leftMenuContainer)
        self.cameraHolderFrame.setObjectName(u"cameraHolderFrame")
        self.cameraHolderFrame.setFrameShape(QFrame.StyledPanel)
        self.cameraHolderFrame.setFrameShadow(QFrame.Raised)
        self.verticalLayout = QVBoxLayout(self.cameraHolderFrame)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.image1 = QLabel(self.cameraHolderFrame)
        self.image1.setObjectName(u"image1")
        self.image1.setMinimumSize(QSize(371, 371))
        font = QFont()
        font.setFamily(u"Times New Roman")
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.image1.setFont(font)
        self.image1.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.image1)

        self.image2 = QLabel(self.cameraHolderFrame)
        self.image2.setObjectName(u"image2")
        self.image2.setMinimumSize(QSize(371, 371))
        self.image2.setMaximumSize(QSize(16777215, 16777215))
        font1 = QFont()
        font1.setFamily(u"Times New Roman")
        font1.setPointSize(11)
        self.image2.setFont(font1)
        self.image2.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.image2)


        self.horizontalLayout_3.addWidget(self.cameraHolderFrame)


        self.horizontalLayout.addWidget(self.leftMenuContainer)

        self.rightMenuContainer = QWidget(self.centralwidget)
        self.rightMenuContainer.setObjectName(u"rightMenuContainer")
        self.rightMenuContainer.setStyleSheet(u"background-color:#868cff;")
        self.horizontalLayout_2 = QHBoxLayout(self.rightMenuContainer)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.functionsFrame = QFrame(self.rightMenuContainer)
        self.functionsFrame.setObjectName(u"functionsFrame")
        self.functionsFrame.setFrameShape(QFrame.StyledPanel)
        self.functionsFrame.setFrameShadow(QFrame.Raised)
        self.verticalLayout_2 = QVBoxLayout(self.functionsFrame)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.frame = QFrame(self.functionsFrame)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.frame)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.resultsLabel = QLabel(self.frame)
        self.resultsLabel.setObjectName(u"resultsLabel")
        self.resultsLabel.setFont(font1)
        self.resultsLabel.setAlignment(Qt.AlignCenter)

        self.verticalLayout_3.addWidget(self.resultsLabel)

        self.groupBox = QGroupBox(self.frame)
        self.groupBox.setObjectName(u"groupBox")
        font2 = QFont()
        font2.setFamily(u"Times New Roman")
        self.groupBox.setFont(font2)
        self.verticalLayout_4 = QVBoxLayout(self.groupBox)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.trainModelButton = QPushButton(self.groupBox)
        self.trainModelButton.setObjectName(u"trainModelButton")
        self.trainModelButton.setFont(font2)
        self.trainModelButton.setCursor(QCursor(Qt.PointingHandCursor))
        icon = QIcon()
        icon.addFile(u":/icons/feather/play.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.trainModelButton.setIcon(icon)
        self.trainModelButton.setIconSize(QSize(24, 24))

        self.verticalLayout_4.addWidget(self.trainModelButton)

        self.pushButton = QPushButton(self.groupBox)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setFont(font2)
        self.pushButton.setCursor(QCursor(Qt.PointingHandCursor))
        icon1 = QIcon()
        icon1.addFile(u":/icons/feather/stop-circle.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.pushButton.setIcon(icon1)
        self.pushButton.setIconSize(QSize(24, 24))

        self.verticalLayout_4.addWidget(self.pushButton)

  



        self.dataAugButton = QPushButton(self.groupBox)
        self.dataAugButton.setObjectName(u"dataAugButton")
        self.dataAugButton.setFont(font1)
        icon4 = QIcon()
        icon4.addFile(u":/icons/feather/cast.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.dataAugButton.setIcon(icon4)
        self.dataAugButton.setIconSize(QSize(24, 24))

        self.verticalLayout_4.addWidget(self.dataAugButton)


        self.verticalLayout_3.addWidget(self.groupBox)


        self.verticalLayout_2.addWidget(self.frame)

        self.frame_2 = QFrame(self.functionsFrame)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setFrameShape(QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.verticalLayout_5 = QVBoxLayout(self.frame_2)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.frame_3 = QFrame(self.frame_2)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setFrameShape(QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Raised)
        self.verticalLayout_7 = QVBoxLayout(self.frame_3)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.frame_5 = QFrame(self.frame_3)
        self.frame_5.setObjectName(u"frame_5")
        self.frame_5.setFrameShape(QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_4 = QHBoxLayout(self.frame_5)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.opencameraButton = QPushButton(self.frame_5)
        self.opencameraButton.setObjectName(u"opencameraButton")
        self.opencameraButton.setCursor(QCursor(Qt.PointingHandCursor))
        icon5 = QIcon()
        icon5.addFile(u":/icons/feather/camera-off.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.opencameraButton.setIcon(icon5)
        self.opencameraButton.setIconSize(QSize(24, 24))

        self.horizontalLayout_4.addWidget(self.opencameraButton)

        self.capturebutton = QPushButton(self.frame_5)
        self.capturebutton.setObjectName(u"capturebutton")
        self.capturebutton.setCursor(QCursor(Qt.PointingHandCursor))
        icon6 = QIcon()
        icon6.addFile(u":/icons/feather/camera.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.capturebutton.setIcon(icon6)
        self.capturebutton.setIconSize(QSize(24, 24))

        self.horizontalLayout_4.addWidget(self.capturebutton)


        self.verticalLayout_7.addWidget(self.frame_5)

        self.frame_4 = QFrame(self.frame_3)
        self.frame_4.setObjectName(u"frame_4")
        self.frame_4.setFrameShape(QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QFrame.Raised)
        self.verticalLayout_6 = QVBoxLayout(self.frame_4)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.classifyFinalImageButton = QPushButton(self.frame_4)
        self.classifyFinalImageButton.setObjectName(u"classifyFinalImageButton")
        self.classifyFinalImageButton.setFont(font1)
        self.classifyFinalImageButton.setCursor(QCursor(Qt.PointingHandCursor))
        self.classifyFinalImageButton.setAutoFillBackground(True)

        self.verticalLayout_6.addWidget(self.classifyFinalImageButton)

        self.classifyReal_timeButton = QPushButton(self.frame_4)
        self.classifyReal_timeButton.setObjectName(u"classifyReal_timeButton")
        self.classifyReal_timeButton.setFont(font1)
        self.classifyReal_timeButton.setCursor(QCursor(Qt.PointingHandCursor))

        self.verticalLayout_6.addWidget(self.classifyReal_timeButton)


        self.verticalLayout_7.addWidget(self.frame_4)


        self.verticalLayout_5.addWidget(self.frame_3)


        self.verticalLayout_2.addWidget(self.frame_2)


        self.horizontalLayout_2.addWidget(self.functionsFrame)


        self.horizontalLayout.addWidget(self.rightMenuContainer)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Classifer", None))
        self.actionNew.setText(QCoreApplication.translate("MainWindow", u"New", None))
        self.image1.setText(QCoreApplication.translate("MainWindow", u"Image 1", None))
        self.image2.setText(QCoreApplication.translate("MainWindow", u"Image 2", None))
        self.resultsLabel.setText(QCoreApplication.translate("MainWindow", u"Results", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Training Function", None))
#if QT_CONFIG(tooltip)
        self.trainModelButton.setToolTip(QCoreApplication.translate("MainWindow", u"Train Model", None))
#endif // QT_CONFIG(tooltip)
        self.trainModelButton.setText(QCoreApplication.translate("MainWindow", u"Train", None))
#if QT_CONFIG(tooltip)
        self.pushButton.setToolTip(QCoreApplication.translate("MainWindow", u"Stop Training", None))
#endif // QT_CONFIG(tooltip)
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"Stop", None))
#if QT_CONFIG(tooltip)
        
#endif // QT_CONFIG(tooltip)
        
#if QT_CONFIG(tooltip)
        
#endif // QT_CONFIG(tooltip)
        
        self.dataAugButton.setText(QCoreApplication.translate("MainWindow", u"Data Augmentation", None))
#if QT_CONFIG(tooltip)
        self.opencameraButton.setToolTip(QCoreApplication.translate("MainWindow", u"Open Camera", None))
#endif // QT_CONFIG(tooltip)
        self.opencameraButton.setText(QCoreApplication.translate("MainWindow", u"Open Camera", None))
#if QT_CONFIG(tooltip)
        self.capturebutton.setToolTip(QCoreApplication.translate("MainWindow", u"Capture Image", None))
#endif // QT_CONFIG(tooltip)
        self.capturebutton.setText(QCoreApplication.translate("MainWindow", u" Capture  ", None))
        self.classifyFinalImageButton.setText(QCoreApplication.translate("MainWindow", u"Classify Image", None))
#if QT_CONFIG(tooltip)
        self.classifyReal_timeButton.setToolTip(QCoreApplication.translate("MainWindow", u"Real-Time", None))
#endif // QT_CONFIG(tooltip)
        self.classifyReal_timeButton.setText(QCoreApplication.translate("MainWindow", u"Classify Real-Time", None))
    # retranslateUi

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Ui_MainWindow()
    window.show()
    sys.exit(app.exec_())
