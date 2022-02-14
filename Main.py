import SortData as Sd #модуль для манипуляции с файлами

import numpy as np #работа с массивами

import cv2 #чтение и обработка изображений

import matplotlib.pyplot as plt #визуализация данных

import random #для перемешивания данных

import Models  as mod #данные по архитектурам нейросетей

from tensorflow import keras

from keras.models import load_model

from keras.callbacks import History 

from PIL import Image

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing.image import load_img

from keras.datasets import cifar10




class Main():

    def get_more_images(imgs):
            vert_flip_imgs = []
            hori_flip_imgs = []
            for i in range(0, imgs.shape[0]):
                a = imgs[i, :, :, 0]
                b = imgs[i, :, :, 1]
                c = imgs[i, :, :, 2]
                av = cv2.flip(a, 1)
                ah = cv2.flip(a, 0)
                bv = cv2.flip(b, 1)
                bh = cv2.flip(b, 0)
                cv = cv2.flip(c, 1)
                ch = cv2.flip(c, 0)
            vert_flip_imgs.append(np.dstack((av, bv, cv)))
            hori_flip_imgs.append(np.dstack((ah, bh, ch)))
            v = np.array(vert_flip_imgs)
            h = np.array(hori_flip_imgs)
            more_images = np.concatenate((imgs, v, h))
            return more_images



    def keras_generator(images_numbers, batch_size):
       while True:
         x_batch = []
         y_batch = []

         for i in range(batch_size): #чтение случайной картинки и маски
            numbers = random.randint(0,len(images_numbers)-1)

            img = cv2.imread('D:\\wda-2021-03-28\\wda-2021-03-28\\sourse\\'+
                             images_numbers[numbers]+'.source.tiff')
            mask = cv2.imread('D:\\wda-2021-03-28\\wda-2021-03-28\\garbage\\'+
                              images_numbers[numbers]+'.garbage.tiff')
        #    print(mask)

            img = cv2.resize(img, (256, 256))
            mask = cv2.resize(mask, (256, 256))
                  
            x_batch += [img] 
            y_batch += [mask]

        # x_batch = np.array(x_batch)/ 255
       #  y_batch = np.array(y_batch)

         x_batch = np.array(x_batch)/255
         y_batch = np.array(y_batch)/255
        
       #  print("Unique values in the mask are: ", np.unique(y_batch))
      #   print("Unique values in the mask are: ")
      #   print("Unique values in the mask are: ", np.unique(x_batch))
         #os.pause

         yield x_batch, y_batch

    def keras_generator_new(masking,data_im):
         x_batch = []
         y_batch = []
         print(len(masking))
         for i in range(len(masking)): #чтение случайной картинки и маски
            number = i
           # print(i)
           # print(train_data_image[number])
           # print(train_data_mask[number])  
          #    print(train_data_image[number])
            img = cv2.imread(data_im[number])
            mask = cv2.imread(masking[number])
            print(data_im[number])


            img = cv2.resize(img, (256, 256))
            mask = cv2.resize(mask, (256, 256))
                  
            x_batch += [img] 
            y_batch += [mask]

         x_batch = np.asarray(x_batch).astype('float32').reshape((-1,1))
         y_batch = np.asarray(y_batch).astype('float32')
         
         return x_batch, y_batch

    if __name__ == "__main__":
        print('Begin')
        Data_list = []


        Image_path= 'D:\\wda-2021-03-28\\wda-2021-03-28\\sourse'
        Garbage_path = 'D:\\wda-2021-03-28\\wda-2021-03-28\\garbage'
        Wda_path = 'D:\\wda-2021-03-28\\wda-2021-03-28\\wda'

        Names = Sd.SortData.get_file_names_numbers(Image_path)

        print(Names)
        #данные для обучения
        train_numbers = []
        val_numbers = []

        for i in range((len(Names)) - 255):
            train_numbers.append(Names[i])

        for i in range(255):
            val_numbers.append(Names[i])

      #  print(len(val_data_mask))
      #  print(len(train_data_mask))

   #     for i in  train_numbers:
          #  for j in val_numbers:
             #   if i == j:
                #    Data_list.append(i)
                    #break
  

        for x, y in keras_generator(train_numbers, 16):
            break
        
    #    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 25))
    #    axes[0].imshow(x[1])
    #    axes[1].imshow(y[1])



    #    plt.show()

     #   history = History()

        best_w = keras.callbacks.ModelCheckpoint('unet_best.h5',
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=True,
                                save_weights_only=True,
                                mode='auto',
                                save_freq=1)

        last_w = keras.callbacks.ModelCheckpoint('unet_last.h5',
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=False,
                                save_weights_only=True,
                                mode='auto',
                                save_freq=1)
        

        my_callbacks = [best_w, last_w] #сохранение текущего и лучшего результатов 

        batch_size = 16



       # image_datagen = ImageDataGenerator(width_shift_range=0.1,
       #          height_shift_range=0.1,) # custom fuction for each image you can use resnet one too.
        
   #     mask_datagen = ImageDataGenerator()  # to make mask as feedable formate (256,256,1)

      #  image_generator =image_datagen.flow_from_directory("D:\wda-2021-03-28\wda-2021-03-28\image",
                                            #        class_mode=None, batch_size = 1, seed = 123)

        #mask_generator = mask_datagen.flow_from_directory("D:\\wda-2021-03-28\\wda-2021-03-28\\garbage",
                                                 #  class_mode=None, batch_size = 16, seed = 123)

     #   mask_generator = mask_datagen.flow_from_directory("D:\wda-2021-03-28\wda-2021-03-28\mask",
            #                                       class_mode=None, batch_size = 1, seed = 123)
       # train_generator = zip(image_generator, mask_generator)

      #  print(train_generator)

        model = mod.Neural.Unet()
        
        batch_size = 16
        epochs = 10 
        img_size = (256,256)


        model.summary()
      #  model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=my_callbacks)
        model.fit(keras_generator(train_numbers, batch_size),
              steps_per_epoch=50,
              epochs=30,
              verbose=1,
              callbacks=my_callbacks,
              validation_data=keras_generator(val_numbers, batch_size),
              validation_steps=50,
              class_weight=None,
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False,
              shuffle= True,
              initial_epoch=0)

        
     #   model.fit(train_generator,
     #         steps_per_epoch=150,
    #          epochs=30,
     #         verbose=1,
     #         callbacks=my_callbacks,
     #         validation_data=keras_generator(val_data_mask, val_image, batch_size),
     #         validation_steps=50,
    #          class_weight=None,
     #         max_queue_size=10,
      #        workers=1,
      #        use_multiprocessing=False,
      #        shuffle=True,
      #        initial_epoch=0)
      
        model.load_weights('unet_last.h5')

        pred = model.predict(x)
        im_id = 1
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(25, 25))

        axes[0].imshow(x[im_id])
        axes[1].imshow(pred[im_id])

        for i in range(256):
            for j in range(256):
              #  pred[1,i,j,0] = pred[1,i,j,0] * 255
                if pred[1,i,j,0] > 0.5:
                    pred[1,i,j,0] = 1
                else: 
                    pred[1,i,j,0] = 0
                    
        axes[2].imshow(y[im_id])
        axes[3].imshow(pred[im_id])
        plt.show()




    #    a = len(Mask_garbage_path) количество элементов списка
    #    b = len(Mask_wda_path)
    #    c = len(Image_Sourse)
        