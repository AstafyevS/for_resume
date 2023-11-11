#!/usr/bin/env python
# coding: utf-8

# <div style="border:solid Chocolate 2px; padding: 40px">
# 
# Привет, меня зовут Александр Куимов. Я буду ревьюером твоего проекта. Ты можешь обращаться ко мне на "ты"😏 Надеюсь, тебя также не смутит, если я буду обращаться к тебе на "ты", но если это неудобно, обязательно скажи об этом!
# 
# Пожалуйста, не удаляй мои комментарии, они будут особенно полезны для нашей работы в случае повторной проверки проекта. 
# 
# Ты также можешь реагировать на мои комментарии своими по шаблону, показанному чуть ниже. Это нужно, чтобы не создавалась путаница😉
# 
# Ты можешь найти мои комментарии, обозначенные <font color='green'>зеленым</font>, <font color='gold'>желтым</font> и <font color='red'>красным</font> цветами, например:
# 
# <br/>
# 
# <div class="alert alert-success">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> В случае, если решение на отдельном шаге является полностью правильным.
# </div>
# 
# <br/>
# 
# <div class="alert alert-warning">
#     <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации💡:</b> В случае, когда решение на отдельном шаге станет еще лучше, если внести небольшие коррективы.
# </div>
# 
# 
# <br/>
# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера <a class="tocSkip"></h2>
# 
#     
# <b>На доработку🤔:</b>
#  В случае, когда решение на отдельном шаге требует существенной переработки и внесения правок. Напоминаю, что проект не может быть принят с первого раза, если ревью содержит комментарии, рекомендующие доработать шаги.
# </div>
#     
#     
# <br/>    
# <div class="alert alert-info">
# <h2> Комментарий студента: <a class="tocSkip"> </h2>
# 
# <b>👋:</b> В такой цветовой ячейке я прошу тебя оставлять свои комментарии. Если исправляешь проект на второй итерации и выше, не забывай пожалуйста указывать номер итерации, например, "Комментарий студента v.2".
# </div> 
# 
# <br/>    
#     
# Увидев у тебя неточность, в первый раз я лишь укажу на ее наличие и дам тебе возможность самому найти и исправить ее. На реальной работе твой руководитель будет поступать также, и я пытаюсь подготовить тебя именно к работе датасаентистом. Но если ты пока не справишься с такой задачей - при следующей проверке я дам более точную подсказку!🤓

# # Определение возраста покупателей

# План работы:
# 1. Проанализируем данные
# 2. Выведем на экран и проанализируем данные относительно целевого признака.
# 3. Выведем на экран примеры данных
# 3. Обучим модель, добившись результатов требуемых в задании
# 4. Сделаем выводы относительно модели и ее обучения

# <div class="alert alert-warning">
#     <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации💡:</b>  
#     
# Я бы посоветовал тебе подробнее раскрыть план своей работы. Это поможет тебе расставлять акценты в выводах.

# <br/>    
# <div class="alert alert-info">
# <h2> Комментарий студента: Поправил <a class="tocSkip"> </h2>

# <div class="alert alert-success">
# <h2> Комментарий ревьюера 2 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b>

# ## Исследовательский анализ данных

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
pd.options.display.float_format = '{:,.2f}'.format
pd.options.display.max_columns = 200


labels=pd.read_csv('/datasets/faces/labels.csv')

labels.info()


# Выборка состоит из 7591 значения. Нулевых значений нет. Колличества вполне достаточно, чтобы модель не переобучалась.

# <div class="alert alert-success">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> 
#     
# Размер выборки выведен. Выборка относительно небольшая, но мы еще можем нарастить ее, применяя аугментации
# 

# In[45]:


grafik=plt.figure(figsize=(15,10))
plt.hist(labels['real_age'], bins=100)
plt.title('Гистограмма целевого признака', fontsize=20)
plt.xlabel('Возраст покупателей', fontsize=14)
plt.ylabel('Колличество покупателей данного возраста', fontsize=14)
plt.xticks(np.arange(0,100,5));


# <br/>
# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера <a class="tocSkip"></h2>
#     
# <b>На доработку🤔:</b> 
#     
# Гистограмма целевого признака выведена. Раз у нас 100 различных возрастов, то и корзин можно указать столько же.
#         
# Пики в районе "юбилейных" лет, они тут могут быть связаны с тем, что настоящий возраст был неизвестен тому, кто размечал датасет, и он попросту округлял его. Важно еще поработать над оформлением графика (должно быть название горизонтальной оси, вертикальной оси и самого графика)

# <br/>    
# <div class="alert alert-info">
# <h2> Комментарий студента: Поправил <a class="tocSkip"> </h2>

# <div class="alert alert-success">
# <h2> Комментарий ревьюера 2 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b>

# Из графика видно, что большинство покупателей от 20 до 40 лет.

# In[3]:


image_gen=ImageDataGenerator(rescale=1/255)
train_datagen = ImageDataGenerator(rescale=1./255)
train_gen_flow = train_datagen.flow_from_dataframe(
    dataframe=labels,
    directory='/datasets/faces/final_files/',
    x_col='file_name',
    y_col='real_age',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw',
    seed=12345)


# In[4]:


features, target=next(train_gen_flow)
fig=plt.figure(figsize=(15,15))
for i in range(15):
    fig.add_subplot(3,5,i+1)
    plt.imshow(features[i])
    plt.xticks([])
    plt.yticks([])
    plt.title(target[i])
    plt.tight_layout()


# <div class="alert alert-success">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b>
#     
# Примеры фотографий выведены. Отлично, что указан возраст у фотографий!

# В датасете представлены фото лиц людей разного качества и освещенности. Присутствуют чернобелые изображения. на некоторых изображениях люди изображены сильно в профиль (85, 21). Также много изоюражений с заваленным горизонтом. Стоит перевести все изображения в чернобелые и привести их к похожей яркости.

# <div class="alert alert-success">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b>
#     
# Анализ проведен корректно.
#         
# Добавлю, общая ориентация фотографий примерно одинакова - в некоторых фото их уже повернули за нас, чтобы выровнять положение лица, значит нам их не обязательно поворачивать, а начальный размер изображений отличается, в силу того, что визуально заметно различное качество фото, где-то есть черные рамки (они не несут полезной информации). В итоге, размер изображений мы приведем к единому перед обучением модели.
#     
# Советую изучить базовые возможности аугментаций в <a href="https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator">документации</a> или в <a href="https://towardsdatascience.com/machinex-image-data-augmentation-using-keras-b459ef87cd22">статье</a>.     
#  </div>

# ## Обучение модели

# Перенесите сюда код обучения модели и её результат вывода на экран.
# 
# 
# (Код в этом разделе запускается в отдельном GPU-тренажёре, поэтому оформлен не как ячейка с кодом, а как код в текстовой ячейке)

# ```python
# 
# from tensorflow.keras.datasets import fashion_mnist
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications.resnet import ResNet50
# import numpy as np
# import pandas as pd
# def load_train(path):
#     train_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.25)
#     train_datagen_flow = train_datagen.flow_from_dataframe(dataframe=pd.read_csv(path + 'labels.csv'),directory=path+'/final_files',x_col='file_name',y_col='real_age',target_size=(224, 224),batch_size=32,class_mode='raw',subset='training',seed=12345)
#     return train_datagen_flow
# def load_test(path):
#     test_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.25)
#     test_datagen_flow = test_datagen.flow_from_dataframe(dataframe=pd.read_csv(path + 'labels.csv'),directory=path + '/final_files',x_col = 'file_name',y_col = 'real_age',target_size=(224, 224),batch_size=32,class_mode='raw',subset='validation',seed=12345)
#     return test_datagen_flow
# def create_model(input_shape):
#     backbone = ResNet50(input_shape=input_shape,weights='imagenet',include_top=False)
#     model = Sequential()
#     model.add(backbone)
#     model.add(GlobalAveragePooling2D())
#     model.add(Dense(100,activation='relu'))
#     model.add(Dense(50,activation='relu'))
#     model.add(Dense(1, activation='relu')) 
#     model.compile(optimizer=Adam(lr=0.0005), loss='mean_squared_error',
#                   metrics=['mae'])    
#     return model
#  
# def train_model(model, train_data, test_data, batch_size=None, epochs=12,
#                 steps_per_epoch=None, validation_steps=None):
#     if steps_per_epoch is None:
#         steps_per_epoch = len(train_data)
#     if validation_steps is None:
#         validation_steps = len(test_data)
#     model.fit(train_data,
#               validation_data=test_data,
#               batch_size=batch_size, 
#               epochs=epochs,
#               steps_per_epoch=steps_per_epoch,
#               validation_steps=validation_steps,
#               verbose=2)  
#     return model
# 
# ```

# <div class="alert alert-warning">
#     <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации💡:</b> 
# 
# Параметру  `target_size` стоит задать значение `(224, 224)`, ведь на таком размере фото обучалась сеть в Imagenet.

# <br/>    
# <div class="alert alert-info">
# <h2> Комментарий студента: Поправил <a class="tocSkip"> </h2>

# <div class="alert alert-success">
# <h2> Комментарий ревьюера 2 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b>

# ```
# 
# 
# 
# <class 'tensorflow.python.keras.engine.sequential.Sequential'>
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to  
#   ['...']
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to  
#   ['...']
# Train for 178 steps, validate for 60 steps
# Epoch 1/9
# 2023-04-29 17:04:03.046162: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
# 2023-04-29 17:04:03.472971: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 178/178 - 51s - loss: 214.2589 - mae: 10.9255 - val_loss: 375.7927 - val_mae: 14.4708
# Epoch 2/9
# 178/178 - 38s - loss: 106.4361 - mae: 7.8679 - val_loss: 319.6808 - val_mae: 13.2533
# Epoch 3/9
# 178/178 - 38s - loss: 64.2137 - mae: 6.1307 - val_loss: 273.7220 - val_mae: 12.2426
# Epoch 4/9
# 178/178 - 40s - loss: 36.7952 - mae: 4.6812 - val_loss: 170.1239 - val_mae: 10.1482
# Epoch 5/9
# 178/178 - 41s - loss: 25.2825 - mae: 3.8973 - val_loss: 113.9959 - val_mae: 8.1981
# Epoch 6/9
# 178/178 - 41s - loss: 17.5004 - mae: 3.2019 - val_loss: 83.2082 - val_mae: 6.8998
# Epoch 7/9
# 178/178 - 41s - loss: 12.8877 - mae: 2.7342 - val_loss: 104.0604 - val_mae: 7.8281
# Epoch 8/9
# 178/178 - 40s - loss: 12.4986 - mae: 2.6856 - val_loss: 80.3759 - val_mae: 6.8338
# Epoch 9/9
# 178/178 - 39s - loss: 11.4967 - mae: 2.5549 - val_loss: 101.7260 - val_mae: 7.7654
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to  
#   ['...']
# 60/60 - 10s - loss: 101.7260 - mae: 7.7654
# Test MAE: 7.7654
# 
# ```

# <div class="alert alert-success">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> 
#     
# Отличный результат по метрике)

# ## Анализ обученной модели

# Модель обучилась за 9 эпох. Валидационная выборка при обучении была 0.25 от общей. В качестве функции потерь применена функция MSE и архитектура ResNet с четырьмя финализирующими слоями. Один усредняющий значения и 3 полносвязных по 100,50 и одному нейрону.

# <div class="alert alert-success">
# <h2> Комментарий ревьюера  <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Выводы вполне логичные)

# <div style="border:solid Chocolate 2px; padding: 40px">
# 
# 
# <h2> Итоговый комментарий ревьюера  <a class="tocSkip"> </h2>    
#     
#   
# У меня сложилось хорошее общее впечатление о проекте, тебе удалось неплохо справиться с проектом. Молодец! Загружены данные для анализа, изучен каждый параметр, модель построена и протестирована. Тебе удалось успешно справиться с первой задачей в области компьютерного зрения. Поздравляю!
#     
# Отмечу отдельные положительные моменты проекта🙂:
#     
# - радует, что проект хорошо структурирован, приятно проверять такие работы;
# - есть красивые визуализации в проекте.
#      
# Есть несколько моментов всего, на которые стоит ещё раз взглянуть, я указал их в моих комментариях по ходу проекта. Предлагаю тебе доработать проект по моим комментариям, чтобы довести его до совершенства.
# 
#     
#  
# </div>

# <div style="border:solid Chocolate 2px; padding: 40px">
# 
# 
# <h2> Итоговый комментарий ревьюера v.2 <a class="tocSkip"> </h2>    
#     
#      
#   
# Теперь почти идеально, молодец! Принимаю твой проект)
#     
# Могу поделиться такими ссылками для углубленного изучения темы спринта:
#     
# - Видеокурс от Stanford University: https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk
#     
# - Курс от Физтех-школы Прикладной Математики и Информатики МФТИ: https://vk.com/dlschool_mipt
#     
# - Курс Нейронные сети и компьютерное зрение от Sumsung: https://stepik.org/course/50352/promo#toc    
#     
#     
# В этом проекте все шаги отработаны. С интересом читал твои размышления, код и выводы. Что ещё для дальнейшего улучшения качества возможно сделать?
#     
# Опишу здесь некоторые идеи:
#     
# 1. Качество данных на входе: оценить еще раз возраст по фото.
# 2. Анализ ошибок: к примеру модель сильно ошибается на сегменте пожилых людей, зато почти идеально работает с детьми. Тогда мы поймем, каких возрастов фото было бы хорошо добавить в выборку для дообучения.
# 3. Дополнительно изучить распределение целевого признака, возможно что выборка была составлена неслучайным образом (видна небольшая бимодальность), посмотреть, каких возрастных групп представлено мало, чтобы их потом добавить в выборку.
#     
# Но цель нашего проекта достигнута. Рад был тебе помочь)
#     
# Удачи тебе в финальном проекте!
#   
# </div>

# ## Чек-лист

# - [x]  Jupyter Notebook открыт
# - [x]  Весь код выполняется без ошибок
# - [x]  Ячейки с кодом расположены в порядке исполнения
# - [x]  Исследовательский анализ данных выполнен
# - [x]  Результаты исследовательского анализа данных перенесены в финальную тетрадь
# - [x]  MAE модели не больше 8
# - [x]  Код обучения модели скопирован в финальную тетрадь
# - [x]  Результат вывода модели на экран перенесён в финальную тетрадь
# - [x]  По итогам обучения модели сделаны выводы

# In[ ]:




