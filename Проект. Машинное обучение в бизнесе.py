#!/usr/bin/env python
# coding: utf-8

# Привет! меня зовут Люман Аблаев. Сегодня я проверю твой проект.
# <br> Дальнейшее общение будет происходить на "ты" если это не вызывает никаких проблем.
# <br> Желательно реагировать на каждый мой комментарий ('исправил', 'не понятно как исправить ошибку', ...)
# <br> Пожалуйста, не удаляй комментарии ревьюера, так как они повышают качество повторного ревью.
# 
# Комментарии будут в <font color='green'>зеленой</font>, <font color='blue'>синей</font> или <font color='red'>красной</font> рамках:
# 
# 
# <div class="alert alert-block alert-success">
# <b>Успех:</b> Если все сделано отлично
# </div>
# 
# <div class="alert alert-block alert-info">
# <b>Совет: </b> Если можно немного улучшить
# </div>
# 
# <div class="alert alert-block alert-danger">
# <b>Ошибка:</b> Если требуются исправления. Работа не может быть принята с красными комментариями.
# </div>
# 
# -------------------
# 
# Будет очень хорошо, если ты будешь помечать свои действия следующим образом:
# <div class="alert alert-block alert-warning">
# <b>Комментарий студента:</b> ..
# </div>
# <div class="alert alert-block alert-warning">
# <b>Изменения:</b> Были внесены следующие изменения ...
# </div>
# 
# <font color='orange' style='font-size:24px; font-weight:bold'>Общее впечатление</font>
# * Спасибо за  качественную работу, мне она понравилась!
# - Я оставил некоторые советы надеюсь они будут полезными или интересными.
# - Есть небольшие недочеты, но думаю у тебя не составить труда их исправить
# - Из-за их совокупности они не может не повлияли на результат, но это за счет везения.
# - Жду обновленную работу!
# 
# 

# <font color='orange' style='font-size:24px; font-weight:bold'>Общее впечатление[3]</font>
# * Спасибо за оперативность
# - Приятно видеть фидбек на комментарии.
# - Критичекие недочеты исправлены 
# - Рад был быть полезен. Удачи в дальнейшем обучении!
# 
# 

# # Выбор локации для скважины

# Допустим, вы работаете в добывающей компании «ГлавРосГосНефть». Нужно решить, где бурить новую скважину.
# 
# Вам предоставлены пробы нефти в трёх регионах: в каждом 10 000 месторождений, где измерили качество нефти и объём её запасов. Постройте модель машинного обучения, которая поможет определить регион, где добыча принесёт наибольшую прибыль. Проанализируйте возможную прибыль и риски техникой *Bootstrap.*
# 
# Шаги для выбора локации:
# 
# - В избранном регионе ищут месторождения, для каждого определяют значения признаков;
# - Строят модель и оценивают объём запасов;
# - Выбирают месторождения с самым высокими оценками значений. Количество месторождений зависит от бюджета компании и стоимости разработки одной скважины;
# - Прибыль равна суммарной прибыли отобранных месторождений.

# ## Загрузка и подготовка данных

# In[1]:


get_ipython().system('pip install pandas_profiling')
import warnings
warnings.filterwarnings('ignore')

import pandas_profiling
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# <div class="alert alert-block alert-success">
# <b>Успех:</b> Импорты  на месте
# </div>
# 

# In[2]:


# Объединим датафреймы по соответствующим столбцам
data0 = pd.read_csv('/datasets/geo_data_0.csv')
data1 = pd.read_csv('/datasets/geo_data_1.csv')
data2 = pd.read_csv('/datasets/geo_data_2.csv')
frames = [data0, data1, data2]
data = pd.concat(frames, ignore_index = True)
data.head()


# Проверим есть ли пропуски или дубликаты в данных

# In[3]:


data.isna().sum()


# In[4]:


print('Количество дубликатов:', data.duplicated().sum())


# In[5]:


data.info()


# In[6]:


data.hist(bins=20, figsize=(15,20))


# Во всех графах видим нормальное распределение кроме product только у всех трех признаков разный масштаб. Позже приведем к одному.

# In[7]:


pandas_profiling.ProfileReport(data)


# <div class="alert alert-block alert-success">
# <b>Успех:</b> Данные осмотрены, но можно было бы еще как минимум изучить корреляции, так как мы планируем использовать линейные модели, а для них это важно. 
# </div>
# 
# 
# <div class="alert alert-block alert-info">
# <b>Совет:</b>  Советую обратить внимание на библиотеки sweetviz и pandas_profiling помогут в проведении более тщательного EDA анализа. Исследовательский анализ можно делать и с помощью ручного вызова функций дефолтных библиотек. Данные библиотеки хороши для максимизации комфорта презентации результатов анализа бизнес-пользователям.  
# 
# Очень просты в использоовании, на вход кладется датафрейм: pandas_profiling.ProfileReport(df) 
# </div>
# 

# 
# <div class="alert alert-block alert-warning">
# <b> Данная библиотека в онлайн версии не поддерживается судя по всему

# <div class="alert alert-block alert-success">
#     
# <b>Ответ[2]:</b> Ее нужно устаановить `!pip install ..`
# </div>

# ## Обучение и проверка модели

# In[8]:


# Разделим данные 3 регионов на целевой признак и тестовую выборку
features0 = data0.drop(['product', 'id'], axis=1)
target0 = data0['product']
features1 = data1.drop(['product', 'id'], axis=1)
target1 = data1['product']
features2 = data2.drop(['product', 'id'], axis=1)
target2 = data2['product']
features0


# 
# <div class="alert alert-block alert-info">
# <b>Совет:</b> Упс.... здесь небольшая ошибка, но она не критическая.<br>
#  Обучать scaler нужно только на трейне, инче это может привести к утечке данных (data leak).
# Про Data Leaks можно почитать подробнее по следующим ссылкам:
# 
# [Ссылка 1](https://towardsdatascience.com/preventing-data-leakage-in-your-machine-learning-model-9ae54b3cd1fb) <br>
# [Ссылка 2](https://machinelearningmastery.com/data-preparation-without-data-leakage/)
# 
# P.S. трансформировать нужно все выборки
# </div>
# 

# In[9]:


# Разделим данные на валидационную выборку и тренировочную
features0_train, features0_valid, target0_train, target0_valid = train_test_split(features0, target0, test_size=0.25, random_state=12345)

features1_train, features1_valid, target1_train, target1_valid = train_test_split(features1, target1, test_size=0.25, random_state=12345)

features2_train, features2_valid, target2_train, target2_valid = train_test_split(features2, target2, test_size=0.25, random_state=12345)


# In[10]:


# Приведем данные признаков к одному масштабу
scaler = StandardScaler()
scaler.fit(features0_train)
features0_train = scaler.transform(features0_train)
features0_valid = scaler.transform(features0_valid)
scaler.fit(features1_train)
features1_train = scaler.transform(features1_train)
features1_valid = scaler.transform(features1_valid)
scaler.fit(features2)
features2 = scaler.transform(features2_train)
features2 = scaler.transform(features2_valid)


# <div class="alert alert-block alert-warning">
# <b>Комментарий студента:</b> Исправил утечку данных
# </div>

# <div class="alert alert-block alert-info">
#     
# <b>Совет[2]:</b> Разве? `scaler.fit(features2)`
# </div>

# In[11]:


# Обучим модель для работы по первому региону
model0 = LinearRegression()
model0.fit(features0_train, target0_train)
pred0 = model0.predict(features0_valid)
rmse0 = mean_squared_error(target0_valid, pred0)**0.5
print('RMSE=',rmse0)
print('Средний запас предсказанного сырья:',pred0.mean())
# Обучим модель для работы по второму региону
model1 = LinearRegression()
model1.fit(features1_train, target1_train)
pred1 = model1.predict(features1_valid)
rmse1 = mean_squared_error(target1_valid, pred1)**0.5
print('RMSE=',rmse1)
print('Средний запас предсказанного сырья:',pred1.mean())
# Обучим модель для работы по третьему региону
model2 = LinearRegression()
model2.fit(features2_train, target2_train)
pred2 = model2.predict(features2_valid)
rmse2 = mean_squared_error(target2_valid, pred2)**0.5
print('RMSE=',rmse2)
print('Средний запас предсказанного сырья:',pred2.mean())


# Что интересно RMSE на втором регионе меньше 0, а па первом и третьем довольно большое

# <div class="alert alert-block alert-success">
# <b>Успех:</b> Шаг проделан хорошо, все метрики посчитаны корректно
# </div> 
# 
# 
# <div class="alert alert-block alert-info">
# <b>Совет </b>  Можно было бы использовать циклы/функции и избежать дублирования кода.
# </div>

# ## Подготовка к расчёту прибыли

# In[12]:


# присваиваем всем переменнным значения заданные в условиях
num_points = 500
best_points = 200
budget=10**10
barell = 450
product = 450000


# <div class="alert alert-block alert-info">
#     
# <b>Совет:</b> Константы принято записывать в ВЕРХНЕМ РЕГИСТРЕ =) Читем здесь https://pythonworld.ru/osnovy/pep-8-rukovodstvo-po-napisaniyu-koda-na-python.html#section-29
#     
# А большие значение можно записывать так: `450_000`.
# </div> 
# 

# In[13]:


#Стоимость разработки одной скважины:
well = budget/best_points
payback_cost = int(well/product)
print('Стоимость одной скважины:', well)
print('Прибыль при окупаемости одной скважины:', payback_cost)


# Стоимость окупаемости больше среднего значения окупаемости во всех трех регионах, поэтому стоит тщательнее отбирать скважины.

# <div class="alert alert-block alert-success">
# <b>Успех:</b> Значение для безубыточной разработки посчитано верно, с выводом согласен  
# </div>

# In[14]:


real_predict_values0 = pd.concat([target0_valid.reset_index(drop=True), pd.DataFrame(pred0)], axis=1).rename(columns={'product':'real', 0:'predictions'})
real_predict_values2 = pd.concat([target2_valid.reset_index(drop=True), pd.DataFrame(pred2)], axis=1).rename(columns={'product':'real', 0:'predictions'})
real_predict_values1 = pd.concat([target1_valid.reset_index(drop=True), pd.DataFrame(pred1)], axis=1).rename(columns={'product':'real', 0:'predictions'})


# ## Расчёт прибыли и рисков 

# #### Напишем функцию для расчета прибыли по выбранным скважинам и предсказаниям модели

# In[15]:


def profit_calc(target, pred, best_points=200, product=450000):
    pred_sorted = pred.sort_values(ascending=False)
    best_points_pred = pred_sorted[:best_points]
    best_points_true = target[best_points_pred.index.unique()]
    pred_profit = int(best_points_pred.sum()*product-budget)
    target_profit = int(best_points_true.sum()*product-budget)
    
    return target_profit, pred_profit


# <div class="alert alert-block alert-success">
# <b>Успех:</b> Ф-ция для подсчета прибыли определена верно
# </div>

# #### Расчитаем среднюю прибыль по регионам

# In[16]:


print('Средняя приыбль на основе реальных значений суммарной доходности лучших шахт в регионе №1, спрогнозированная и реальная:', profit_calc(real_predict_values0['real'], real_predict_values0['predictions'], best_points, product))
print('Средняя приыбль на основе реальных значений суммарной доходности лучших шахт в регионе №2, спрогнозированная и реальная:', profit_calc(real_predict_values1['real'], real_predict_values1['predictions'], best_points, product))
print('Средняя приыбль на основе реальных значений суммарной доходности лучших шахт в регионе №3, спрогнозированная и реальная:', profit_calc(real_predict_values2['real'], real_predict_values2['predictions'], best_points, product))


# ## Применим технику Bootstrap

# In[17]:


def bootstrap(target:pd.Series, pred:pd.Series, count=1000, num_points = 500):
    profit_values_real = []
    profit_values_pred = []
    state = np.random.RandomState(12345)
    for i in range(count):
        real_subsample = target.sample(n=num_points, replace=True, random_state=state)
        pred_subsample = pred[real_subsample.index]
        profit_real, profit_pred = profit_calc(real_subsample, pred_subsample,200,450000)
        profit_values_real.append(profit_real)
        profit_values_pred.append(profit_pred)
    profit_values_real = pd.Series(profit_values_real)
    profit_values_pred = pd.Series(profit_values_pred)
    return profit_values_real, profit_values_pred


# <div class="alert alert-block alert-success">
# <b>Успех:</b> С бутстрапом тоже все в порядке
# </div>
# 

# 
# 
# <div class="alert alert-block alert-danger">
#     
# <b>Ошибка:</b> Ты перепутал местами `profit_real, profit_pred` - но  более того нам нужно только profit_real
# </div>
# 

# <div class="alert alert-block alert-warning">
# <b>Комментарий студента:</b> не понял этого коментария
# </div>

# 
# 
# <div class="alert alert-block alert-danger">
#     
# <b>Ошибка[2]:</b> Посмотри, что и в каком порядке тебе возвращает функция прибыли, и как ты этот результат принимаешь 
# </div>
# 

# <div class="alert alert-block alert-warning">
# <b>Комментарий студента:</b> Понял, подправил
# </div>

# 
# <div class="alert alert-block alert-success">
# <b>Успех[3]:</b> Есть контакт
# </div>
# 

# Применение самой функции

# In[18]:


profit_values_real0, profit_values_pred0 = bootstrap(real_predict_values0['real'], real_predict_values0['predictions'], 1000, 500)
profit_values_real1, profit_values_pred1 = bootstrap(real_predict_values1['real'], real_predict_values1['predictions'], 1000, 500)
profit_values_real2, profit_values_pred2 = bootstrap(real_predict_values2['real'], real_predict_values2['predictions'], 1000, 500)


# Средняя прибыль после Bootstrap

# ### Доверительный интервал

# In[19]:


def confidence_int(df, alpha = 0.95):
    interval_values_low = df.quantile(0.025)
    interval_values_up = df.quantile(0.975)
    return int(interval_values_low), int(interval_values_up)


# In[20]:


print('95% доверительный интервал реальных значений суммарной прибыли для лучших скважин региона 1 равен:', confidence_int(profit_values_real0))
print('95% доверительный интервал предсказанных значений суммарной прибыли для лучших скважин региона 1 равен:', confidence_int(profit_values_pred0))
print()
print('95% доверительный интервал реальных значений суммарной прибыли для лучших скважин региона 2 равен:', confidence_int(profit_values_real1))
print('95% доверительный интервал предсказанных значений суммарной прибыли для лучших скважин региона 2 равен:', confidence_int(profit_values_pred1))
print()
print('95% доверительный интервал реальных значений суммарной прибыли для лучших скважин региона 3 равен:', confidence_int(profit_values_real2))
print('95% доверительный интервал предсказанных значений суммарной прибыли для лучших скважин региона 3 равен:', confidence_int(profit_values_pred2))
print()


# <div class="alert alert-block alert-success">
# <b>Успех:</b> Дов интервал считается верным образом!  
# </div>
# 

# ### Доверительный интервал

# In[21]:


def loss_prob(df):
    loss = (df < 0).mean() * 100
    return loss


# <div class="alert alert-block alert-info">
# <b>Совет: </b>
#     
# Риски можно лаконично считать в 1 строку: `(df < 0).mean() * 100`
# </div>                                                              
# 

# In[22]:


print('Вероятность убытка в регионе 1:', loss_prob(profit_values_real0))
print('Вероятность убытка в регионе 2:', loss_prob(profit_values_real1))
print('Вероятность убытка в регионе 3:', loss_prob(profit_values_real2))


# 
# 
# <div class="alert alert-block alert-danger">
# <b>Ошибка:</b> Нам нужно считать реальные прибыли ( отсортированные по предсказаниям)
# </div>
# 

# <div class="alert alert-block alert-success">
# <b>Успех[2]:</b> Осталось исправить верхний недочет
# </div>
# 

# ## Выводы

# Требования по вероятности убытка меньше чем 2,5% удовлетворяют только регион номер 2. Также он является лидером по прибыльности. Выбор этого региона для добычи будет правильным.

# 
# <div class="alert alert-block alert-success">
# <b>Успех[3]:</b> Отлично, проведено детальное исследование и был выбран правильный регион!
# </div>
# 
# 
# 
# <div class="alert alert-block alert-info">
# <b>Совет[3]: </b> Если интересно изучить тему бутстрапа глубже - я оставлю  ниже несколько интересных и полезных ссылок по теме:
#         
# 1. "Bootstrap Hypothesis Testing in Statistics with Example" 
#         
# https://www.youtube.com/watch?v=9STZ7MxkNVg
#         
# 2. "How to Calculate Bootstrap Confidence Intervals For Machine Learning Results in Python" 
#         
# https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/
#         
# 3. "A Gentle Introduction to the Bootstrap Method" 
# 
# https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/
#         
# 4. "An Introduction to the Bootstrap Method" 
#         
# https://towardsdatascience.com/an-introduction-to-the-bootstrap-method-58bcb51b4d60#:~:text=The%20basic%20idea%20of%20bootstrap,population%20mean)%20on%20sample%20data.&amp;text=A%20sample%20from%20population%20with%20sample%20size%20n
#         
# 5. "Bootstrapping and Resampling in Statistics with Example" 
#         
#         
# https://www.youtube.com/watch?v=O_Fj4q8lgmc
# </div>
# 

# <div class="alert alert-block alert-info">
# <b>Совет: </b>
#     
# Не забудь обновить вывод, если потребуется. И старайся его делать поподробнее. 
# </div>                                                              
# 

# ## Чек-лист готовности проекта

# Поставьте 'x' в выполненных пунктах. Далее нажмите Shift+Enter.

# - [x]  Jupyter Notebook открыт
# - [ ]  Весь код выполняется без ошибок
# - [ ]  Ячейки с кодом расположены в порядке исполнения
# - [ ]  Выполнен шаг 1: данные подготовлены
# - [ ]  Выполнен шаг 2: модели обучены и проверены
#     - [ ]  Данные корректно разбиты на обучающую и валидационную выборки
#     - [ ]  Модели обучены, предсказания сделаны
#     - [ ]  Предсказания и правильные ответы на валидационной выборке сохранены
#     - [ ]  На экране напечатаны результаты
#     - [ ]  Сделаны выводы
# - [ ]  Выполнен шаг 3: проведена подготовка к расчёту прибыли
#     - [ ]  Для всех ключевых значений созданы константы Python
#     - [ ]  Посчитано минимальное среднее количество продукта в месторождениях региона, достаточное для разработки
#     - [ ]  По предыдущему пункту сделаны выводы
#     - [ ]  Написана функция расчёта прибыли
# - [ ]  Выполнен шаг 4: посчитаны риски и прибыль
#     - [ ]  Проведена процедура *Bootstrap*
#     - [ ]  Все параметры бутстрепа соответствуют условию
#     - [ ]  Найдены все нужные величины
#     - [ ]  Предложен регион для разработки месторождения
#     - [ ]  Выбор региона обоснован

# In[ ]:




