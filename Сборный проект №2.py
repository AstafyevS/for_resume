#!/usr/bin/env python
# coding: utf-8

# <font color='blue' size=5><b>Комментарий ревьюера</b></font>
# 
# <font color='blue'>Привет, Сергей! Меня зовут Павел Григорьев, и я буду проверять этот проект.<br>Моя основная цель - не указать на совершённые тобой ошибки, а поделиться своим опытом и помочь тебе совершенствоваться как профессионалу.<br>Спасибо за проделанную работу! Предлагаю общаться на «ты».</font>
# <details>
# 	<summary><u>Инструкция по организационным мометнам (кликабельно)</u></summary>
# <font color='blue'>Я буду использовать различные цвета, чтобы было удобнее воспринимать мои комментарии:</font>
# 
# 
# ---
# 
# 
# <font color='blue'>синий текст - просто текст комментария</font>
# 
# <font color='green'>✔️ и зеленый текст - все отлично</font>
# 
# <font color='orange'>⚠️ и оранжевый текст - сделано все правильно, однако есть рекомендации, на что стоит обратить внимание</font>
# 
# <font color='red'>❌ и красный текст - есть недочеты</font>
# 
# 
# </details>    
#     </br>
# <font color='blue'>Пожалуйста, не удаляй мои комментарии в случае возврата работы, так будет проще разобраться, какие были недочеты, а также сразу увидеть исправленное. </font>
# 
# Ответы на мои комментарии лучше тоже помечать.
# Например: <font color='purple'><b>Комментарий студента</b></font>
# 
# <font color='blue'><b>Давай смотреть, что получилось!</b></font>

# <font color='blue' size=3><b>Комментарий ревьюера 2</b></font>
# 
# <font color='blue'>Привет еще раз. Спасибо, за исправления. Оформление комментариев по работе сохраняется. Только обозначим, что это вторая итерация.</font>

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Подготовка-данных" data-toc-modified-id="Подготовка-данных-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Подготовка данных</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Выясним-есть-ли-дубликаты-и-пропуски-в-данных" data-toc-modified-id="Выясним-есть-ли-дубликаты-и-пропуски-в-данных-1.0.1"><span class="toc-item-num">1.0.1&nbsp;&nbsp;</span>Выясним есть ли дубликаты и пропуски в данных</a></span></li><li><span><a href="#Проверим-правильность-расчета-эффективности-обогащения." data-toc-modified-id="Проверим-правильность-расчета-эффективности-обогащения.-1.0.2"><span class="toc-item-num">1.0.2&nbsp;&nbsp;</span>Проверим правильность расчета эффективности обогащения.</a></span><ul class="toc-item"><li><span><a href="#Напишем-функцию-для-вычисления-эффективности-обогащения" data-toc-modified-id="Напишем-функцию-для-вычисления-эффективности-обогащения-1.0.2.1"><span class="toc-item-num">1.0.2.1&nbsp;&nbsp;</span>Напишем функцию для вычисления эффективности обогащения</a></span></li></ul></li></ul></li><li><span><a href="#Признаки,-недоступные-в-тестовой-выборке" data-toc-modified-id="Признаки,-недоступные-в-тестовой-выборке-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Признаки, недоступные в тестовой выборке</a></span></li><li><span><a href="#Обработка-данных" data-toc-modified-id="Обработка-данных-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Обработка данных</a></span><ul class="toc-item"><li><span><a href="#Заполним-пропуски" data-toc-modified-id="Заполним-пропуски-1.2.1"><span class="toc-item-num">1.2.1&nbsp;&nbsp;</span>Заполним пропуски</a></span></li></ul></li><li><span><a href="#Признаки,-недоступные-в-тестовой-выборке" data-toc-modified-id="Признаки,-недоступные-в-тестовой-выборке-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Признаки, недоступные в тестовой выборке</a></span></li><li><span><a href="#Признаки,-недоступные-в-тестовой-выборке" data-toc-modified-id="Признаки,-недоступные-в-тестовой-выборке-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Признаки, недоступные в тестовой выборке</a></span></li></ul></li><li><span><a href="#Анализ-данных" data-toc-modified-id="Анализ-данных-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Анализ данных</a></span><ul class="toc-item"><li><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Ящик-с-усами" data-toc-modified-id="Ящик-с-усами-2.0.0.1"><span class="toc-item-num">2.0.0.1&nbsp;&nbsp;</span>Ящик с усами</a></span></li><li><span><a href="#Распределение-размеров-гранул-сырья-на-обучающей-и-тестовой-выборках" data-toc-modified-id="Распределение-размеров-гранул-сырья-на-обучающей-и-тестовой-выборках-2.0.0.2"><span class="toc-item-num">2.0.0.2&nbsp;&nbsp;</span>Распределение размеров гранул сырья на обучающей и тестовой выборках</a></span></li></ul></li><li><span><a href="#Суммарная-концентрация-всех-веществ-на-разных-стадиях:-(сырье,-черновой-и-финальный-концентраты)" data-toc-modified-id="Суммарная-концентрация-всех-веществ-на-разных-стадиях:-(сырье,-черновой-и-финальный-концентраты)-2.0.1"><span class="toc-item-num">2.0.1&nbsp;&nbsp;</span>Суммарная концентрация всех веществ на разных стадиях: (сырье, черновой и финальный концентраты)</a></span></li><li><span><a href="#С-выбросами" data-toc-modified-id="С-выбросами-2.0.2"><span class="toc-item-num">2.0.2&nbsp;&nbsp;</span>С выбросами</a></span></li><li><span><a href="#Без-выбросов" data-toc-modified-id="Без-выбросов-2.0.3"><span class="toc-item-num">2.0.3&nbsp;&nbsp;</span>Без выбросов</a></span></li><li><span><a href="#Приведение-к-единому-виду-(удаление-из-тренировочной-выборки-столбцов,-отсутствующих-в-тестовой-выборке)" data-toc-modified-id="Приведение-к-единому-виду-(удаление-из-тренировочной-выборки-столбцов,-отсутствующих-в-тестовой-выборке)-2.0.4"><span class="toc-item-num">2.0.4&nbsp;&nbsp;</span>Приведение к единому виду (удаление из тренировочной выборки столбцов, отсутствующих в тестовой выборке)</a></span></li></ul></li></ul></li><li><span><a href="#Модель" data-toc-modified-id="Модель-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Модель</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Подготовка-параметров-для-прогнозирования-rougher.output.recovery" data-toc-modified-id="Подготовка-параметров-для-прогнозирования-rougher.output.recovery-3.0.1"><span class="toc-item-num">3.0.1&nbsp;&nbsp;</span>Подготовка параметров для прогнозирования rougher.output.recovery</a></span></li></ul></li><li><span><a href="#Подготовка-параметров-для-прогнозирования-final.output.recovery" data-toc-modified-id="Подготовка-параметров-для-прогнозирования-final.output.recovery-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Подготовка параметров для прогнозирования final.output.recovery</a></span><ul class="toc-item"><li><span><a href="#Без-выбросов" data-toc-modified-id="Без-выбросов-3.1.1"><span class="toc-item-num">3.1.1&nbsp;&nbsp;</span>Без выбросов</a></span></li><li><span><a href="#Прогноз-эффективности-обогащения-чернового-концентрата-(rougher.output.recovery)" data-toc-modified-id="Прогноз-эффективности-обогащения-чернового-концентрата-(rougher.output.recovery)-3.1.2"><span class="toc-item-num">3.1.2&nbsp;&nbsp;</span>Прогноз эффективности обогащения чернового концентрата (rougher.output.recovery)</a></span></li><li><span><a href="#Прогноз-эффективности-обогащения-финального-концентрата-(final.output.recovery)" data-toc-modified-id="Прогноз-эффективности-обогащения-финального-концентрата-(final.output.recovery)-3.1.3"><span class="toc-item-num">3.1.3&nbsp;&nbsp;</span>Прогноз эффективности обогащения финального концентрата (final.output.recovery)</a></span></li></ul></li><li><span><a href="#Случайный-лес" data-toc-modified-id="Случайный-лес-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Случайный лес</a></span><ul class="toc-item"><li><span><a href="#Прогноз-эффективности-обогащения-чернового-концентрата-(rougher.output.recovery)" data-toc-modified-id="Прогноз-эффективности-обогащения-чернового-концентрата-(rougher.output.recovery)-3.2.1"><span class="toc-item-num">3.2.1&nbsp;&nbsp;</span>Прогноз эффективности обогащения чернового концентрата (rougher.output.recovery)</a></span></li><li><span><a href="#Прогноз-эффективности-обогащения-финального-концентрата-(final.output.recovery)" data-toc-modified-id="Прогноз-эффективности-обогащения-финального-концентрата-(final.output.recovery)-3.2.2"><span class="toc-item-num">3.2.2&nbsp;&nbsp;</span>Прогноз эффективности обогащения финального концентрата (final.output.recovery)</a></span></li></ul></li><li><span><a href="#Итоговые-расчеты" data-toc-modified-id="Итоговые-расчеты-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Итоговые расчеты</a></span></li></ul></li><li><span><a href="#Линейная-регрессия" data-toc-modified-id="Линейная-регрессия-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Линейная регрессия</a></span><ul class="toc-item"><li><span><a href="#На-исходной-выборке" data-toc-modified-id="На-исходной-выборке-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>На исходной выборке</a></span><ul class="toc-item"><li><span><a href="#На-тестовой-выборке" data-toc-modified-id="На-тестовой-выборке-4.1.1"><span class="toc-item-num">4.1.1&nbsp;&nbsp;</span>На тестовой выборке</a></span></li></ul></li></ul></li><li><span><a href="#Чек-лист-готовности-проекта" data-toc-modified-id="Чек-лист-готовности-проекта-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Чек-лист готовности проекта</a></span></li></ul></div>

# # Восстановление золота из руды

# Подготовьте прототип модели машинного обучения для «Цифры». Компания разрабатывает решения для эффективной работы промышленных предприятий.
# 
# Модель должна предсказать коэффициент восстановления золота из золотосодержащей руды. Используйте данные с параметрами добычи и очистки. 
# 
# Модель поможет оптимизировать производство, чтобы не запускать предприятие с убыточными характеристиками.
# 
# Вам нужно:
# 
# 1. Подготовить данные;
# 2. Провести исследовательский анализ данных;
# 3. Построить и обучить модель.
# 
# Чтобы выполнить проект, обращайтесь к библиотекам *pandas*, *matplotlib* и *sklearn.* Вам поможет их документация.

# # Пожалуйста проверьте как можно скорее сегодня жесткий дедлайн! Спасибо!

# ## Подготовка данных
%autosave 20
# In[1]:


get_ipython().system('pip install -q optuna')


# In[2]:


import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.dummy import  DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import optuna
from sklearn.metrics import make_scorer


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'> Хорошее оформление импортов! \
# Импорты собраны в одной ячейке, сгрупированы по функциональности. Можно ещё разделить группы пустой строкой.</font>

# In[3]:


df_full = pd.read_csv('/datasets/gold_recovery_full_new.csv')
df_test = pd.read_csv('/datasets/gold_recovery_test_new.csv')
df_train = pd.read_csv('/datasets/gold_recovery_train_new.csv')


# In[4]:


df_full.head(10)


# Выведем общий анализ данных

# In[5]:


df_full.info()


# #### Выясним есть ли дубликаты и пропуски в данных

# In[6]:


df_full.isna().sum()


# In[7]:


df_full.duplicated().sum()


# In[8]:


df_full


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>Данные загружены корректно, первичный осмотр проведен.</font>

# #### Проверим правильность расчета эффективности обогащения.

# ##### Напишем функцию для вычисления эффективности обогащения

# In[9]:


def recovery_calc(row):
    c=row['rougher.output.concentrate_au']
    t=row['rougher.output.tail_au']
    f=row['rougher.input.feed_au']
    result = ((c*(f-t))/(f*(c-t)))*100
    return result
     


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>Расчёт верный!</font>

# In[10]:


df_train['recovery_calc']=df_train.apply(recovery_calc, axis=1)
df_train[['recovery_calc','rougher.output.recovery']]


# Визуально данные очень похожи, но все же вычислим MAE

# In[11]:


MAE = mean_absolute_error(df_train['recovery_calc'], df_train['rougher.output.recovery'])
MAE


# Вычисления очень точные

# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>👍</font>

# In[12]:


df_train = df_train.drop(columns='recovery_calc')


# Недоступны:
# 
# 1 оба целевых признака (final.output.recovery, final.output.recovery)
# 2 расчётные характеристики параметров сырья (rougher.calculation...)
# 3 параметры продукта флотации (rougher.output.concentrate_..)
# 4 параметры продукта первичной очистки (primary_cleaner.output.concentrate_..)
# 5 параметры продукта вторичной очистки (final.output.concentrate_..)
# 6 отвальные хвосты флотации (rougher.output.tail_..)
# 7 отвальные хвосты первичной очистки (primary_cleaner.output.tail_...)
# 8 отвальные хвосты вторичной очистки (secondary_cleaner.output.tail_)
# 9 отвальные хвосты финальные (общие) (final.output.tail_..)
# иными словами в тестовой выборке отсутствуют
# 
# параметры продукта после каждой стадии
# отвальные хвосты после каждой стадии, в т.ч. общие
# расчетные харакетристики сырья
# целевые параметры процесса
# а присутствуют:
# 
# все входные пармстры для флотации и первичной очистки (input)
# параметры состояния на каждом этапе (state)
# 
# вывод:
# 
# необходимо в тестовую выборку добавить оба целевых признака

# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'> Верно!</font>

# ### Признаки, недоступные в тестовой выборке

# In[13]:


train_params = df_train.columns
test_params = df_test.columns
diff_train_test_params = sorted(list(set(train_params) - set(test_params) ))
print('количество столбцов в тренировочной выборке = ', len(train_params))
print('количество столбцов в тестовой выборке = ', len(test_params))
print('список признаков, недоступных в тествой выборке:\n', diff_train_test_params)


# ### Обработка данных

# In[14]:


# изменим тип
df_full['date'] = pd.to_datetime(df_full['date'], format='%Y-%m-%d %H:%M:%S')
df_test['date'] = pd.to_datetime(df_test['date'], format='%Y-%m-%d %H:%M:%S')
df_train['date'] = pd.to_datetime(df_train['date'], format='%Y-%m-%d %H:%M:%S')

# сделаем индексом
df_full.set_index('date', inplace=True)
df_test.set_index('date', inplace=True)
df_train.set_index('date', inplace=True)


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'> Верно, нужно привязаться ко времени.</font>

# #### Заполним пропуски

# In[15]:


print('в выборках присутствуют короткие промежутки отсуствующих значений')
display(df_full.loc[( ('2017-12-17 05:00:00' < df_full.index) & (df_full.index < '2017-12-17 15:00:00')), 'rougher.input.feed_sol'])

print('так же присутствуют средне долгие промежутки отсуствующих значений')
display(df_full.loc[(('2017-12-16 02:00:00' < df_full.index) & (df_full.index < '2017-12-17 12:00:00') ), 'rougher.input.floatbank11_xanthate'])

print('и совсем длинные периоды')
display(df_full.loc[( ('2017-02-27 17:59:59' < df_full.index) & (df_full.index < '2017-03-12 10:59:59')), 'rougher.input.floatbank11_xanthate'])


# Можно заметить, что значения в предыдущих и последующих записях отличаются незначительно. Поэтому для заполненжия пропусков применим заполенение средним по n значениям. С учетом того, что встречаются разрывы различной продолжительности будем заполнеять средним по возрастанию. Напишем для этого функцию.

# In[16]:


def empty_values_replacer(df, empty_values_range=10):
    """
    Данная функция заполняет пропуски средним окна из n значений. Размер окна перебирается итеративно по возрастанию.
    """
    col_list = df.columns
    for i in range(2, empty_values_range):
        for col in col_list:
            try:
                df.loc[df[col].isna(), col] = df[col].rolling(f'{i}h').mean()[df[df[col].isna()].index].copy()
            except Exception as e:
                print(f'column = {col}, exception = {e}')
    return df


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>Классная функция!</font>

# <font color='blue'><b>Комментарий ревьюера: </b></font> ⚠️\
# <font color='darkorange'> В Пандас есть метод ffill - заполнение пропусков предыдущим известным значением.</font>

# In[17]:


df_full = empty_values_replacer(df_full, 25)


# In[18]:


df_full.info()


# Как видно, пропуски уменьшились, но все еще присутствуют. Заполним пропуски в тестовой выбоке. Но сперва убедимся что тестовая вобрка (df_test) является частью исходных данных (df_full)
# 
# 

# In[19]:


# объединим тестовую и полную выборку чтобы сравнить значения
df_full_train_joined =  df_full.join(df_train, lsuffix='_full', rsuffix='_train')


# In[20]:


print('в тренировочной выборке нет соответсвющих полной выборке записей в количестве ', len(df_full_train_joined.loc[df_full_train_joined['rougher.output.recovery_train'].isna(),['rougher.output.recovery_full', 'rougher.output.recovery_train']]),'шт.')
print('при этом')
print('в полной вборке выборке "не хватает" ', len(df_full_train_joined.loc[df_full_train_joined['rougher.output.recovery_full'].isna(),['rougher.output.recovery_full', 'rougher.output.recovery_train']]),'записей')
print('при этом')
print('для непустых записей из тренировочной выборки не совпадает' , len(df_full_train_joined.loc[(~df_full_train_joined['rougher.output.recovery_train'].isna() &
                                                                                                   (df_full_train_joined['rougher.output.recovery_full']!=df_full_train_joined['rougher.output.recovery_train']))
,['rougher.output.recovery_full', 'rougher.output.recovery_train']]), 'записей')
print('и это')
display(df_full_train_joined.loc[(~df_full_train_joined['rougher.output.recovery_train'].isna() &
                                  (df_full_train_joined['rougher.output.recovery_full']!=df_full_train_joined['rougher.output.recovery_train']))
,['rougher.output.recovery_full', 'rougher.output.recovery_train']]  )
print('что по сути одно и тоже, и разница, видимо, в долях')


# Получается, полная выбока перекрывает собой тренировочную и тестовую выбоки. А, поскольку, пропуски (хоть и частично) в df_full мы уже заполнили, затянем их в тестовую выборку. Подготовим для этого функцию.

# <font color='blue'><b>Комментарий ревьюера: </b></font> ⚠️\
# <font color='darkorange'> Пропуски в выборках то же совпадают.</font>

# In[21]:


def value_replacer(df1, df2):
    """
    данная функция по каждоиму столбцу датафрейма df1 заполняет в нем пропуски значениями из df2.
    если пропуски в обоих - заполняет пропусками
    """
    col_list = df1.columns
    for col in col_list:
        try:
            df1.loc[df1[col].isna(), col] = df2.loc[df1[col].isna().index,col]
        except Exception as e:
            print(e)
    return df1


# In[22]:


df_test = value_replacer(df_test, df_full)
df_train = value_replacer(df_train, df_full)


# In[23]:


display(df_train.info())
display(df_test.info())


# Как видно, от пропусков в тестовом датасете избавились, а вот в тренировочном они еще остались. Удалим строки с пропусками из тренировочного датасета.

# In[24]:


df_train = df_train.dropna()
df_train.info()


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>👍</font>

# ### Признаки, недоступные в тестовой выборке

# ### Признаки, недоступные в тестовой выборке

# Было 14149 записей, стало - 13897. 152 записи удалено. На фоне 14 тыс это порядка 1%. Уменьшение незначительное.
# 
# Подытог: в тренировочной и тестовой выборках от пропусков избавислись.

# ## Анализ данных

# Выведем колличество Au, Ag, Pb на разных этапах очистки

# In[25]:


phase_en = ['rougher.output.','primary_cleaner.output.','final.output.','rougher.input.feed']
phase_ru = ['флотация', 'первичная очистка', 'финальные характеристики', 'исходное сырье']
phase_en_ru = list(zip(phase_en, phase_ru))
concentrates = [ 'concentrate_au', 'concentrate_ag', 'concentrate_pb', 'concentrate_sol']
titles =  ['Aurum', 'Argentum', 'Plumbum', 'Sol' ]


# In[35]:


def barplot_concentration_printer(phase:list, concentrates:list,  titles:list, suptitle:str):
    """
    Функция отображает горизонтальные графики типа barplot для заданного элемента на трех стадиях очистки
    """
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15,5))
    fig.suptitle(suptitle, fontsize = 15)
    for concentrate, title, ax in zip(concentrates, titles, axes.ravel()):
        df_train[[phase[0][0] + concentrate, phase_en_ru[1][0] + concentrate, phase_en_ru[2][0] + concentrate, phase_en_ru[3][0] + concentrate]]            .rename(columns={phase_en_ru[0][0]  + concentrate:phase_en_ru[0][1],phase_en_ru[1][0]  + concentrate:phase_en_ru[1][1],phase_en_ru[2][0] + concentrate:phase_en_ru[3][1],phase_en_ru[3][0] + concentrate:phase_en_ru[2][1] }).mean().plot(kind='bar', ax=ax, title = title)
    plt.show()

barplot_concentration_printer(phase_en_ru, concentrates, titles, 'Средняя концентрация на различных этапах очистки')


# Среднее значение повторяет медианное (что очевидно, поскольку распрпделения тяготеют к нормальной форме) и дополнительной информации при анализе изменений не дают.

# Выводы:
# 
# 1. Концентрация золота возрастает более чем в 2 раза
# 2. Концентрация серебра падает более чем в 2 раза
# 3. Концентрация свинца так незначительно растет (примерно на четверть)
# 4. Концентрация солей падает примерно в 3 раза.

# <font color='blue'><b>Комментарий ревьюера: </b></font> ⚠️\
# <font color='darkorange'> Можно ещё посмотреть начальные концентрации `rougher.input.feed_{металл}`</font>

# ##### Ящик с усами

# Изучим распределения через диаграмму "ящик сусами". Возможно появятся подбробонсти

# In[27]:


def boxplot_concentration_printer(df:pd.DataFrame, phase:list, concentrate:str,  title:str):
    """
    Функция отображает вертикальные графики типа boxplot для заданного элемента на трех стадиях очистки
    """
    fig, axes = plt.subplots(nrows=len(phase), ncols=1, figsize=(14,5))
    fig.suptitle(title, fontsize = 15)
    for i , ax in zip(phase, axes.ravel()):
        df.rename(columns={str(i[0]+concentrate):str(i[1])}).boxplot(column=[str(i[1])], vert=False, ax=ax)
    plt.show()


# In[29]:


boxplot_concentration_printer(df_train, phase_en_ru, 'concentrate_au', 'Распределение концентарции на различных этапах очистки. Золото')


# In[ ]:


boxplot_concentration_printer(df_train, phase_en_ru, 'concentrate_ag', 'Реаспределение концентарции на различных этапах очистки. Серебро')


# In[ ]:


boxplot_concentration_printer(df_train, phase_en_ru, 'concentrate_pb', 'Реаспределение концентарции на различных этапах очистки. Свинец')


# Выводы:
# 
# 1. Концентрация золота возрастает
# 2. Концентрация серебра падает
# 3. Концентрация свинца так незначительно растет
# 4. Распределения носят нормальный характер без ярко-выраженных скосов с примерно симметричным количеством выбросов, как следствие медианное значение близко к ранее найденному среднему
# тем не менее выбросы диапазон выбросов сверху распрпделения становится все короче, а снизу - все длиннее. Выглядит как-будто концентрация достигает своего максимума.

# Выбросы диапазон выбросов сверху распрпделения становится все короче, а снизу - все длиннее. Выглядит как-будто концентрация достигает своего максимума.

# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>👍</font>

# ##### Распределение размеров гранул сырья на обучающей и тестовой выборках

# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20,8))
fig.suptitle('Реаспределение размеров гранул сырья на различных этапах очистки. Золото', fontsize = 15)

df_test.rename(columns={'rougher.input.feed_size':'размер гранул сырья'}).boxplot(column=['размер гранул сырья'], vert=False, ax=axes[0])
df_train.rename(columns={'rougher.input.feed_size':'размер гранул сырья'}).boxplot(column=['размер гранул сырья'], vert=False, ax=axes[1])
axes[0].set_xlim(0, 500)
axes[1].set_xlim(0, 500)
axes[0].title.set_text('Тестовая выборка')
axes[1].title.set_text('Тренировочная выборка')


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,6))
fig.suptitle('Реаспределение концентарции на различных этапах очистки. Золото', fontsize = 15)

df_test['rougher.input.feed_size'].rename('размер гранул сырья').plot(kind = 'hist', ax=axes[0], bins = 100, grid=True, density=True)
df_train['rougher.input.feed_size'].rename('размер гранул сырья').plot(kind = 'hist', ax=axes[1], bins = 100, grid=True, density=True)
axes[0].set_xlim(0, 500)
axes[1].set_xlim(0, 500)
axes[0].title.set_text('Тестовая выборка')
axes[1].title.set_text('Тренировочная выборка')


# <font color='blue'><b>Комментарий ревьюера: </b></font> ⚠️\
# <font color='darkorange'> В данном случае нужно использовать параметр "density=True" в методе ".hist()". Так бы нам удалось перейти от количества к доле.
# </font>

# In[ ]:


print('Средний размер гранул сырья на тренировочной выборке =', df_train['rougher.input.feed_size'].mean())
print('Средний размер гранул сырья на тестовой выборке =', df_test['rougher.input.feed_size'].mean())
print('Относительная разница в размере сосатвляет', round((df_train['rougher.input.feed_size'].mean()-df_test['rougher.input.feed_size'].mean())/df_train['rougher.input.feed_size'].mean()*100, 2), '%')


# На гистограмме заметно что:
# 
# 1. Тестовая выборка имеет более явное смещение вправо, в то время как тренировочная выглядит более нормально
# 2. Медианное значение так же на гистограме больше на тренировочной выборке чем на тестовой

# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'><s> Нужен вывод о сравнении распределений переменных.</s></font>

# <font color='purple'><b>Добавил</b></font>

# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ⚠️\
# <font color='darkorange'> Нужен вывод похожи распределения или нет. "Если распределения сильно отличаются друг от друга, оценка модели будет неправильной."</font>

# #### Суммарная концентрация всех веществ на разных стадиях: (сырье, черновой и финальный концентраты)

# In[ ]:


phase_en = ['rougher.input.feed_','rougher.output.concentrate_','final.output.concentrate_']
phase_ru = ['сырье', 'первичная очистка', 'финальная очистка']
phase_en_ru = list(zip(phase_en, phase_ru))
concentrates = [ 'au', 'ag', 'pb', 'sol']
titles =  ['Aurum', 'Argentum', 'Plumbum', 'Sol' ]


# In[ ]:


def concentrate_printer_4elem(df, descs, concs, phases):
    for desc, phase in zip(descs, phases) :
        print(desc, round(df[[phase + concs[0], phase + concs[1], phase + concs[2], phase + concs[3]]].sum().sum(), 2))

def concentrate_printer_3elem(df, descs, concs, phases):
    for desc, phase in zip(descs, phases) :
        print(desc, round(df[[phase + concs[0], phase + concs[1], phase + concs[2]]].sum().sum(), 2))


# In[ ]:


descriptions = ['суммарная концентрация веществ на стадии флотации =', 'суммарная концентрация веществ на стадии первичной очистки =', 'суммарная концентрация веществ финального продукта =']
concentrates_4elem = [ 'au', 'ag', 'pb', 'sol']
concentrates_3elem = [ 'au', 'ag', 'pb']


# #### С выбросами

# In[ ]:


print('с учетом солей:')
print('суммарная концентрация веществ на стадии флотации =', df_train[['rougher.input.feed_au', 'rougher.input.feed_ag', 'rougher.input.feed_pb', 'rougher.input.feed_sol']].sum().sum())
print('суммарная концентрация веществ на стадии первичной очистки =', df_train[['rougher.output.concentrate_au', 'rougher.output.concentrate_ag', 'rougher.output.concentrate_pb', 'rougher.output.concentrate_sol']].sum().sum())
print('суммарная концентрация веществ на финального продукта =', df_full[['final.output.concentrate_au', 'final.output.concentrate_ag', 'final.output.concentrate_pb', 'final.output.concentrate_sol']].sum().sum())


# In[ ]:


float_conc = ['rougher.input.feed_au', 'rougher.input.feed_ag', 'rougher.input.feed_pb', 'rougher.input.feed_sol']
first_conc = ['rougher.output.concentrate_au', 'rougher.output.concentrate_ag', 'rougher.output.concentrate_pb', 'rougher.output.concentrate_sol']
final_conc = ['final.output.concentrate_au', 'final.output.concentrate_ag', 'final.output.concentrate_pb', 'final.output.concentrate_sol']
float_conc_wosol = ['rougher.input.feed_au', 'rougher.input.feed_ag', 'rougher.input.feed_pb']
first_conc_wosol = ['rougher.output.concentrate_au', 'rougher.output.concentrate_ag', 'rougher.output.concentrate_pb']
final_conc_wosol = ['final.output.concentrate_au', 'final.output.concentrate_ag']


# In[ ]:


print('без учета солей:')
print('суммарная концентрация веществ на стадии флотации =', df_full[['rougher.input.feed_au', 'rougher.input.feed_ag', 'rougher.input.feed_pb']].sum().sum())
print('суммарная концентрация веществ на стадии первичной очистки =', df_full[['rougher.output.concentrate_au', 'rougher.output.concentrate_ag', 'rougher.output.concentrate_pb']].sum().sum())
print('суммарная концентрация веществ на финального продукта =', df_full[['final.output.concentrate_au', 'final.output.concentrate_ag', 'final.output.concentrate_pb']].sum().sum())


# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'><s>В задании требуется следующее:<br>
# Исследуйте суммарную концентрацию всех веществ на разных стадиях: в сырье, в черновом и финальном концентратах.<br>
# в сырье — это столбцы rougher.input.feed_<br>
# в черновом концентрате  - это столбцы rougher.output.concentrate_<br>
#     в финальном концентрате - это столбцы final.output.concentrate_</s></font>

# <font color='purple'><b>Комментарий студента: Поправил</b></font>

# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ✔️\
# <font color='green'>👍</font>

# #### Без выбросов

# <font color='blue'><b>Комментарий ревьюера: </b></font> ⚠️\
# <font color='darkorange'> Достоверными выбросами можно считать только нулевые значения суммарных концентраций, поскольку они объяснимы. Распределения имеют неизвестный характер и полагаться на квантили ненадёжно.</font>

# In[ ]:


def outliers_cutter(df, phases, concs, koef=1.5):
    """
    Данная функция для каждого значения концентрации на каждой из стадий делает срез по диапазону Q1-1.5*IQR - Q3+1.5*IQR
    """
    df = df.copy()
    for phase in phases:
        for conc in concs:

            Q1 = df[phase + conc].quantile(0.25)
            Q3 = df[phase + conc].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[phase + conc] > Q1-koef*IQR ) & (df[phase + conc] < Q3+koef*IQR)]

    return df


# In[ ]:


df_train_cuted = outliers_cutter(df_train, phase_en, concentrates)


# In[ ]:


concentrate_printer_4elem(df_train_cuted, descriptions, concentrates, phase_en)
print()
concentrate_printer_3elem(df_train_cuted, descriptions, concentrates_3elem, phase_en)


# Разница существенная

# In[ ]:


boxplot_concentration_printer(df_train, phase_en_ru, 'au', 'Реаспределение концентарции на различных этапах очистки. Золото')


# In[ ]:


boxplot_concentration_printer(df_train_cuted, phase_en_ru, 'au', 'Реаспределение концентарции на различных этапах очистки. Золото')


# Длинные хвосты выбросов стали заметно короче.

# Выводы:
# 
# 1.Не учитывая соли видно, что суммарная концентрация металлов выросла примерно на 60%
# 2.Учитывая соли видно, что суммарная концентрация на этапе первичной очистки проседает, затем приходит в прежнее состояние. (вероятно за счет повышения концентрации металлов)
# 3.По рассчитанным значениям средней концентрациям на разлиных стадиях видно что концентрация Золота вырастает примерно в два раза
# 4.Очистка выборки от выбросов существенно изменяет значения концентрации в т.ч.и суммарные
# 
# Очистим тренировочную выборку от выбросов и будем обучать модели без них

# In[ ]:


# сохраним изначальную версию тренировочной выборки в df_train_init
df_train_init = df_train.copy()
# и отфильтруем записи с выбросоми
df_train = outliers_cutter(df_train, phase_en, concentrates, koef=1.5)


# In[ ]:


params_to_cut = []
for p in phase_en:
    for c in concentrates:
        params_to_cut.append(p+c)
print('пересечение между множеством столбцов тестовой выборки и параметров на "стрижку выбросов" составляет:', set(params_to_cut) & set(df_test.columns))
print('т.е. пересекающихся значений' )
print('(просто иллюстрация для проверки) а вот для множеств 1,2,3 и 3,4,5 общим является:', set([1,2,3]) & set([3,4,5]))


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'> Выбросы удалены.</font>

# #### Приведение к единому виду (удаление из тренировочной выборки столбцов, отсутствующих в тестовой выборке)

# В соответствии с описанием задания и п.1.3. тестовая выборка не содержит ряд параметров, в т.ч. целевых признаков. Так же сказано, что "Некоторые параметры недоступны, потому что замеряются и/или рассчитываются значительно позже.". К сожалению, в задании не сказано какой набор данных будет поступать в "боевых" условиях. Поэтому считаем, что они будут соответствовать параметрам из тестового датасета. В связи с чем обогащать его параметрами из полного набора не будем, т.к. он получен, видимо, позже. Для того чтобы модель можно было обучить необходимо привести в соответствие набр параметров в тестовой и тренировочной выборках. В связи с чем удалим из тренировочной выборки параметры, отсутствующие в тестовой.

# In[ ]:


# для обеспечения валидации модели добавим в тестовую выборку целевые параметры из полной выборки
df_test['rougher.output.recovery'] = df_full.loc[df_test.index,'rougher.output.recovery']
df_test['final.output.recovery'] = df_full.loc[df_test.index,'final.output.recovery']

# подготовим список параметров для обоих выборок
train_params = df_train.columns
test_params = df_test.columns
# получим параметры, отсутствующие в тестовой
params_to_drop_from_test = sorted(list(set(train_params) - set(test_params) ))


# Воспользуемся ранее подготовленным списком разности столбцов между тренировочной и тестовой выборкой - diff_train_test_params

# In[ ]:


df_train = df_train.drop(columns=params_to_drop_from_test)
df_train_init = df_train_init.drop(columns=params_to_drop_from_test)


# Проверим соответствие множеств столбцов в тестовой и тренировочной выборках

# In[ ]:


print('столбцы в тренировочной и тестовой выборках соответствую друг другу = ', set(df_train.columns) == set(df_test.columns))
print('столбцы в изначальной тренировочной и тестовой выборках соответствую друг другу = ', set(df_train_init.columns) == set(df_test.columns))


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>👍</font>

# ## Модель

# In[ ]:


def smape(true:pd.Series, predict:np.array):
    """
    Данная функция считает метрику sMAPE на основании двух векторов

    """

    res = (1/len(true))*np.sum((np.abs(true - predict))/((np.abs(true)+np.abs(predict))/2))*100
    return res

def overal_smape(smape_rougher, smape_final):
    """
    Данная функция рассчитывает итоговое значение sMAPE
    """
    return abs(smape_rougher*0.25 + smape_final*0.75)


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'> `smape` и `overal_smape` расчитаны верно.</font>

# #### Подготовка параметров для прогнозирования rougher.output.recovery

# Предположительно, эффективность обогащения чернового концентрата (rougher.output.recovery) имеет смысл прогнозировать на этапе флотации, т.к. на последующих этапах мы уже ее по идее знаем. Исходя из этого составим список параметров для обучения модели и целевой парметр.

# In[ ]:


rougher_params_list = ['rougher.state.floatbank10_a_air', 'rougher.state.floatbank10_b_air',
                       'rougher.state.floatbank10_c_air', 'rougher.state.floatbank10_d_air',
                       'rougher.state.floatbank10_e_air', 'rougher.state.floatbank10_f_air',
                       'rougher.state.floatbank10_a_level', 'rougher.state.floatbank10_b_level',
                       'rougher.state.floatbank10_c_level', 'rougher.state.floatbank10_d_level',
                       'rougher.state.floatbank10_e_level', 'rougher.state.floatbank10_f_level']

rougher_target = 'rougher.output.recovery'


# <font color='blue'><b>Комментарий ревьюера: </b></font> ⚠️\
# <font color='darkorange'> Старайся разбивать код на строчки так, что бы всё было перед глазами. По стандарту строка кода не должна превышать 79 символов.</font>

# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ✔️\
# <font color='green'>👍</font>

# <font color='blue'><b>Комментарий ревьюера: </b></font> ⚠️\
# <font color='darkorange'> И зачем задавать явно? Можно отфильтровать список.</font>

# In[ ]:


# разобъем изначальную тренировочную выборку для прогнозирования черногвого концентрата
X_rougher_train_init, X_rougher_valid_init, y_rougher_train_init, y_rougher_valid_init = train_test_split(df_train_init[rougher_params_list], df_train_init[rougher_target], test_size=0.1 )
print(X_rougher_train_init.shape)
print(X_rougher_valid_init.shape)
print(y_rougher_train_init.shape)
print(y_rougher_valid_init.shape)


# <font color='blue'><b>Комментарий ревьюера: </b></font> ⚠️\
# <font color='darkorange'> В задании предлагается оценить модели кроссвалидацией, для кроссвалидации отдельная валидационная выборка не нужна. Отделяя валидационную выборку мы сокращаем данные для обучения моделей.</font>

# In[ ]:


# разобъем отфильтрованную тренировочную выборку для прогнозирования черногвого концентрата
X_rougher_train, X_rougher_valid, y_rougher_train, y_rougher_valid = train_test_split(df_train[rougher_params_list], df_train[rougher_target], test_size=0.1 )
print(X_rougher_train.shape)
print(X_rougher_valid.shape)
print(y_rougher_train.shape)
print(y_rougher_valid.shape)


# ### Подготовка параметров для прогнозирования final.output.recovery

# Предположительно на выходные параметры должны влиять состояние на последней стадии очистки, но есть вероятность что влияют все параметры. Чтобы решить дилемму создадим два варианта списка параметров

# In[ ]:


#final_params_list_1 = ['secondary_cleaner.state.floatbank2_a_air', 'secondary_cleaner.state.floatbank2_a_level', 'secondary_cleaner.state.floatbank2_b_air', 'secondary_cleaner.state.floatbank2_b_level', 'secondary_cleaner.state.floatbank3_a_air', 'secondary_cleaner.state.floatbank3_a_level', 'secondary_cleaner.state.floatbank3_b_air', 'secondary_cleaner.state.floatbank3_b_level', 'secondary_cleaner.state.floatbank4_a_air', 'secondary_cleaner.state.floatbank4_a_level', 'secondary_cleaner.state.floatbank4_b_air', 'secondary_cleaner.state.floatbank4_b_level', 'secondary_cleaner.state.floatbank5_a_air', 'secondary_cleaner.state.floatbank5_a_level', 'secondary_cleaner.state.floatbank5_b_air', 'secondary_cleaner.state.floatbank5_b_level', 'secondary_cleaner.state.floatbank6_a_air', 'secondary_cleaner.state.floatbank6_a_level']

final_params_list_2 = df_train.drop(columns=['final.output.recovery', 'rougher.output.recovery'], axis=1).columns

final_target = 'final.output.recovery'


# #### Без выбросов

# In[ ]:


# разобъем изначальную тренировочную выборку для прогнозирования финального концентрата
X_final_train_init, X_final_valid_init, y_final_train_init, y_final_valid_init = train_test_split(df_train_init[final_params_list_2], df_train_init[final_target], test_size=0.1 )
print(X_final_train_init.shape)
print(X_final_valid_init.shape)
print(y_final_train_init.shape)
print(y_final_valid_init.shape)


# In[ ]:


# разобъем отфильтрованную тренировочную выборку финального концентрата
X_final_train, X_final_valid, y_final_train, y_final_valid = train_test_split(df_train[final_params_list_2], df_train[final_target] , test_size = 0.1)
print(X_final_train.shape)
print(X_final_valid.shape)
print(y_final_train.shape)
print(y_final_valid.shape) 


# #### Прогноз эффективности обогащения чернового концентрата (rougher.output.recovery)

# In[ ]:


features = df_train[rougher_params_list]
target = df_train[rougher_target]
features_final = df_train[final_params_list_2]
target_final = df_train[final_target]


# In[ ]:


model_dr = DummyRegressor()

model_dr.fit(target, features)

scorer = make_scorer(smape, greater_is_better=False)
dr_smape_rough = cross_val_score(model_dr, features, target, scoring=scorer, cv=5).mean()
print('sMAPE_rougher=', (dr_smape_rough)*-1)


# In[ ]:


dr_smape_final = cross_val_score(model_dr, features_final, target_final, scoring=scorer, cv=5).mean()
print('sMAPE_final=', dr_smape_final*-1)


# #### Прогноз эффективности обогащения финального концентрата (final.output.recovery)

# In[ ]:


print('Overal sMAPE_rougher=', overal_smape(dr_smape_rough, dr_smape_final))


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'> Классно, что сравниваешь с константной моделью!</font>

# <font color='blue'><b>Комментарий ревьюера: </b></font> ⚠️\
# <font color='darkorange'> Но все оценки должны быть кроссвалидацией.</font>

# <font color='purple'><b>Комментарий студента: Поправил</b></font>

# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ✔️\
# <font color='green'> Отлично, что оценка по кроссвалидацией по sMAPE!</font>

# Итого, полуичили baseline для валидационной и тестовой выборок для "дамми"-модели на тренировочных данных без фильтрации от выбросов
# 
# Baseline получен. Есть к чему стремиться. Посмотрим что дадут другие модели

# ### Случайный лес

# In[ ]:


# определим естиматор
random_forest_model = RandomForestRegressor(random_state=42, n_jobs=-1)


# In[ ]:



# зададим распрпделение параметров для optuna
param_distributions_randomforest_optuna = {
    'n_estimators': optuna.distributions.IntDistribution(10,1000,10),
    'max_depth': optuna.distributions.IntDistribution(2,40,1),
    'min_samples_split': optuna.distributions.IntDistribution(20,60,5),
    'min_samples_leaf': optuna.distributions.IntDistribution(1, 201, 10),
    'max_features': optuna.distributions.CategoricalDistribution(['sqrt','log2']),
    'bootstrap' : optuna.distributions.CategoricalDistribution([True, False])}


# <font color='blue'><b>Комментарий ревьюера : </b></font> ✔️ <br>
# <font color='green'>Верно, что не забываешь указать greater_is_better=False, OptunaSearchCV Максимизирует метрику, а SMAPE — ошибка, её минимизируем.</font>

# #### Прогноз эффективности обогащения чернового концентрата (rougher.output.recovery)

# In[ ]:


opt_search_rougher = optuna.integration.OptunaSearchCV(estimator=random_forest_model,
                                                       param_distributions=param_distributions_randomforest_optuna, 
                                                       cv=15 , 
                                                       n_trials=100, 
                                                       scoring=scorer, 
                                                       timeout=10, 
                                                       random_state=42)


# In[ ]:


opt_search_rougher.fit(features, target)
opt_search_rougher_score = opt_search_rougher.best_score_

print('sMAPE_rougher = ', opt_search_rougher_score*-1)


# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'><s> Среднюю кроссвалидацию лучшей модели можно получить как `opt_search_rougher.best_score_`\
#     Оценивать модель для выбора лучшей нужно по ней.</s></font>

# <font color='purple'><b>Комментарий студента: Поправил</b></font>

# <font color='blue'><b>Комментарий ревьюера : </b></font> ✔️\
# <font color='green'>Подбор гиперпараметров проведён верно. Использована целевая метрика, направление отбора верное ).</font>

# #### Прогноз эффективности обогащения финального концентрата (final.output.recovery)

# In[ ]:


opt_search_final = optuna.integration.OptunaSearchCV(random_forest_model, param_distributions_randomforest_optuna, 10 , n_trials=100, scoring=scorer, timeout=600, random_state=42)


# In[ ]:


opt_search_final.fit(features_final, target_final)
opt_search_final_score = opt_search_final.best_score_
print('sMAPE_final = ', opt_search_final_score*-1)


# In[ ]:


opt_search_final_param = opt_search_final.best_params_
print(opt_search_final_param)


# In[ ]:


opt_search_best = opt_search_final.best_estimator_
print(opt_search_best)


# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ✔️\
# <font color='green'>👍</font>

# ### Итоговые расчеты

# In[ ]:


print('итоговое значение sMAPE на двух целевых параметрах = ', overal_smape(smape_rougher = opt_search_rougher_score, smape_final = opt_search_final_score))


# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'><s> Выбирать модели на тестовых данных нельзя - так мы подстраиваемся конкреитно под тестовую выборку. Тестирование проводится только для лучшей модели отобранной по результатам кроссвалидации.</s></font>

# <font color='purple'><b>Комментарий студента: Поправил</b></font>

# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ✔️\
# <font color='green'>👍</font>

# ## Линейная регрессия

# ### На исходной выборке

# In[ ]:


linreg_model = LinearRegression(n_jobs=-1)
linreg_model.fit(features, target)
lm_smape_rough = cross_val_score(linreg_model, features, target, scoring=scorer, cv=5).mean()
print('sMAPE_rougher на линейной модели. (rough) = ', lm_smape_rough*-1)

linreg_model.fit(features_final, target_final)
lm_smape_final = cross_val_score(linreg_model, features_final, target_final, scoring=scorer, cv=5).mean()
print('sMAPE_rougher на линейной модели. (final) = ', lm_smape_final*-1)


# In[ ]:


print('итоговое значение sMAPE= ', overal_smape(smape_rougher = lm_smape_rough*-1, smape_final = lm_smape_final*-1))


# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ✔️\
# <font color='green'>👍</font>

# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'><s> Оценивать модель нужно кроссвалидацией на основе нашей метрики, то есть smape. Для этого из нее нужно сделать scorer с помощью функции make_scorer. Результат нужно передать в параметр scoring функции основанной на кросс-валидации (например cross_val_score).</s></font>

# #### На тестовой выборке

# In[ ]:


df_test.fillna(0, inplace = True)


# In[ ]:


best_model = opt_search_best
best_model.fit(X_rougher_train, y_rougher_train)
best_model_predict_valid = best_model.predict(X_rougher_valid)
smape_best_model_valid = smape(y_rougher_valid, best_model_predict_valid)
print('sMAPE_rougher на линейной модели. (валидационная выборка) = ', smape_best_model_valid)

best_model_test = best_model.predict(df_test[rougher_params_list])
smape_best_model_test_score = smape(df_test[rougher_target], best_model_test)
print('sMAPE_rougher на линейной модели. (тестовая выборка) = ', smape_best_model_test_score)


# In[ ]:


best_model = opt_search_best
best_model.fit(X_final_train, y_final_train)
best_model_predict_valid = best_model.predict(X_final_valid)
smape_best_model_valid_final = smape(y_final_valid, best_model_predict_valid)
print('sMAPE_rougher на линейной модели. (валидационная выборка) = ', smape_best_model_valid_final)

best_model_test = best_model.predict(df_test[final_params_list_2])
smape_best_model_test_score_final = smape(df_test[final_target], best_model_test)
print('sMAPE_rougher на линейной модели. (тестовая выборка) = ', smape_best_model_test_score_final)


# In[ ]:


print('итоговое значение sMAPE на валидационной выборке = ', overal_smape(smape_rougher = smape_best_model_valid, smape_final = smape_best_model_valid_final))

print('итоговое значение sMAPE на тестовой выборке = ', overal_smape(smape_rougher = smape_best_model_test_score, smape_final = smape_best_model_test_score_final))


# Случайный лес справился лучше

# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ✔️\
# <font color='green'>Тестирование проведено верно.</font>

# 3. Заключение
# В настоящем проекте проведена следующая работа и сделаны следующие выводы:
# 
# Проведена предварительная обработка данных в т.ч. заполнение пропусков: пропуски были заполнены средним из соседних значений, заполнены пропуски до 25-ти значений поряд. Пропуски больше чем 25 значений были удалены.
# Проведена валидация признака rougher.output.recovery: Приведенное в исходных данных значение было успешно валидировано. Разница в расчетных и предоставленных значениях исчезающе мала.
# Проведены пнализ данных и исследования:
# различия состава параметров в тестовой и тренировочной выборках:
# в тестовой выборке отсутствуют
# параметры продукта после каждой стадии
# отвальные хвосты после каждой стадии, в т.ч. общие
# расчетные харакетристики сырья
# целевые параметры процесса
# в тестовой выборке присутствуют:
# все входные параметры для флотации и первичной очистки (input)
# параметры состояния на каждом этапе (state)
# изменения концентрации металлов на различных этапах очистки:
# Концентрация золота возрастает
# Концентрация серебра падает
# Концентрация свинца незначительно растет
# распределения носят нормальный характер без ярко-выраженных скосов с примерно симметричным количеством выбросов, как следствие медианное значение близко к ранее найденному среднему
# распределения размеров гранул сырья на обучающей и тестовой выборках: В тестовой и тренировочной выборках относительная разница в среднего размера гранул сырья составляет 6.23%
# 
# суммарная концентрация веществ на стадии флотации = 691200.81
# суммарная концентрация веществ на стадии первичной очистки = 614247.23
# суммарная концентрация веществ финального продукта = 689620.0
# 
# суммарная концентрация веществ на стадии флотации = 397851.25
# суммарная концентрация веществ на стадии первичной очистки = 514000.26
# суммарная концентрация веществ финального продукта = 598564.08
# 
# Подготовлена "кастомная" функция для оценки промежуточных и итоговой sMAPE
# Построен baseline на dummy-моделе
# Обучена модель randomforestRegressor с использованием optuna и модель LinearRegression. Линейная регрессия за счет отсутствия возможности более тонкой настройки дала несколько хуже результаты, но тем не менее все равно примемлемые.
# Тестовая выборка отличается от тренировчной, в результате чего обученные на валидационной выборке модели дают успешные предсказания и плохо работают на тестовой.
# 
# исходная - без обработки. кросс-валидация = 15 фолдов, размер тестовой выборки 25%
# отфильтрованная от выбросов в значенияхпараметров output. кросс-валидация = 10 фолдов, размер тестовой выборки 25%
# исходная - без обработки. кросс-валидация = 10 фолдов, размер тестовой выборки 10%
# отфильтрованная от выбросов в значенияхпараметров output. кросс-валидация = 15 фолдов, размер тестовой выборки 10
# 
# наилучший результат получился на валидационной выборке для случайного леса.
# увеличение размера кросс-валидации и уменьшение тестовой выборки не влияет на результат (в целом +- то же самое)
# исходя из этого можно сделать вывод что предварительная обработка данных имеет смысл.

# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'> Очень приятно видеть вывод в конце проекта!\
# Приведены ответы на главные вопросы проекта.</font>

# <font color='blue'><b>Итоговый комментарий ревьюера 2</b></font>\
# <font color='green'> Сергей, проект принят! \
# Все этапы пройдены. Все критичные рекомендации учтены.\
# Надеюсь, тебе понравился процесс выполнения и результат.</font> \
# <font color='blue'><b>Спасибо, удачи в освоении профессии!</b></font>

# <font color='blue'><b>Итоговый комментарий ревьюера</b></font>\
# <font color='green'>Сергей, хороший проект получился!
# Большое спасибо за проделанную работу. Видно, что приложено много усилий.
# Выводы и рассуждения получились содержательными, их было интересно читать.
# </font>
# 
# <font color='blue'>Что нужно исправить:</font>
# <ul><font color='red'>Напиши вывод по сравнению размеров гранул в трейн и тест.</font></ul>
# <ul><font color='red'>Все оценки моделей должны быть кроссвалидацией по sMAPE (кроме теста, конечно))</font></ul>
# 
# <font color='blue'>Что можно сделать лучше:</font>
# <font color='darkorange'>В работе я оставил несколько советов. Буду рад, если ты учтешь их.</font></ul>
# 
# <font color='blue'><b>Жду новую версию проекта :)</b></font>

# ## Чек-лист готовности проекта

# - [x]  Jupyter Notebook открыт
# - [ ]  Весь код выполняется без ошибок
# - [ ]  Ячейки с кодом расположены в порядке выполнения
# - [ ]  Выполнен шаг 1: данные подготовлены
#     - [ ]  Проверена формула вычисления эффективности обогащения
#     - [ ]  Проанализированы признаки, недоступные в тестовой выборке
#     - [ ]  Проведена предобработка данных
# - [ ]  Выполнен шаг 2: данные проанализированы
#     - [ ]  Исследовано изменение концентрации элементов на каждом этапе
#     - [ ]  Проанализированы распределения размеров гранул на обучающей и тестовой выборках
#     - [ ]  Исследованы суммарные концентрации
# - [ ]  Выполнен шаг 3: построена модель прогнозирования
#     - [ ]  Написана функция для вычисления итогового *sMAPE*
#     - [ ]  Обучено и проверено несколько моделей
#     - [ ]  Выбрана лучшая модель, её качество проверено на тестовой выборке
