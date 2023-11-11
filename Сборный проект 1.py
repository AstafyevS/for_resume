#!/usr/bin/env python
# coding: utf-8

# <div style="border:solid blue 3px; padding: 20px">
# 
# **Привет, Сергей!** 👋
# 
# Меня зовут Алексей Манелюк, я буду проверять твой проект. Предлагаю общаться на "ты" 😉, но если неудобно - скажи и будем на "вы".
# 
# Моя цель - не указать на совершенные тобой ошибки, а поделиться своим опытом и помочь тебе сделать крутой проект, который не стыдно будет положить в твое портфолио.
# 
# В тексте проекта ты увидишь мои комментарии, пожалуйста, не удаляй, не перемещай и не изменяй их. У комментариев могут быть приписки **V2**, **V3** и т.д., которые соответствуют версии проверки. Я буду использовать цветовую разметку:
# <p>
# <div class="alert-danger">
# <b>Комментарий ревьюера ❌</b>
# 
# Так выделены самые важные замечания. Без их отработки я не смогу принять проект. </div>
# 
# <div class="alert-warning">
# <b>Комментарий ревьюера ⚠️</b>
# 
# Так выделены небольшие замечания или полезные советы. Постарайся учесть их.</div>
# <p>
# <div class="alert-success">
# <b>Комментарий ревьюера ✔️</b>
# 
# Так выделены правильные решения.</div>
# 
# Предлагаю работать в диалоге: если что-то дорабатываешь или отвечаешь на мой комментарий - пиши об этом. Мне будет удобней, если ты выделишь свои комментарии:
# <p>
# <div class="alert-info">
# <b>Комментарий студента</b>
# 
# Например, так.</div>
# 
# Все это поможет мне быстрее принять твой проект.</div>

# <p>
# <div class="alert-info">
# <b>Комментарий студента</b>
# 
# Можно на "ты". Как отправлю тебе свежую версию огромная просьба проверить по возможности сразу. Сегодня последний срок жесткого дедлайна. Изза вынужденных переездов отстал от графика. Дедлайн просил перенести. Нагнал. Но перенесли только на сегодня.</div>

# # Проект по анализу данных интернет магазина "Стримчик"

# В данном проекте необходимо выявить закономерности и основные данные, которые помогут спланировать рекламную компанию на 2017. Нам доступны данные о продаже игр до 2016 года.

# <div class="alert-warning">
# <b>Комментарий ревьюера ⚠️</b>
# 
# 
# 📌 После названия, обязательно нужно добавлять бриф, цели исследования, описание данных и план работы.</div>

# <p>
# <div class="alert-info">
# <b>Комментарий студента</b>
# 
# Добавил.</div>

# <div class="alert-success">
# <b>Комментарий ревьюера V2 ✔️</b>
# 
# Можно чуть подробнее: описание данных, план работы.</div>

# ## Предобработка данных

# In[1]:


import pandas as pd
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set()
import numpy as np
from scipy import stats as st


# In[2]:


data = pd.read_csv('/datasets/games.csv', sep=',')

display(data.head())

data.info()

print('\nЧисло полных дубликатов:', data.duplicated().sum())
print('\nЧисло дубликатов по столбцам Название-платформа:', data.duplicated(subset=['Name', 'Platform']).sum())

data.describe()
    


# Изучим эти дубликаты

# In[3]:


data[data.duplicated(subset=['Name', 'Platform'], keep=False)]


# In[4]:


data = data.drop_duplicates(subset=['Name', 'Platform'], keep='first')


# Проверим их снова

# In[5]:


print('\nЧисло дубликатов по столбцам Название-платформа:', data.duplicated(subset=['Name', 'Platform']).sum())


# <div class="alert-warning">
# <b>Комментарий ревьюера ⚠️</b>
# 
# 📌 Лучше разделять ячейки загрузки библиотек и данных. Если понадобится подключить дополнительную библиотеку, не нужно будет заново грузить данные и перезаписывать переменные.
# 
# Дубликаты также стоит проверить и по столбцам: название - платформа</div>

# <p>
# <div class="alert-info">
# <b>Комментарий студента</b>
# 
# Поправил.</div>

# <div class="alert-success">
# <b>Комментарий ревьюера V2 ✔️</b>
# 
# Отлично! Правда с удалением всех дубликатов ты поторопился - некоторые были переизданиями.</div>

# ### Вывод

# Всего в датафрейме 16715 записей займемся их предобработкой

# <div class="alert-success">
# <b>Комментарий ревьюера ✔️</b>
# 
# Данные загружены верно, получена общая информация о датафрейме. 🚀</div>

# <div class="alert-warning">
# <b>Комментарий ревьюера ⚠️</b>
# 
# Добавь пожалуйста в свою работу подзаголовки, в соответствии с шагами задания.
# 
# `#` - 1 уровень
# `##` - 2 уровень
# `###` - 3 уровень
#     
#    
# </div>

# ### Приводим названия столбцов к нижнему регистру

# In[6]:


data.columns = map(str.lower, data.columns)


# <div class="alert-success">
# <b>Комментарий ревьюера ✔️</b>
# 
# 📌 Привести названия столбцов к нижнему регистру можно и так: 
# 
# `df.columns = df.columns.str.lower()`</div>

# ### Создадим столбец с общими продажами по всем регионам

# In[7]:


data['total_sales'] = data['na_sales'] + data['eu_sales'] + data['other_sales'] + data['jp_sales']


# ### Проверим данные на пропуски

# In[8]:


#Удаляем пропуски даты релиза по году, т.к. доля этих значений во всем датафрейме очень мала(времененное решение)
data.dropna(subset=['year_of_release'], inplace = True)


# In[9]:


data.isna().sum()


# Пропуски в рейтингах пользователей и критиков оставляем, т.к. их довольно много. Возможно они есть в таблице, т.к. игра настолько не популярна, что не нашлось пользователей желающих ее оценить. Выделим игры без рейтинга в отдельную категорию NR(NoRating)

# In[10]:


data['rating'] = data['rating'].fillna('NR')


# <div class="alert-warning">
# <b>Комментарий ревьюера ⚠️</b>
# 
# Имеет смысл выделить игры без рейтинга в отдельную категорию, они понадобятся в дальнейшем анализе. Также можно проверить и заменить устаревшие значения рейтинга.
# 
# Что касается оценок критиков и пользователей - согласен, можно оставить пропуски.</div>

# <div class="alert-warning">
# <b>Комментарий ревьюера ⚠️</b>
# 
# Полностью закомментированный код лучше удалить из работы.</div>

# <p>
# <div class="alert-info">
# <b>Комментарий студента</b>
# 
# Поправил.</div>

# <div class="alert-success">
# <b>Комментарий ревьюера V2 ✔️</b>
# 
# Супер! Теперь игры без рейтинга не выпадут из портретов пользователей дальше.</div>

# ### Преобразуем year_of_release в int и убираем дробную часть года.

# In[11]:


#Преобразуем year_of_release в int и убираем дробную часть года.
data['year_of_release'] = round(data['year_of_release'].astype(int, errors='ignore'), 0)
data['user_score'] = data['user_score'].fillna(value=False)


# In[12]:


data.info()


# ### Исследуем абревиатуру 'tbd' в столбце user_score

# In[13]:


#Исследуем абревиатуру 'tbd' в столбце user_score
tbd = data.loc[data['user_score'] == 'tbd']
tbd.count()


# Абревиатура tbd встречается достсточно в 2376 случаях пока что никакой зависимости не замечено. Она означает, что рейтинг для данной игры не определен. Поэтому заменим его на нулевое значение.

# <div class="alert-warning">
# <b>Комментарий ревьюера ⚠️</b>
# 
# Ты же выше писал, что пропуски в оценках не будешь заполнять, почему решил теперь заполнить? И где заполнение ключом 9999?</div>

# <div class="alert-info">
# <b>Комментарий студента</b>
# 
# Поправил. В коментарии были остатки другого решения от которого решил отказаться. По TBD мешает построению графиков если оставить текстовое значение.</div>

# In[14]:


data = data.replace('tbd', False)
data['user_score'] = data['user_score'].astype(float)
data.info()


# In[15]:


# Код ревьюера
data['user_score'].sort_values().unique()


# ### Вывод

# Пропуски находились в графах даты релиза по году, пользовательских оценок, оценок критиками и рейтинга по возрасту. Оставлены только пропуски по оценкам пользователей и критиков, остальные заполнены или удалены. Из 16700 осталось 1640.

# <div class="alert-warning">
# <b>Комментарий ревьюера ⚠️</b>
# 
# В конце предобработки имеет смысл написать краткий вывод о проделанной работе: все ли пропуски и дубликаты обработаны, какие типы данных изменены, сравнить первоначальную таблицу с обработанной на предмет того, сколько данных мы выбросили. </div>

# <div class="alert-info">
# <b>Комментарий студента</b>
# 
# Сделано.</div>

# <div class="alert-success">
# <b>Комментарий ревьюера V2 ✔️</b>
# 
# 👍</div>

# ## Исследовательский анализ данных

# ### Изучим сколько игр выпускалось в разные годы. Важны ли все данные по годам для прогнозирования дальнейших продаж.

# In[16]:


display(data.groupby('year_of_release')['total_sales'].count())
data.groupby('year_of_release')['total_sales'].count().plot(title='Количество релизов игр по годам',
                                                           xlabel='Года',
                                                           ylabel='Выпущеные игры'
)


# С 1980 до 2001 в год выпускалось до 500 игр. Этим колличеством релизов в год, тем более учитывая, что с тех пор индустрия сильно поменялась можно принебречь.

# <div class="alert-warning">
# <b>Комментарий ревьюера ⚠️</b>
# 
# Здесь стоит добавить визуализацию. </div>

# <div class="alert-info">
# <b>Комментарий студента</b>
# 
# Сделано.</div>

# <div class="alert-success">
# <b>Комментарий ревьюера V2 ✔️</b>
# 
# Хорошо бы убрать надпись `<AxesSubplot:title={'center':'Количество релизов игр по годам'}, xlabel='Года', ylabel='Выпущеные игры'>`. Размер графика можно увеличить параметром `figsize`.</div>

# ### Изучим данные по разным игровым платформам

# Получим общее представление о жизненом цикле платформ на которые выпускаются игры на примере нескольких платформ из разных периодов.

# In[17]:


data[data['platform'] == 'NES']['year_of_release'].plot(kind='hist',alpha=0.5,figsize=(10,10), title='Продажи по годам', xlabel='Года', ylabel='Количество продаж')
data[data['platform'] == 'SNES']['year_of_release'].plot(kind='hist',alpha=0.5,figsize=(10,10), title='Продажи по годам', xlabel='Года', ylabel='Количество продаж')


# <div class="alert-warning">
# <b>Комментарий ревьюера ⚠️</b>
# 
# Для того, чтобы график смотрелся аккуратнее стоит убрать надпись `<AxesSubplot:>`, добавить название, подписать оси. Размер графика можно отрегулировать параметром `figsize`.
#     
# Эти замечания касаются всех графиков в работе.</div>

# <div class="alert-success">
# <b>Комментарий ревьюера V2 ✔️</b>
# 
# Эту надпись можно убрать, если дописать `plt.show()` в конце кода графика.</div>

# In[18]:


data[data['platform'] == 'Wii']['year_of_release'].hist(bins=11,alpha=0.5)
data[data['platform'] == 'PS2']['year_of_release'].hist(bins=11,alpha=0.5)


# Делаем вывод, что в случае прекращения продаж в одном из годов жизненый цикл платформ прерывается:следовательно они становятся не актуальными по отношению к будущему году. Рассматривать их не стоит. Жизненый цикл платформы в среднем составляет 10 лет.

# <div class="alert-danger">
# <b>Комментарий ревьюера ❌</b>
# 
# За какой характерный срок появляются новые и исчезают старые платформы? </div>

# <div class="alert-info">
# <b>Комментарий студента</b>
# 
# Сделано.</div>

# <div class="alert-success">
# <b>Комментарий ревьюера V2 ✔️</b>
# 
# Верно, в среднем за 10 лет платформа теряет популярность.</div>

# Выясним игры каких платформ продаются последние пять лет.

# In[19]:


data_actual = data.query('year_of_release > 2014')

data_actual_platforms = data.query('year_of_release > 2014')['platform'].unique()
data_actual_platforms


# <div class="alert-danger">
# <b>Комментарий ревьюера ❌</b>
# 
# Для целей прогнозирования продаж на следующий год даже в традиционных бизнесах редко берут данные более чем за 2-3 года. А в такой динамично меняющейся индустрии, как компьютерные игры и вовсе не стоит брать слишком большой временной интервал - иначе обязательно захватишь уже отжившие тренды. </div>

# <div class="alert-info">
# <b>Комментарий студента</b>
# 
# Понял, принял, согласен.</div>

# <div class="alert-success">
# <b>Комментарий ревьюера V2 ✔️</b>
# 
# Теперь актуальный период для анализа и прогноза выбран верно.</div>

# Поподробнее взглянем на продажи за последние 5 лет на этих платформах.

# In[20]:


#data[data['platform'] == 'PS3'].groupby('year_of_release')['total_sales'].sum()
#data_actual.groupby(['year_of_release','platform'])['total_sales'].sum().hist('platform')
#data_actual[data_actual['platform'] =='Wii'].pivot_table(index='year_of_release', values='total_sales')
data_actual.groupby(['platform','year_of_release'])['total_sales'].sum()


# In[21]:


data[data['platform'] == 'PS3']['year_of_release'].hist(alpha = 0.5,label='PS3')

data[data['platform'] == 'PS2']['year_of_release'].hist(alpha = 0.5)

data[data['platform'] == 'X360']['year_of_release'].hist(alpha = 0.5)

data[data['platform'] == 'PS4']['year_of_release'].hist(alpha = 0.5)

data[data['platform'] == '3DS']['year_of_release'].hist(alpha = 0.5)

data[data['platform'] == 'XOne']['year_of_release'].hist(alpha = 0.5)


# Выберем платформы с наибольшими сумарными продажами

# In[22]:


data_actual.pivot_table(index='year_of_release', values='total_sales', columns='platform', aggfunc = sum)#.plot(kind='hist', figsize=[15,15])


# Это 3DS с 15.14, Xone 26.15 и PS4 69.25. А самыми перспективными из них являются всвязи со табильно растущими продажами и новизной являются: PS4 и XOne.

# <div class="alert-warning">
# <b>Комментарий ревьюера ⚠️</b>
# 
# Эти платформы ты определяешь как потенциально прибыльные?</div>

# <div class="alert-info">
# <b>Комментарий студента</b>
# 
# Да, дополнил.</div>

# <div class="alert-success">
# <b>Комментарий ревьюера V2 ✔️</b>
# 
# С выбором потенциально прибыльных платформ согласен.</div>

# Платформы у которых отсутствуют продажи последний год можем исключить из актуальной выборки.

# In[23]:


data_actual=data.query('platform == "3DS" or platform == "PC" or platform == "PS3" or platform == "PS4" or platform == "PSV" or platform == "Wii" or platform == "WiiU" or platform == "X360" or platform == "XOne"')


# Все платформы в 2016 году упали по продажам. Скорее всего данные по 2016 году не полные.
# Наиболее актуальны PS4, XOne. Продажи на них последние 3 года растут. В то время как на довольно актуальной на данный момент 3DS начали снижаться еще в 2014 году.

# Посторим график диаграммы размаха по глобальным продажам игр в разбивке по платформам.

# In[24]:


#data_actual.pivot_table(index='platform', values='total_sales', aggfunc = sum).boxplot()#.plot(kind='hist', figsize=[15,15])
#data_actual.groupby('platform')['total_sales'].sum()
#data_actual.boxplot(figsize=(10,5))
#plt.ylim(-5, 100)
#data_actual.pivot_table(index=['platform', 'name'], values='total_sales').boxplot(x='platform','name')
sns.boxplot(data=data_actual, y='total_sales', x='platform')
plt.ylim(0, 3)
plt.figure(figsize=(25,9))


# Медиана у всех платформ находится у первого квартиля. На разных платформах присутствуют выбросы в со стороны 3 квартиля. Больше всего выбросов у X360,PS4,PS3,XOne,Wii. Делаем вывод, что там имеются крайне удачные проекты.

# <div class="alert-danger">
# <b>Комментарий ревьюера ❌</b>
# 
# По заданию надо построить график «ящик с усами» по глобальным продажам игр в разбивке по платформам. То есть на одном графике отобразить несколько платформ - для сравнения медиан, межквартильных размахов и выбросов. Группировка по платформам и годам не нужна, нам интересны все продажи игр.</div>

# <div class="alert-info">
# <b>Комментарий студента</b>
# 
# Сделал.</div>

# <div class="alert-success">
# <b>Комментарий ревьюера V2 ✔️</b>
# 
# Ок. Тут также стоит увеличить размер визуализации для наглядности.</div>

# In[25]:


print('Кореляция продаж и оценки критиками')
data_actual[data_actual['platform'] == 'PS3'].pivot_table(index='total_sales', values='critic_score').plot(grid=True, style='o', figsize=(10,5), alpha=0.7)
plt.show()
print('Корреляция равна:',data_actual[data_actual['platform'] == 'PS3']['total_sales'].corr(data_actual[data_actual['platform'] == 'PS3']['critic_score']))


# In[26]:


print('Кореляция продаж и оценки пользователями')
data_actual[data_actual['platform'] == 'PS3'].loc[~data_actual['user_score'].isnull() & ~data_actual['critic_score'].isna()].pivot_table(index='total_sales', values='user_score').plot(grid=True, style='o', figsize=(10,5), alpha=0.7)
plt.show()
print('Корреляция равна:',data_actual[data_actual['platform'] == 'PS3']['total_sales'].corr(data_actual[data_actual['platform'] == 'PS3']['user_score']))


# <div class="alert-danger">
# <b>Комментарий ревьюера ❌</b>
# 
# А почему ты решил строить диаграмму рассеяния по сводной таблице? Туда попадают усредненные данные и вообще в чем ее смысл? 👇
# 
# Кроме того, для расчета корректного коэффициента корреляции нужно убрать пропуски из выборки.</div>

# <div class="alert-info">
# <b>Комментарий студента</b>
# 
# Пропуски и нулевые значения я убрал а диаграмма эта прям в проекте описана..."Посмотрите, как влияют на продажи внутри одной популярной платформы отзывы пользователей и критиков. Постройте диаграмму рассеяния и посчитайте корреляцию между отзывами и продажами. Сформулируйте выводы". Я исправлять все начала потом в задание посмотрел.)</div>

# <div class="alert-warning">
# <b>Комментарий ревьюера V2 ⚠️</b>
# 
# Правильнее было бы сделать это так: 👇</div>

# In[49]:


# Код ревьюера

ps4_data = data_actual.query('platform == "PS4" & user_score > 0 & critic_score > 0')

# Диаграмма оценки пользователей и продажи
plt.figure(figsize = (15, 5))
sns.scatterplot(x = 'user_score', y = 'total_sales', data = ps4_data)
plt.show()
print('Коэффициент корреляции:', ps4_data['user_score'].corr(ps4_data['total_sales']))

# Диаграмма оценки критиков и продажи
plt.figure(figsize = (15, 5))
sns.scatterplot(x = 'critic_score', y = 'total_sales', data = ps4_data)
plt.show()
print('Коэффициент корреляции:', ps4_data['critic_score'].corr(ps4_data['total_sales']))


# In[27]:


# Код ревьюера
data_actual.loc[data['platform'] == 'PS3'].pivot_table(index='total_sales', values='user_score')


# Корреляция по оценкам пользователей низкая, а по оценка критиков средняя, что говорит нам о том, что для финансового успеха игры эти параметры не первостепенные.

# In[28]:


print('Кореляция продаж и оценки критиками')
data_actual[data_actual['platform'] == 'XOne'].loc[~data_actual['user_score'].isnull() & ~data_actual['critic_score'].isna()].pivot_table(index='total_sales', values='critic_score').plot(grid=True, style='o', figsize=(10,5), alpha=0.7)
plt.show()
print('Корреляция равна:',data_actual[data_actual['platform'] == 'XOne']['total_sales'].corr(data_actual[data_actual['platform'] == 'XOne']['critic_score']))


# In[29]:


print('Кореляция продаж и оценки пользователями')
data_actual.loc[data['platform'] == 'XOne'].loc[~data_actual['user_score'].isnull() & ~data_actual['critic_score'].isna()].pivot_table(index='total_sales', values='user_score').plot(grid=True, style='o', figsize=(10,5), alpha=0.7)
plt.show()
print('Корреляция равна:',data_actual[data_actual['platform'] == 'XOne']['total_sales'].corr(data_actual[data_actual['platform'] == 'XOne']['user_score']))


# In[30]:


print('Кореляция продаж и оценки критиками')
data_actual[data_actual['platform'] == '3DS'].loc[~data_actual['user_score'].isnull() & ~data_actual['critic_score'].isna()].pivot_table(index='total_sales', values='critic_score').plot(grid=True, style='o', figsize=(10,5), alpha=0.7)
plt.show()
print('Корреляция равна:',data_actual[data_actual['platform'] == '3DS']['total_sales'].corr(data_actual[data_actual['platform'] == '3DS']['critic_score']))


# In[31]:


print('Корреляция продаж и оценки пользователями')
data_actual.loc[data['platform'] == '3DS'].loc[~data_actual['user_score'].isnull() & ~data_actual['critic_score'].isna()].pivot_table(index='total_sales', values='user_score').plot(grid=True, style='o', figsize=(10,5), alpha=0.7)
plt.show()
print('Корреляция равна:',data_actual[data_actual['platform'] == '3DS']['total_sales'].corr(data_actual[data_actual['platform'] == '3DS']['user_score']))


# Корреляции этих показателей по другим платформам схожи.

# In[32]:


data_actual.pivot_table(index='genre', values='total_sales', aggfunc='count').sort_values(by='total_sales',ascending=False)#.plot(grid=True, style='o', figsize=(10,5), alpha=0.7)


# Больше всего релизов в жанре Action, а самый редко выпускаемый жанр - Puzzle

# In[33]:


print(data_actual.pivot_table(index='genre', values='total_sales', aggfunc='sum').sort_values(by='total_sales',ascending=False))

data_actual.pivot_table(index='genre', values='total_sales', aggfunc='sum').sort_values(by='total_sales',ascending=False)


# In[34]:


data_actual.pivot_table(index='genre', values='total_sales', aggfunc='mean').sort_values(by='total_sales',ascending=False)


# In[35]:


data_actual.pivot_table(index='genre', values='total_sales', aggfunc='median').sort_values(by='total_sales',ascending=False)


# Жанр Action по общему колличеству продаж также лидирует, а Puzzle также находится на последнем месте. А вот стабильно хорошие продажи(определенные по средннему значению и медиане) у жанра Shooter. Игры именно такого жанра стоит выставлять на полки магазина.

# <div class="alert-success">
# <b>Комментарий ревьюера V2 ✔️</b>
# 
# Верно, это говорит о том, что в среднем каждая игра жанра Shooter продается в большем количестве экземпляров.</div>

# <div class="alert-danger">
# <b>Комментарий ревьюера ❌</b>
#     
# Ты правильно определил лидера и аутсайдера продаж. Но по заданию также нужно определить наиболее прибыльный жанр.
# 
# Обрати внимание, что общие продажи - плохая метрика для поиска наиболее прибыльного жанра. За высокими показателями общих продаж может скрываться множество мелких игр с низкими продажами. Или 2-3 звезды и куча провалов. Лучше найти жанр, где игры стабильно приносят высокий доход - для этого стоит рассмотреть средние или медианные продажи.</div>

# <div class="alert-warning">
# <b>Комментарий ревьюера ⚠️</b>
# 
# Не забывай пожалуйста заканчивать каждый шаг работы кратким выводом.</div>

# <div class="alert-info">
# <b>Комментарий студента</b>
# 
# Сделал</div>

# ## Определимся с предпочтениями по регионам

# ### Северная Америка

# In[36]:


display(data_actual.pivot_table(index='genre', values='na_sales', aggfunc='sum').sort_values(by='na_sales',ascending=False).head(5))
data_actual.pivot_table(index='genre', values='na_sales', aggfunc='sum').sort_values(by='na_sales',ascending=False).head(5).plot()
#data_actual.pivot_table(index='genre', values='na_sales', aggfunc='sum').sort_values(by='na_sales',ascending=False).plot()
display(data_actual.pivot_table(index='platform', values='na_sales', aggfunc='sum').sort_values(by='na_sales',ascending=False).head(5))
data_actual.pivot_table(index='platform', values='na_sales', aggfunc='sum').sort_values(by='na_sales',ascending=False).head(5).plot()
display(data_actual.pivot_table(index='rating', values='na_sales', aggfunc='sum').sort_values(by='na_sales',ascending=False))
data_actual.pivot_table(index='rating', values='na_sales', aggfunc='sum').sort_values(by='na_sales',ascending=False).plot()


# <div class="alert-warning">
# <b>Комментарий ревьюера ⚠️</b>
# 
# В целом, подход у тебя правильный, но из-за широких границ актуального периода в результаты попали устаревшие платформы, ориентироваться на них при планировании рекламной кампании заказчику нельзя.
# 
# Аналогично и жанры нужно рассматривать в рамках актуального периода.
# 
# Кроме того, для наглядности стоит добавить визуализации.</div>

# <div class="alert-info">
# <b>Комментарий студента</b>
# 
# Визуализацию добавил. Временные рамки актуальности изменил до 2 последних лет.</div>

# <div class="alert-success">
# <b>Комментарий ревьюера V2 ✔️</b>
# 
# Хорошо. Тип визуализации стоило бы выбрать столбчатую диаграмму. А также взять только топ-5</div>

# In[52]:


# Код ревьюера
data_actual.pivot_table(index='rating', values='na_sales', aggfunc='sum')    .sort_values(by='na_sales',ascending=False).head(5).plot(kind = 'bar')
plt.xticks(rotation = 0)
plt.show()


# Самые продаваемые игры с рейтингом M и E. Продажи "для всех" лишь на втором месте. Вероятнее всего присвоение рейтинга M и наличие "запрещенного для детей" контента вызывает интерес у публики.
# Превалируют продажи Action на Xbox360.

# In[37]:


display(data_actual.pivot_table(index='genre', values='eu_sales', aggfunc='sum').sort_values(by='eu_sales',ascending=False).head(5))
data_actual.pivot_table(index='genre', values='eu_sales', aggfunc='sum').sort_values(by='eu_sales',ascending=False).head(5).plot()
#data_actual.pivot_table(index='genre', values='na_sales', aggfunc='sum').sort_values(by='na_sales',ascending=False).plot()
display(data_actual.pivot_table(index='platform', values='eu_sales', aggfunc='sum').sort_values(by='eu_sales',ascending=False).head(5))
data_actual.pivot_table(index='platform', values='eu_sales', aggfunc='sum').sort_values(by='eu_sales',ascending=False).head(5).plot()
display(data_actual.pivot_table(index='rating', values='eu_sales', aggfunc='sum').sort_values(by='eu_sales',ascending=False))
data_actual.pivot_table(index='rating', values='eu_sales', aggfunc='sum').sort_values(by='eu_sales',ascending=False).plot()


# В жанрах в Европе все похоже на Северную Америку, а вот в платформах PS3 вышла на первое место. Продажи по рейтингу распределены аналогичным образом.

# In[38]:


display(data_actual.pivot_table(index='genre', values='jp_sales', aggfunc='sum').sort_values(by='jp_sales',ascending=False).head(5))
#data_actual.pivot_table(index='genre', values='na_sales', aggfunc='sum').sort_values(by='na_sales',ascending=False).plot()
display(data_actual.pivot_table(index='platform', values='jp_sales', aggfunc='sum').sort_values(by='jp_sales',ascending=False).head(5))
display(data_actual.pivot_table(index='rating', values='jp_sales', aggfunc='sum').sort_values(by='jp_sales',ascending=False))


# В Японии совсем другое распределение во всем. В платформах здесь лидирует 3DS, а в жанрах Role-Playing. Игры с самым продаваемым рейтингом - "Без рейтинга".

# <div class="alert-danger">
# <b>Комментарий ревьюера ❌</b>
# 
# При твоем способе подсчета игры без рейтинга оказываются полностью исключенными из анализа. Но продажи именно этих игр могут указать на ключевое различие в регионах.</div>

# <div class="alert-info">
# <b>Комментарий студента</b>
# 
# Добавил. Действительно в японии игры без рейтинга продавались лучше всего.</div>

# <div class="alert-success">
# <b>Комментарий ревьюера V2 ✔️</b>
# 
# Правильно, теперь видно, что рейтинг ESRB не имеет значения в Японии.</div>

# ## Проверка гипотез

# ### Гипотеза: средние пользовательские рейтинги платформ Xbox One и PC равны
# 
# H_0: Средние пользовательские рейтинги платформ Xbox One и PC равны H_1: Средний пользовательский рейтинг Xbox One ≠ PC

# In[39]:


pc_rate = data_actual.loc[data_actual['platform'] == 'PC'].loc[data_actual['user_score'] > 0]['user_score']
xbox_rate = data_actual[data_actual['platform'] == 'XOne'].loc[data_actual['user_score'] > 0]['user_score']
results = st.ttest_ind(
    pc_rate,
    xbox_rate,
    equal_var = False
    
)
alpha = 0.05
print('p-значение', results.pvalue)

if results.pvalue < alpha:
    print("Отвергаем нулевую гипотезу")
else:
    print("Не получилось отвергнуть нулевую гипотезу")


# <div class="alert-danger">
# <b>Комментарий ревьюера ❌</b>
# 
# Метод проверки гипотезы выбран верно. В значениях оценок не должно быть нулевых значений, тогда результат будет достоверный.</div>

# <div class="alert-info">
# <b>Комментарий студента</b>
# 
# И там и там исправил и графики соответственно изменились.</div>

# <div class="alert-warning">
# <b>Комментарий ревьюера V2 ⚠️</b>
# 
# Пропуски и tbd в оценках пользователей были обработаны не совсем верно, у меня получился другой результат статистического теста. 👇
# </div>

# In[72]:


# Код ревьюера
data_rev = pd.read_csv('/datasets/games.csv')
data_rev.columns = data_rev.columns.str.lower()

#Замена 'tbd' на Nan
data_rev['user_score'] = data_rev['user_score'].replace('tbd', np.NaN)
#Перевод в числовой тип
data_rev['user_score'] = data_rev['user_score'].astype('float')

xone_user_score = data_rev.query('year_of_release > 2014 & platform == "XOne" & user_score > 0')['user_score']
pc_user_score = data_rev.query('year_of_release > 2014 & platform == "PC" & user_score > 0')['user_score']

results = st.ttest_ind(xone_user_score, pc_user_score)
alpha = 0.05

print('p-value: ', results.pvalue)
if results.pvalue < alpha:
    print('Отвергаем нулевую гипотезу')
else:
    print('Не получилось отвергнуть нулевую гипотезу')


# In[73]:


# Код ревьюера
xone_user_score.hist(alpha = .5)
pc_user_score.hist(alpha = .5)
plt.show()


# ### Гипотеза №2: Средние пользовательские рейтинги жанров Action и Sports разные.
#    

# H_0: Средние пользовательский рейтинг жанра Action ≠ среднему пользовательскому рейтингу жанра Sports.

#  H_1: Средние пользовательский рейтинг жанра Action = среднему пользовательскому рейтингу жанра Sports.

# In[41]:


action_rate = data_actual.loc[data_actual['genre'] == 'Action'].loc[data_actual['user_score'] > 0]['user_score']
sports_rate = data_actual.loc[data_actual['genre'] == 'Sports'].loc[data_actual['user_score'] > 0]['user_score']

alpha = 0.05

results = st.ttest_ind(
    action_rate,
    sports_rate,
    equal_var=False
)

print('p-значение', results.pvalue)

if results.pvalue < alpha:
    print('Отвергаем нулевую гипотезу')
else:
    print('Нулевую гипотезу отвергнуть не удалось')


# In[42]:


# Код ревьюера
action_rate.hist();


# <div class="alert-danger">
# <b>Комментарий ревьюера ❌</b>
# 
# Аналогичное замечание и к проверке второй гипотезы.</div>

# <div class="alert-info">
# <b>Комментарий студента</b>
# 
# И там и там исправил и графики соответственно изменились.</div>

# <div class="alert-warning">
# <b>Комментарий ревьюера V2 ⚠️</b>
# 
# См замечание к проверке первой гипотезы.
# </div>

# ## Общие выводы

# В данном исследовании нам удалость выяснить множество особенностей которые помогут нам предсказать успех продаж той или иной игры в магазине "Стримчик".
# 
# + Данные показали нам небольшой жизненый цикл игровых платформ наибольшие продажи в котором продолжаются в течении пяти лет. Из чего следует вывод, что **старые платформы расчитывая на массового пользователя продавать не стоит**. Разве что на любителей "Ретро"
# 
# + Наиболее популярные платформы за последние 5 лет-PS3, Xbox 360 и 3DS. На 2015-16 года по ним наблюдается снижение продаж и по этому **стоит обратить основное внимание на платформы набирающие обороты в продажах, а именно PS4 и Xbox One**. Предположим, что со временем ситуация будет схожа.
# 
# + Корреляция между оценками пользователей и продажами довольно низкая, а вот между оценками критиков и продажами больше. Так что **при выборе того какую игру ставить на полку магазина стоит большее внимание обращать на _оценку критиков_**.
# 
# + По регионам Северная Америка и Европа лучше всего продавать жанры Action, Shooter, Sports в рейтинге M, E, T с той лишь разницей, что в **Америке** он продается лучше для платформ Xbox, WII, PS3 а в **Европе** для PS3, Xbox, Wii тогда как в **Японии** самый популярные жанры - Role-Playing, Action, Sports в рейтингах - NR, E, T, а самые любимые платформы в Японии - 3DS, PS3, Wii.
# 
# + Гипотезы о том, что: 
#     - средние пользовательские рейтинги платформ Xbox One и PC равны
#     - средние пользовательские рейтинги жанров Action и Sports разные
#     
# Не подтвердились

# <div class="alert-warning">
# <b>Комментарий ревьюера ⚠️</b>
# 
# Вывод в целом хороший. Будет здорово, если добавишь конкретику по регионам: 2-3 консоли, 2-3 жанра и рейтинги. Также не забудь обновить вывод после внесения изменений.</div>

# <div style="border:solid blue 3px; padding: 20px">
# <b>Общий комментарий ревьюера:</b>
# 
# Ты проделал большую работу, но нужно кое-что поправить чтобы я ее принял.
#     
# Есть несколько критических замечаний:
#     
# - по выбору актуального периода для анализа и прогноза
# - по определению среднего жизненного цикла платформ
# - по графику "ящики с усами"
# - по определению корреляций оценок и продаж
# - по оценке прибыльности жанров
# - по играм без рейтинга и портретам игроков различных регионов
# - по проверке гипотез
#     
#     
# Также, постарайся отработать желтые комментарии и улучшить оформление работы: подзаголовки, выводы по шагам.
#     
# Буду ждать твой доработанный проект. ⌛️</div>

# <div style="border:solid blue 3px; padding: 20px">
# <b>Общий комментарий ревьюера V2:</b>
# 
# Спасибо, что доработал проект. С учетом дедлайна, работу принимаю. Но обрати внимание на некоторые замечания и рекомендации с версией V2. Их стоит отработать в финальной версии проекта, прежде чем выкладывать его в портфолио.
# 
# Успехов в дальнейшем обучении! 🤝</div>

# In[ ]:




