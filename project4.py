#!/usr/bin/env python
# coding: utf-8

# ## Проект:
# ### Выделение определяющих успешность игры закономерностей

# #### Задача:
# 
# _Выявить определяющие успешность игры закономерности._
# 
# Интернет-магазин «Стримчик» продаёт по всему миру компьютерные игры. Из открытых источников доступны исторические данные о продажах игр, оценки пользователей и экспертов, жанры и платформы (например, Xbox или PlayStation). Нужно выявить определяющие успешность игры закономерности. Это позволит сделать ставку на потенциально популярный продукт и спланировать рекламные кампании.
# 
# 
# Дана таблица с данными до 2016 года. Представим, что сейчас декабрь 2016 г., и необходимо спланировать кампанию на 2017-й. Нужно отработать принцип работы с данными. Не важно, прогнозируются продажи на 2017 год по данным 2016-го или же 2027-й — по данным 2026 года.
# 
# В наборе данных попадается аббревиатура ESRB (Entertainment Software Rating Board) — это ассоциация, определяющая возрастной рейтинг компьютерных игр. ESRB оценивает игровой контент и присваивает ему подходящую возрастную категорию, например, «Для взрослых», «Для детей младшего возраста» или «Для подростков».

# #### Описание входных данных:
# 
# - Name — название игры
# - Platform — платформа
# - Year_of_Release — год выпуска
# - Genre — жанр игры
# - NA_sales — продажи в Северной Америке (миллионы долларов)
# - EU_sales — продажи в Европе (миллионы долларов)
# - JP_sales — продажи в Японии (миллионы долларов)
# - Other_sales — продажи в других странах (миллионы долларов)
# - Critic_Score — оценка критиков (максимум 100)
# - User_Score — оценка пользователей (максимум 10)
# - Rating — рейтинг от организации ESRB (англ. Entertainment Software Rating Board). Эта ассоциация определяет рейтинг компьютерных игр и присваивает им подходящую возрастную категорию.
# 
# Путь к файлу:
# /datasets/games.csv

# ### Шаг 1. Изучение данных.

# In[159]:


import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st

games_df = pd.read_csv('/datasets/games.csv')

print (games_df.info())
games_df.head()


# Всего в таблице 11 колонок и 16714 записей.
# 
# Порядок преобразования колонок:
# - Обработать имена колонок, приведя их к нижнему регистру.
# - **name** - удалить пропуски
# - **year** - перевести в int16, заполнить пропуски
# - **critic_score** - заполнить пропуски
# - **user_score** - заполнить пропуски
# - **rating** - заполнить пропуски
# - **genre** - изменить тип на "category"
# - **platform** - изменить тип на "category"
# - **rating** - изменить тип на "category"
# 

# ### Шаг 2: предобработка данных

# Заменим названия колонок:

# In[160]:


games_df.columns = ['name', 'platform', 'year', 'genre', 'na_sales', 'eu_sales', 'jp_sales', 'other_sales', 'critic_score', 'user_score', 'rating']
games_df.head()


# In[161]:


games_df[games_df['name'].isna()]


# Удаляем пропуски в name - если не знаем какая игра и какой жанр как оценивать?
# Так же и с year - наши расчеты базирются именно на временных отрезках,

# In[162]:


games_df.dropna(subset=['name', 'year'], inplace=True)
games_df.reset_index(drop=True, inplace=True)
games_df[games_df['name'].isna()]


# Переводим колонку "год" в int16:

# In[163]:


games_df['year'] = pd.to_datetime(games_df['year'].astype(int), format='%Y').dt.year.astype('int16')
print (games_df.info())


# Проверяем, есть ли оценка "0" среди пользовательской оценки и оценки критиков:

# In[164]:


games_df['critic_score'].value_counts().reset_index().query('index == 0.0')


# In[165]:


games_df['user_score'].value_counts().reset_index().query('index == 0.0')


# Нулей нет, соответственно пропуски можем заполнить нулями - пока игру никто не оценил, её оценка "0"

# In[166]:


games_df['critic_score'] = games_df['critic_score'].fillna(0.0)
games_df['user_score'] = games_df['user_score'].fillna(0.0)
games_df[games_df['user_score'] == 'tbd']


# In[167]:


games_df.loc[games_df['user_score'] == 'tbd', 'user_score'] = 0.0
games_df['user_score'] = games_df['user_score'].astype('float')
games_df.info()


# Просмотрим список всех значений поля "rating"

# In[168]:


games_df['rating'].value_counts()


# Википедия дает следующую расшифровку данных:
# 
# «EC» («Early childhood») — «Для детей младшего возраста»<br>
# «E» («Everyone») — «Для всех» ("K-A" ("Kids to Adults"))<br>
# «E10+» («Everyone 10 and older») — «Для всех от 10 лет и старше»<br>
# «T» («Teen») — «Подросткам»<br>
# «M» («Mature») — «Для взрослых»<br>
# «AO» («Adults Only 18+») — «Только для взрослых»<br>
# «RP» («Rating Pending») — «Рейтинг ожидается»<br>
# 
# Заменим устаревшую аббревиатуру "K-A" на современую "EC":

# In[169]:


games_df.loc[games_df['rating'] == 'K-A', 'rating'] = 'E'
games_df['rating'].value_counts()


# Нам нужно оценить продажи по играм, учитывая рейтинг. Соответственно нет смысла оценивать те игры, чей рейтинг только ожидается. Удалим такие игры:

# In[170]:


games_df.drop(games_df[games_df['rating'] == 'RP'].index, inplace=True)
games_df['rating'].value_counts()


# Заполним пропуски в возрастном рейтинге типом other - чтобы потом можно было легко отбросить эту долю

# In[171]:


games_df['rating'] = games_df['rating'].fillna('other')
print (games_df['rating'].value_counts())
print (games_df.info())


# Пропусков нет, посмотрим, есть ли дубли в данных:

# In[172]:


print (games_df.duplicated().sum(), '\n')
print (games_df['genre'].value_counts(), '\n')
print (games_df['platform'].value_counts())


# Дублей тоже нет, все норм.
# 
# Уменьшим потребляемый объем памяти, переведя все возможные object в category:

# In[173]:


games_df['platform'] = games_df['platform'].astype('category')
games_df['genre'] = games_df['genre'].astype('category')
games_df['rating'] = games_df['rating'].astype('category')
print (games_df.info())


# Посчитаем все продажи:

# In[174]:


games_df['all_sales'] = games_df['other_sales'] + games_df['jp_sales'] + games_df['eu_sales'] + games_df['na_sales']
games_df.head()


# ### Шаг 3: Исследовательский анализ данных

# In[175]:


games_df['year'].sort_values().unique()


# Задача стоит следующая: "Найти определяющие успешность игры закономерности". 
# Думаю, данные до 2000 года можно отбросить - эра компьютеров и игр, которые могут влиять вцелом на рейтинг, началась после 2000 года. 
# Года до 2000 были прекрасны, но это была другая эра.

# In[176]:


def print_line_plot_by_year(year, top):
    """ Построение линейного графика, с фильтром по домам и топ-значениям """
    games_df_filtered = games_df.query('year >= @year')
    games_df_top_sale = pd.pivot_table(games_df_filtered, index=['year', 'platform'], values='all_sales', aggfunc='sum', fill_value=0)
    games_df_top_sale.reset_index(inplace=True)
    
    top_platform = games_df_top_sale.groupby(['platform']).agg(sales=('all_sales', 'sum')).sort_values(by='sales', ascending=False).head(top).index.tolist()
    
    games_df_top_sale = games_df_top_sale.query('platform in @top_platform')
    games_df_top_sale['platform'].cat.remove_unused_categories()
    
    sns.set(font_scale=1.6)
    sns.relplot(
        x='year', 
        y='all_sales', 
        hue='platform', 
        kind="line", 
        marker='o', 
        hue_order=top_platform, 
        palette='Paired', 
        data=games_df_top_sale, 
        height=11, 
        aspect=1.5
    )
    plt.show()

    return top_platform

print_line_plot_by_year(2000, 14)


# На графике видно, как менялись платформы и пользователи с течением времени.
# Напрмер, PS2 была самой популярной платформой до 2005 года, потом продажи резко рухнули - пришел X360 и PS3. 
# С 2013 года начала развиваться PS4, но так и не достигла популярности своих предков. 
# Вообще, к 2016 году продажи по всем направлениям упали, PS4 еще вроде держится на пошатывающейся ножке. Примечательно то, что PC не входит в ТОП10, но входит в топ 15
# и занимает примерно равные позиции на протяжении всего временного отрезка. PC всегда были и всегда будут. Так же, как и пираты, которые не покупают игры на PC, а просто
# скачивают их с торрента. 
# 
# 5 лет - примерно за такой срок появляются новые платформы и исчезают старые. Т.е., чтобы грамотно проанализировать рынок, нужно взять данные с 2011 года, по 2016.

# In[177]:


top_platforms = print_line_plot_by_year(2011, 10)
top_platforms


# Не назвал бы данные очень веселыми и утешительными. Хьюстон, у нас упали продажи!!
# Из данных графика видно, что максимальные продажи у PS4. Так же, если вспомним исторические данные, то у PS всегда были топовые продажи, которые падали только с выходом новой PS.
# C 2015 по 2016 год наблюдается рост выручки у платформы "3DS". Так же, неплохие продажи были у PSP, но в 2015 году данные о продажах проподают. Видимо, платформа сильно устарела.
# В принципе, все приставки распродаются, как горячие пончики. Есть смысл оставить семейство PS для дальнейшего изучения, семейство XBox, обязательно-стабильный PC, в общем отбираем топ-10.
# 
# Построим диаграмму размаха продаж по этим платформам:

# In[178]:


prepared_df = games_df.query('year >= 2011 and platform in @top_platforms')
whisker_plot_df = prepared_df[['year', 'platform', 'all_sales']]
whisker_plot_df


# In[179]:


def print_whisker_plot(x_column, y_column, df, order_list, ylim):
    whisker_plot = sns.catplot(
        kind='box',
        y=x_column, 
        x=y_column, 
        palette='Blues', 
        data=df, 
        order=order_list,        
        height=11, 
        aspect=1.5
    )
    whisker_plot.set(ylim=ylim)
    plt.show()
    
print_whisker_plot('all_sales', 'platform', whisker_plot_df, top_platforms, (0, 3))


# Мы видим на диаграмме, что правый ус длиннее максимума, у каждого из ящиков, соответственно, много выбросов в сторону максимума. Данный график поможет нам установить среднее значение без этих выбросов.
# 
# Итак, максимально прибыльные платформы:
# xbox 360, xbox One, PS3-4 и WiiU
# 
# Менее прибыльные:
# 
# Wii, 3DS, PC, DS, PSV

# ---
# 
# Посмотрим, как влияют на продажи внутри одной популярной платформы отзывы пользователей и критиков. Для этого построим диаграмму рассеяния и посчитаем корреляцию между отзывами и продажами для платформы "X360":

# In[180]:


scatter_df = prepared_df.query('platform == "X360"')[['all_sales', 'critic_score', 'user_score']]

def print_scatter_matrix(count_params, df, x = None, y = None):
    if count_params == 1:
        g = sns.jointplot(
            'all_sales', 
            'critic_score', 
            data=df,
            color='#4CB391'
        )
        plt.show()
        
    elif count_params == 2:
        pd.plotting.scatter_matrix(df, figsize=(16, 16))
        plt.show()
        
    else:
        return False

print_scatter_matrix(1, scatter_df, 'all_sales', 'critic_score')
print_scatter_matrix(1, scatter_df, 'all_sales', 'user_score')


# По данным графикам видно, что зависимость отзывов от продаж есть, но небольшая. Причем продажи зависят от пользовательских отзывов больше, чем от отзывов критиков. Но все равно, зависимость незначительная.
# 
# Чтобы подтвердить предположение, построим попарную диаграмму расеивания:
# 

# In[181]:


print_scatter_matrix(2, scatter_df)


# Предыдущие выводы подтведились. Количество продаж особо не зависит ни от оценки пользователей, ни от оценки критиков. Маркетологи рулят продажами.
# 
# Построим эти же графики для других платформ, менее популярных:

# In[182]:


scatter_df = prepared_df.query('platform != "X360"')[['all_sales', 'critic_score', 'user_score']]
print_scatter_matrix(1, scatter_df, 'all_sales', 'critic_score')
print_scatter_matrix(1, scatter_df, 'all_sales', 'user_score')
print_scatter_matrix(2, scatter_df)


# По остальным платформам графики выглядят так же. Небольшая зависимость есть между user_score, critic_score и all_sales. Но её не нужно учитывать.

# In[183]:


all_sales_by_platform = pd.pivot_table(
        prepared_df, 
        index='genre',
        values=['all_sales'],
        aggfunc='sum'
    )

indexNames = all_sales_by_platform[all_sales_by_platform['all_sales'] <= 0].index
all_sales_by_platform.drop(indexNames , inplace=True)
top_genres = all_sales_by_platform.index
all_sales_by_platform.plot(kind='bar', y='all_sales', figsize=(10, 8))
plt.show()


# По графику видно, что самые прибыльные жанры это Action и Shooter. На втором месте Role-Playing и Sports.
# 
# ### Шаг 4. Составить портрет пользователя каждого региона
# В данном блоке необходимо для пользователя каждого региона (NA, EU, JP):
# - Определить самые популярные платформы (топ-5). 
# - Описать различия в долях продаж.
# - Определить самые популярные жанры (топ-5). 
# - Поясните разницу.
# - Ответить на вопрос "Влияет ли рейтинг ESRB на продажи в отдельном регионе?"

# Разбиваем продажи на лидеров - чтобы понять, какая игра на какой платформе в каком регионе лидирует:

# In[184]:


def get_leader_region(row):
    na_sales = row[4]
    eu_sales = row[5]
    jp_sales = row[6]
    
    if eu_sales < na_sales > jp_sales:
        return 'na'
    elif na_sales < eu_sales > jp_sales:
        return 'eu'
    return 'jp'

prepared_df['leader_region'] = prepared_df.agg(get_leader_region, axis=1)
prepared_df


# Посмотрим разницу в продажах по платформам:

# In[185]:


def print_barplot_by_region(df, x, y, category, height, order=None, rotate=False):
    """  """
    g = sns.catplot(
        x=x, 
        y=y, 
        col=category,
        data=df, 
        saturation=.5,
        kind="bar", 
        ci=None, 
        aspect=.6,
        order=order,
        height=height
    )
    if rotate == True:
        [plt.setp(
            ax.get_xticklabels(), 
            rotation=45,
            horizontalalignment='right',
            fontweight='light',
            fontsize='x-large'
        ) for ax in g.axes.flat]
    plt.show()
    
print_barplot_by_region(prepared_df, 'platform', 'all_sales', 'leader_region', 11, top_platforms)


# Посмотрим разницу в продажах по жанрам игр:

# In[186]:


print_barplot_by_region(prepared_df, 'genre', 'all_sales', 'leader_region', 15, None, True)


# Так же посмотрим общую информацию в сводной таблице по продажам:

# In[187]:


pivot_by_region = pd.pivot_table(
    prepared_df,
    index=['leader_region', 'platform'],
    values=['all_sales'],
    aggfunc=['sum']
)

pivot_by_region.reset_index(inplace=True)
pivot_by_region.columns=['leader_region', 'platform', 'all_sales']
pivot_by_region.sort_values(by=['leader_region', 'all_sales'], ascending=False, inplace=True)
pivot_by_region


# Посмотрим, зависит ли рейтинг ESRB от количества продаж по каждому региону:

# In[188]:


scatter_df_na = prepared_df.query('leader_region == "na"')[['rating', 'all_sales']]
scatter_df_eu = prepared_df.query('leader_region == "eu"')[['rating', 'all_sales']]
scatter_df_jp = prepared_df.query('leader_region == "jp"')[['rating', 'all_sales']]
scatter_df_na['rating_code'] = scatter_df_na.query('rating != "other"')['rating'].cat.codes
scatter_df_eu['rating_code'] = scatter_df_eu.query('rating != "other"')['rating'].cat.codes
scatter_df_jp['rating_code'] = scatter_df_jp.query('rating != "other"')['rating'].cat.codes

print (f'Европа: \n{scatter_df_eu.corr()}\n')
print (f'Северная Америка: \n{scatter_df_na.corr()}\n')
print (f'Япония: \n{scatter_df_jp.corr()}')


# Из таблиц корреляции видно, что есть незначительное влияение продаж от рейтинга только у Японии. У Европы влияние меньше, и у Америки еще меньше.

# По результатам исследования данных образовалась следующая картина:
# 
# #### Europe
# Предпочтения по платформам:
# - PS4 (225.76 млн)
# - PS3 (213.53 млн)
# 
# Предпочтения по жанрам:
# - Shooter
# - Role-Playing
# - Sports
# - Action
# 
# Влияет ли рейтинг ESRB на продажи?
# - Практически не влияет
# 
# #### North America
# Предпочтения по платформам:
# - X360 (338.59 млн)
# - PS3 (181.03 млн)
# - 3DS (146.20 млн)
# - XOne (137.95 млн)
# 
# Предпочтения по жанрам:
# - Shooter
# - Role-Playing
# - Platform
# - Action
# 
# Влияет ли рейтинг ESRB на продажи?
# - Не влияет
# 
# #### Japan
# Предпочтения по платформам:
# - 3DS (86.32 млн)
# - PS3 (51.01 млн)
# - DS (12.24)
# - PSV (28.35)
# - Wii (8.73 млн)
# 
# Предпочтения по жанрам:
# - Simulation
# - Platform
# - Role-Playing
# 
# Влияет ли рейтинг ESRB на продажи?
# - Влияет незначительно
# 
# Интересно то, что в Японии популярностью пользуются одни платформы, а выручку приносят немного другие.
# 
# ---
# 
# Общие продажи по регионам:

# In[189]:


def get_all_sales_by_region(df, region):
    return df.query('leader_region == @region')['all_sales'].sum()

print (f'Всего продаж в Северной Америке: {get_all_sales_by_region(pivot_by_region, "na"):.2f}')
print (f'Всего продаж в Европе: {get_all_sales_by_region(pivot_by_region, "eu"):.2f}')
print (f'Всего продаж в Японии: {get_all_sales_by_region(pivot_by_region, "jp"):.2f}')
       


# ### Шаг 5. Проверка гипотез
# 
# Необходимо проверить 2 гипотезы:
# - Средние пользовательские рейтинги платформ Xbox One и PC одинаковые.
# - Средние пользовательские рейтинги жанров Action (англ. «действие») и Sports (англ. «виды спорта») разные.

# Для проверки гипотез будем использовать инди-тест, который проверяет равенство средних двух генеральных совокупностей. Нулевая гипотеза - параметры равны, альтернативная - параметры различаются. Проверка производится с помощью метода _ttest_ind()_ библиотеки _stats_. полученное значение p-value будем сравнивать с alpha, которое по стандарту зададим, как .05
# 
# #### Гипотеза 1:
# - Нулевая гипотеза:
#     - Средние пользовательские рейтинги платформ Xbox One и PC одинаковые.
# - Альтернативная гипотеза:
#     - Средние пользовательские рейтинги платформ Xbox One и PC различаются.

# In[190]:


alpha = .05

results = st.ttest_ind(
    prepared_df.query('platform == "XOne"')['user_score'],
    prepared_df.query('platform == "PC"')['user_score'],
)

print ('p-значение:', results.pvalue)

if results.pvalue < alpha:
    print ('\nОтвергаем гипотезу.')
else:
    print ('\nГипотеза прошла проверку.')


# #### Гипотеза 2:
# - Нулевая гипотеза:
#     - Средние пользовательские рейтинги жанров Action и Sports одинаковые.
# - Альтернативная гипотеза:
#     - Средние пользовательские рейтинги жанров Action и Sports разные.

# In[191]:


results = st.ttest_ind(
    prepared_df.query('genre == "Action"')['user_score'],
    prepared_df.query('genre == "Sports"')['user_score'],
)

print ('p-значение:', results.pvalue)

if results.pvalue < alpha:
    print ('\nОтвергаем гипотезу.')
else:
    print ('\nГипотеза прошла проверку.')
    


# Итого: <br>
# Средние пользовательские рейтинги платформ Xbox One и PC одинаковые.<br>
# Средние пользовательские рейтинги жанров Action (англ. «действие») и Sports (англ. «виды спорта») одинаковые - причем, судя по тому, как p-value стремится к 1, средние рейтинги очень похожи.

# ### Общий вывод
# 
# #### В данной работе изучены зависимости продаж игр, от платформы, жанра, года, оценок и рейтинга.
# 
# Выполнено изучение данных:
# - Изучены представленные данные таблицы.
# - Определена дальнейшая работа по предобработке данных.
# 
# Выполнена предобработка данных:
# - Колонки переименовоны в удобочитаемый формат.
# - Часть пропусков удалена, часть обработана.
# - Все колонки приведены к нужному типу.
# - Посчитаны общие продажи в колонке _all_sales_.
# - Проверены данные на дубликаты.
# 
# Выполнен исследовательский анализ данных:
# - Проанализированы продажи игр по годам и платформам.
# - Определен период жизни платформы - 5 лет.
# - Подготовлены данные для дальнйшей работы - только последние 5 лет таблицы и только платформы, которые входят в топ 10 по продажам.
# - Выделено среднее значение по продажам каждой из платформ.
# - Найдена самая прибыльная платформа - X360.
# - Проанализированы зависимости отзывов пользователей и критиков от продаж.
# - Выделены самые прибыльные жанры по платформам.
# 
# Составлен портрет пользователей каждого региона в предпочтении игр и платформ.
# 
# Проверены гипотезы:
# - Средние пользовательские рейтинги платформ Xbox One и PC одинаковые.
#     - Гипотеза прошла проверку.
# - Средние пользовательские рейтинги жанров Action (англ. «действие») и Sports (англ. «виды спорта») разные.
#     - Гипотеза не прошла проверку, средние пользовательские рейтинги практически одинаковые, p-value стремится к 1.
#     
# #### Итого:
# 
# Всего продаж в Северной Америке: 1079.23<br>
# Всего продаж в Европе: 641.57<br>
# Всего продаж в Японии: 203.81<br>
# 
# Чтобы скорректировать бюджет на следующий год необходимо:
# - Брать данные для анализа за последние 5 лет.
# - Делать упор на топ-платформы. В нашем случае это платформы семейства X-Box и PS.
# - Делать упор на прибыльные и популярные жанры. В нашем случае это Action, Shooter, Role-Playing и Sports.
# - Для разных регионов рекламировать свои игры и свои платформы:
#     - Европа:
#         - PS3, PS4
#         - Shooter, Role-Playing, Sports, Action
#     - Северная Америка:
#         - X360, PS3, 3DS, XOne
#         - Shooter, Role-Playing, Platform, Action
#     - Япония:
#         - 3DS, PS3, DS, PSV, Wii
#         - Simulation, Platform, Role-Playing
# - Для Японии нужно учитывать ESRB рейтинг игр.
# - Средние пользовательские рейтинги платформ Xbox One и PC одинаковые, соответственно нужно учитывать настроение общества и не разлеять эти платформы при построении бюджета.
# - Средние пользовательские рейтинги жанров Action и Sports  одинаковые, соответственно нужно учитывать настроение общества и не разлеять эти платформы при построении бюджета.
# 

# In[ ]:




