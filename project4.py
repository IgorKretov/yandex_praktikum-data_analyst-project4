import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

""" Шаг1 - изучение данных """
""" 
«EC» («Early childhood») — «Для детей младшего возраста»
«E» («Everyone») — «Для всех» ("K-A" ("Kids to Adults"))
«E10+» («Everyone 10 and older») — «Для всех от 10 лет и старше»
«T» («Teen») — «Подросткам»
«M» («Mature») — «Для взрослых»
«AO» («Adults Only 18+») — «Только для взрослых»
«RP» («Rating Pending») — «Рейтинг ожидается»

tbd - To Be Determined - To help ensure that METASCORES accurately reflect the reviews given by critics for any particular movie, game, television show or album, 
we do not display a METASCORE for those items that do not have at least four (4) reviews in our database. Once this minimum number of reviews is reached, 
the METASCORE will display.
"""

games_df = pd.read_csv('datasets/games.csv', encoding='utf-8', index_col=0)

print (games_df.info())
print (games_df.head())

""" 
Описываем данные:

названия столбцов...

Name - удалить пропуски
Year - в datetime, заполнить пропуски
criti_score,
user_score
rating - заполнить пропуски

переименовать колонки

Шаг 2: предобработка данных
"""

games_df.columns = ['name', 'platform', 'year', 'genre', 'na_sales', 'eu_sales', 'jp_sales', 'other_sales', 'critic_score', 'user_score', 'rating']
print (games_df.head())

""" Удаляем пропуски в name - если не знаем какая игра, как оценивать? """

print (games_df[games_df['name'].isna()])
games_df.dropna(subset=['name', 'year'], inplace=True)
games_df.reset_index(drop=True, inplace=True)
print (games_df[games_df['name'].isna()])

games_df['year'] = pd.to_datetime(games_df['year'].astype(int), format='%Y').dt.year.astype('int16')

print (games_df.info())

print (games_df['critic_score'].value_counts().reset_index().query('index == 0.0'))
print (games_df['user_score'].value_counts().reset_index().query('index == 0.0'))
games_df['critic_score'] = games_df['critic_score'].fillna(0.0)
games_df['user_score'] = games_df['user_score'].fillna(0.0)
print (games_df[games_df['user_score'] == 'tbd'])
games_df.loc[games_df['user_score'] == 'tbd', 'user_score'] = 0.0
games_df['user_score'] = games_df['user_score'].astype('float')
print (games_df.info())

""" Нулей нет, поэтому NAN-ы заменим нулями. Пока игру никто не оценил - её рейтинг 0 """

print (games_df['rating'].value_counts())
""" Заменим устаревший K-A на новый EC """
games_df.loc[games_df['rating'] == 'K-A', 'rating'] = 'E'
print (games_df['rating'].value_counts())
""" Нам нужно оценить продажи по играм, соответственно нет смысла оценивать те игры, чей рейтинг только ожидается. """
games_df.drop(games_df[games_df['rating'] == 'RP'].index, inplace=True)
print (games_df['rating'].value_counts())
""" Заполним пропуски в возрастном рейтинге типом other - чтобы потом можно было легко отбросить эту долю """
games_df['rating'] = games_df['rating'].fillna('other')
print (games_df['rating'].value_counts())
print (games_df.info())
""" Пропусков нет, посмотрим, есть ли дубли """
print (games_df.duplicated().sum())
print (games_df['genre'].value_counts())
print (games_df['platform'].value_counts())
""" Уменьшим потребляемый объем памяти, переведя все возможные object в category """
games_df['platform'] = games_df['platform'].astype('category')
games_df['genre'] = games_df['genre'].astype('category')
games_df['rating'] = games_df['rating'].astype('category')
print (games_df.info())
""" Дублей тоже нет, все норм 
Посчитаем все продажи:
"""
games_df['all_sales'] = games_df['other_sales'] + games_df['jp_sales'] + games_df['eu_sales'] + games_df['na_sales']
print (games_df.head())

""" Шаг 3: Исследовательский анализ данных """
print (games_df['year'].sort_values().unique())

""" В задаче стоит "Определяющие успешность игры закономерности". 
Думаю, данные до 2000 года можно отбросить - эра компьютеров и игр, которые могут влиять вцелом на рейтинг, началась после 2000 года. 
Года до 2000 были прекрасны, но это была другая эра
"""

def line_plot_by_year(year, top):
    """  """
    games_df_filtered = games_df.query('year >= @year')
    games_df_top_sale = pd.pivot_table(games_df_filtered, index=['year', 'platform'], values='all_sales', aggfunc='sum', fill_value=0)
    games_df_top_sale.reset_index(inplace=True)
    top_platform = games_df_top_sale.groupby(['platform']).agg(sales=('all_sales', 'sum')).sort_values(by='sales', ascending=False).head(top).index.tolist()
    print (top_platform)
    games_df_top_sale = games_df_top_sale.query('platform in @top_platform')
    games_df_top_sale['platform'].cat.remove_unused_categories()
    print (games_df_top_sale)
    print (games_df_top_sale['platform'].nunique())

    sns.relplot(x='year', y='all_sales', hue='platform', kind="line", marker='o', hue_order=top_platform, palette='Paired', data=games_df_top_sale)
    plt.figure(figsize=(16, 6))
    plt.show()

    return top_platform

# line_plot_by_year(2000, 14)
top_platforms = line_plot_by_year(2011, 10)
# print (top_platforms)

""" 
На графике видно, как менялись платформы и пользователи с течением времени.
Напрмер, PS2 была самой популярной платформой до 2005 года, потом продажи резко рухнули - пришел X360 и PS3. 
С 2013 года начала развиваться PS4, но так и не достигла популярности своих предков. 
Вообще, к 2016 году продажи по всем направлениям упали, PS4 еще вроде держится на пошатывающейся ножке. Примечательно то, что PC не входит в ТОП10, но входит в топ 15
и занимает примерно равные позиции на протяжении всего временного отрезка. PC всегда были и всегда будут. Так же, как и пираты, которые не покупают игры на PC, а просто
скачивают их с торрента. 

5 лет - примерно за такой срок появляются новые платформы и исчезают старые. Т.е., чтобы грамотно проанализировать рынок, нужно взять данные с 2011 года, по 2016.
"""

# games_df_filtered = games_df.query('year >= 2011')
# games_df_top_sale = pd.pivot_table(games_df_filtered, index=['year', 'platform'], values='all_sales', aggfunc='sum', fill_value=0)
# games_df_top_sale.reset_index(inplace=True)
# top_platform = games_df_filtered.groupby(['platform']).agg(sales=('all_sales', 'sum')).sort_values(by='sales', ascending=False).head(14).index.tolist()
# games_df_filtered = games_df_filtered.query('platform in @top_platform')
# games_df_filtered['platform'].cat.remove_unused_categories()
# sns.relplot(x='year', y='all_sales', hue='platform', kind="line", marker='o', palette='Paired', data=games_df_filtered)
# plt.show()

""" 
Не назвал бы данные очень веселыми и утешительными. Хьюстон, у нас упали продажи!!
Из данных графика видно, что максимальные продажи у PS4. Так же, если вспомним исторические данные, то у PS всегда были топовые продажи, которые падали только с выходом новой PS.
C 2015 по 2016 год наблюдается рост выручки у платформы "3DS". Так же, неплохие продажи были у PSP, но в 2015 году данные о продажах проподают. Видимо, платформа сильно устарела.
В принципе, все приставки распродаются, как горячие пончики. Есть смысл оставить семейство PS для дальнейшего изучения, семейство XBox, обязательно-стабильный PC, в общем отбираем топ-10.

Построим диаграмму размаха продаж по этим платформам:
"""

whisker_plot_df = games_df.query('year >= 2011 and platform in @top_platforms')[['year', 'platform', 'all_sales']]
# print (whisker_plot_df)
# whisker_plot = sns.boxplot(y='all_sales', x='platform', palette='Blues', data=whisker_plot_df, order=top_platforms)
# whisker_plot.set(ylim=(0, 4))
# plt.show()

""" Мы видим на диаграмме, что правый ус длиннее максимума, соответственно, много выбросов в сторону максимума - график помогает установить среднее значение
без этих выбросов.
Итак, максимально прибыльные платформы:
xbox 360, xbox One, PS3-4 и WiiU
"""

sales_by_platform = whisker_plot_df[['all_sales', 'platform']]

sales_by_platform.groupby('platform').reset_index().plot(kind='bar', y='all_sales')
plt.show()