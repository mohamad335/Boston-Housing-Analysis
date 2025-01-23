import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data = pd.read_csv('data/boston.csv', index_col=0)
#how many student are there per teacher on avarage
avrg_student_per_teacher = data['PTRATIO'].mean()
print(f"Average students per teacher: {avrg_student_per_teacher}")
#average price of home in the dataset
avrg_price = data['PRICE'].mean()
print(f"Average price of home: {avrg_price}")
min_value_chas= data['CHAS'].min()
print(f"Minimum value of chas: {min_value_chas}")
max_value_chas= data['CHAS'].max()
print(f"Maximum value of chas: {max_value_chas}")
max_room_per_dwelling = data['RM'].max()
print(f"Maximum number of rooms per dwelling: {max_room_per_dwelling}")
min_room_per_dwelling = data['RM'].min()
print(f"Minimum number of rooms per dwelling: {min_room_per_dwelling}")
# Function to create and save displots
def create_and_save_displots(data, features):
    for feature in features:
        y_axis_titles = {
        'PRICE': 'Frequency of Home Prices',
        'RM': 'Frequency of Rooms per Dwelling',
        'DIS': 'Frequency of Distance to Employment Centers',
        'RAD': 'Frequency of Highway Accessibility Index'
    }
        plot = sns.displot(data[feature], kde=True, aspect=2)
        plot.set_axis_labels(feature, y_axis_titles[feature])
        plot.fig.suptitle(f'Distribution of {feature}', y=1.03)  # Adjust title position
        plot.figsave(f'images/{feature}_distribution.png')
        plt.close()
# Create and save displots
features = ['PRICE', 'RM', 'DIS', 'RAD']

#create a bar chart plotly for CHAST to see how many home away from the river
def chas_bar_chart():
    chas_value=data['CHAS'].value_counts().sort_index()
    fig = px.bar(x=['No','Yes'], 
                y=chas_value, 
                color=chas_value,
                labels={'x': 'Next to Charles River', 'y': 'Number of Homes'}, 
                title='Frequency of CHAS Values')
    fig.write_image('images/CHAS_bar_chart.png')
    fig.show()
chas_bar_chart()