import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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
                title='Next charles river ?')
    fig.write_image('images/CHAS_bar_chart.png')
    fig.close()

# Function to create and save jointplots
def create_and_save_jointplots(data, x_feature, y_feature, filename, color):
    with sns.axes_style('darkgrid'):
        plot = sns.jointplot(x=data[x_feature], 
                             y=data[y_feature], 
                             height=4, 
                             kind='scatter',
                             color=color, 
                             joint_kws={'alpha':0.5})
        # Set axis labels
        plot.set_axis_labels(x_feature, y_feature, fontsize=12)
        # Add title to the plot
        plot.fig.suptitle(f'{x_feature} vs {y_feature}', y=1.03, fontsize=14)
        # Adjust the title position
        plot.fig.subplots_adjust(top=0.95)
        # Save the plot
        plot.savefig(f'images/joinplot/{filename}.png')
        plt.close()

# Create and save jointplots
create_and_save_jointplots(data, 'DIS', 'NOX', 'DIS_vs_NOX', 'blue')
create_and_save_jointplots(data, 'INDUS', 'NOX', 'INDUS_vs_NOX', 'green')
create_and_save_jointplots(data, 'LSTAT', 'RM', 'LSTAT_vs_RM', 'red')
create_and_save_jointplots(data, 'LSTAT', 'PRICE', 'LSTAT_vs_PRICE', 'purple')
create_and_save_jointplots(data, 'RM', 'PRICE', 'RM_vs_PRICE', 'orange')
#split training and test dataset
features=data.drop('PRICE', axis=1)
X_train, X_test, y_train, y_test = train_test_split(features, data['PRICE'], test_size=0.2, random_state=10)
# % of training set
train_pct = 100*len(X_train)/len(features)
print(f'Training data is {train_pct:.3}% of the total data.')

# % of test data set
test_pct = 100*X_test.shape[0]/features.shape[0]
print(f'Test data makes up the remaining {test_pct:0.3}%.')

#multivariable regression
regression = LinearRegression()
regression.fit(X_train, y_train)
r_squared = regression.score(X_test, y_test)
print(f'R-squared value of the model: {r_squared:.3}')
#evaluate the coefficient of the model
regression_coef = pd.DataFrame(data=regression.coef_, index=X_train.columns, columns=['Coefficient'])
#premium for having an extra room
extra_room_premium = regression_coef.loc['RM'].values[0]*1000
print(f'Premium for having an extra room in the dwelling: {extra_room_premium:.2f}')
#Analyse the estimate values and Regression Residuals
predict_values=regression.predict(X_train)
residuals = (y_train - predict_values )
# create two scatter plot the first of y_train and the second for residuals vs predict price
def y_train_sactter_plot():
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train, predict_values,c='indigo', alpha=0.5)
    #add a line of best fit
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Price')
    plt.savefig('images/scatter_plotly/y_train_sactter_plot.png')
    plt.close()
    #residuals vs predict price
    plt.figure(figsize=(10, 6))
    plt.scatter(predict_values, residuals,c='indigo', alpha=0.5)
    #add a line of best fit
    plt.plot([predict_values.min(), predict_values.max()], [0, 0], 'r--', lw=2)
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Price')
    plt.savefig('images/scatter_plotly/residuals_vs_predict_price.png')
    plt.close()
y_train_sactter_plot()
#calculate the mean and the skewness of residuals
residuals_mean = round(residuals.mean(),2)
residuals_skew = round(residuals.skew(),2)
print(f'Mean of residuals: {residuals_mean:.2f}')
print(f'Skewness of residuals: {residuals_skew:.2f}')
#use seaborn to displot a histogram of residuals
def residuals_histogram():
    sns.displot(residuals, kde=True, aspect=2)
    plt.xlabel('price')
    plt.ylabel('count')
    plt.title(f'Residuals Skew {residuals_skew} and Mean {residuals_mean}')
    plt.subplots_adjust(top=0.9)
    plt.savefig('images/histogram_plot/residuals_histogram.png')
    plt.close()
residuals_histogram()
#by seaborn use displot to create histogram of price
price_skew=round(data['PRICE'].skew(), 2)
print(f'Skewness of price: {price_skew:.2f}')
def price_histogram():
    sns.displot(data['PRICE'], kde=True, aspect=2)
    plt.xlabel('price')
    plt.ylabel('count')
    plt.title(f'Price Skew {price_skew}')
    plt.subplots_adjust(top=0.9)
    plt.savefig('images/histogram_plot/price_histogram.png')
    plt.close()
price_histogram()
#histogram for log price and modify the skew
log_price = np.log(data['PRICE'])
log_price_skew = round(log_price.skew(), 2)
print(f'Skewness of log price: {log_price_skew:.2f}')
def log_price_histogram():
    sns.displot(log_price, kde=True, aspect=2)
    plt.xlabel('log price')
    plt.ylabel('count')
    plt.title(f'Log Price Skew {log_price_skew}')
    plt.subplots_adjust(top=0.9)
    plt.savefig('images/histogram_plot/log_price_histogram.png')
    plt.close()
#as we see we make the skew better by log the price value it become closer than to zero
log_price_histogram()
#use the log price to train the model
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(features,
                                                                    log_price,
                                                                    test_size=0.2,
                                                                    random_state=10)

regression_log = LinearRegression()
regression_log.fit(X_train_log, y_train_log)
r_squared_log = regression_log.score(X_test_log, y_test_log)
print(f'R-squared value of the model: {r_squared_log:.3}')
#evaluate the coefficient of the model
regression_log_coef = pd.DataFrame(data=regression_log.coef_,
                                    index=X_train_log.columns,
                                      columns=['Coefficient'])
predict_values_log = regression_log.predict(X_train_log)
residuals_log= (y_train_log - predict_values_log)

'''okay before when we plot the residuals histogram we see that the skew was 1.11 is away from zero
so we log the price to make it better and control error '''
def y_train_log_sactter_plot():
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train_log, predict_values_log, c='indigo', alpha=0.5)
    #add a line of best fit
    plt.plot([y_train_log.min(), y_train_log.max()], [y_train_log.min(), y_train_log.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Price')
    plt.savefig('images/scatter_plotly/y_train_log_sactter_plot.png')
    plt.close()
    #residuals vs predict price
    plt.figure(figsize=(10, 6))
    plt.scatter(predict_values_log, residuals_log, c='indigo', alpha=0.5)
    #add a line of best fit
    plt.plot([predict_values_log.min(), predict_values_log.max()], [0, 0], 'r--', lw=2)
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Price')
    plt.savefig('images/scatter_plotly/residuals_log_vs_predict_price.png')
    plt.close()
y_train_log_sactter_plot()
residuals_log_mean = round(residuals_log.mean(), 2)
print(f'Mean of residuals: {residuals_log_mean:.2f}')
residuals_log_skew = round(residuals_log.skew(), 2)
print(f'Skewness of residuals: {residuals_log_skew:.2f}')
#the rsdiuals histogram by seaborn as you see the skew are 0.09 it become more improved than the previous 1.46
def residuals_log_histogram():
    sns.displot(residuals_log, kde=True, aspect=2)
    plt.xlabel('price')
    plt.ylabel('count')
    plt.title(f'Residuals Skew {residuals_log_skew} and Mean {residuals_log_mean}')
    plt.subplots_adjust(top=0.9)
    plt.savefig('images/histogram_plot/residuals_log_histogram.png')
    plt.close()
residuals_log_histogram()








 



