# adapted from Beau Hilton's SO answer here:
# https://stackoverflow.com/questions/49386920/download-kaggle-dataset-by-using-pythonimport requests
import requests
import pandas as pd

def checkdata(df):
    print(df.info())

# Kaggle login & data set URLs
login_url = 'https://www.kaggle.com/account/login'
train_url = 'https://www.kaggle.com/c/titanic/download/train.csv'
test_url = 'https://www.kaggle.com/c/titanic/download/test.csv'

# load Kaggle credentials
kaggle_creds = {}
with open('secrets/kaggle_creds.txt', 'r') as info:
    for line in info:
        kaggle_creds.update(eval(line))

# login to Kaggle and pull train & test data without using Kaggle API
with requests.Session() as c:
    response = c.get(login_url).text
    AFToken = response[response.index('antiForgeryToken')+19:response.index('isAnonymous: ')-12]
    kaggle_creds['__RequestVerificationToken']=AFToken
    c.post(login_url + "?isModal=true&returnUrl=/", data=kaggle_creds)
    download = c.get(train_url)
    decoded_content = download.content.decode('utf-8')

    # write train set to local file 
    f = open('train.csv', 'w')
    for line in decoded_content:
        if line:
            f.write(line)
    f.close()

    download = c.get(test_url)
    decoded_content = download.content.decode('utf-8')

    # write test set to local file 
    f = open('test.csv', 'w')
    for line in decoded_content:
        if line:
            f.write(line)
    f.close()

# check success of data pull
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print('Training data:')
checkdata(train)

print('Testing data:')
checkdata(test)
