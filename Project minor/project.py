import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from IPython import get_ipython
from flask import Flask,request, render_template
import re
import requests as req


#data manipulation
df = pd.read_csv('data.csv',',',error_bad_lines=False) 
df = pd.DataFrame(df)
df = df.sample(n=10000)
col = ['label','url']
df = df[col]
#Deleting nulls
df = df[pd.notnull(df['url'])]

df.columns = ['label', 'url']
df['category_id'] = df['label'].factorize()[0]
category_id_df = df[['label', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'label']].values)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

BAD_len = df[df['label'] == "bad"].shape[0]
GOOD_len = df[df['label'] == "good"].shape[0]
plt.bar(11,BAD_len,2, label="BAD URL")
plt.bar(15,GOOD_len,2, label="GOOD URL")
plt.legend()
plt.ylabel('Number of examples')
plt.title('Proportion of examples')
plt.show()

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
lens = df.url.str.len()
lens.hist(bins = np.arange(0,300,10))
plt.show()



#tokenizer
def getTokens(input):
    tokensBySlash = str(input.encode('utf-8')).split('/')
    allTokens=[]
    for i in tokensBySlash:
        tokens = str(i).split('-')
        tokensByDot = []
        for j in range(0,len(tokens)):
            tempTokens = str(tokens[j]).split('.')
            tokensByDot = tokensByDot + tempTokens
        allTokens = allTokens + tokens + tokensByDot
    allTokens = list(set(allTokens))
    if 'com' in allTokens:
        allTokens.remove('com')
    return allTokens


#TF-IDF
y = [d[1]for d in df] 
myUrls = [d[0]for d in df] 
vectorizer = TfidfVectorizer( tokenizer=getTokens ,use_idf=True, smooth_idf=True, sublinear_tf=False)
features = vectorizer.fit_transform(df.url).toarray()
labels = df.label
features.shape

#model training
model = LogisticRegression(random_state=0)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.20, random_state=0)
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print ("train accuracy =", train_score)
print ("test accuracy =", test_score)

#error detection using heatmap
from sklearn.metrics import confusion_matrix
import seaborn as sns
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=category_id_df.label.values, yticklabels=category_id_df.label.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')


def formaturl(url):
    if not re.match('(?:http|ftp|https)://', url):
        return 'http://{}'.format(url)
    return url




#deploy ml in webpage using flask
app = Flask(__name__)


@app.route('/')
def project():
    return render_template("main.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
  if request.method == "POST": 
    url1=request.form['url']
    X_predict = formaturl(url1)
    
    try:
        response = req.get(X_predict)
        if response.status_code == 200:
            print('Web site exists')
            res = req.head(X_predict,allow_redirects=True)
            expanded = res.url
            print(expanded)
            X_predict = [expanded]
    
    
  
     
            print(X_predict)
    
            X_predict = vectorizer.transform(X_predict)
            y_Predict = model.predict(X_predict)
            probability = model.predict_proba(X_predict)
            output = '{0:.{1}f}'.format(probability[0][1], 2)
            print(y_Predict)

            if y_Predict[0]=="good":
                return render_template('main.html',pred='Url is Safe and Secure with P(G/g) = {}'.format(output),url=expanded,redirection='Click the Redirect button below to go to the site',colour="#00cc66")
            else:
                return render_template('main.html',pred = 'Url is Malicious!!! with P(G/b) = {}'.format(output),url=expanded,redirection='Redirection to the site can be harmfull',colour='#ff3333')

    except req.exceptions.ConnectionError:
            print('Web site does not exist') 
            X_predict = vectorizer.transform([X_predict])
            y_Predict = model.predict(X_predict)
            probability = model.predict_proba(X_predict)
            output = '{0:.{1}f}'.format(probability[0][1], 2)
            if y_Predict[0]=="good":
                return render_template('main.html',pred='Good Url with P(G/g) = {}. Website Does Not Exist!!!'.format(output),url=request.form['url'],colour="#ffbb33")
            else:
                return render_template('main.html',pred = 'Bad Url with P(G/b) = {}. Website Does Not Exist!!!'.format(output),url=request.form['url'],colour='#ffbb33')

        
            
  

if __name__ == '__main__':
    app.run()


