import re # module bieu thuc chinh quy regular expression
import pandas as pd # thu vien xu li du lieu
import nltk # thu vien xu ly ngon ngu tu nhien

dataset = pd.read_csv("dataset3000.csv") # doc file data vao dataframe mang ten dataset
print("Cau truc 1 object email trong dataset: ", dataset.columns) # nhan cac cot cua dataset # Index(['text', 'spam'], dtype='object')
print("kich thuoc dataset truoc khi xoa trung lap: ", dataset.shape) # kich thuoc dataset # (5728, 2)
print("So phan tu null trong dataset: ", pd.DataFrame(dataset.isnull().sum())) # kiem tra tong so phan tu null trong dataset # text  0  spam  0

print("\nTien xu ly...")
dataset.drop_duplicates(inplace = True) # xoa trung lap trong dataset
print("kich thuoc dataset sau khi xoa trung lap: ", dataset.shape) # (5695, 2)

dataset['text']=dataset['text'].map(lambda text: text[9:]) # xoa string "Subject: " dau moi column text cua dataset
dataset['text'] = dataset['text'].map(lambda text:re.sub('[^a-zA-Z]+', ' ',text)).apply(lambda x: (x.lower()).split())
# loai bo dau cau va so, chuyen chu thuong ve chu hoa va tach cac tu trong column text cua dataset va map vao cac list

from nltk.corpus import stopwords # module xu ly stopwords
from nltk.stem.porter import PorterStemmer # module tra ve tu goc
ps = PorterStemmer()
corpus = dataset['text'].apply(lambda text_list:' '.join(list(map(lambda word:ps.stem(word),(list(filter(lambda text:text not in set(stopwords.words('english')),text_list)))))))
# chuyen doi cac tu trong columns text thanh tu goc, loai bo stop word

print("\nGiai doan hoc...")
from sklearn.feature_extraction.text import CountVectorizer # module chuyen document thanh ma tran vector cac tu
cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray() # x = chuyen corpus thanh 1 mang (ma tran) tan suat xuat hien cac tu
y = dataset.iloc[:, 1].values # y = gia tri cua columns 1 (spam) trong dataset

from sklearn.model_selection import train_test_split # module chia array hoac ma tran thanh tap train, test ngau nhien
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, stratify = y) # chia tap train/test theo ti le 7/3 ap dung lay mau phan tang

from sklearn.naive_bayes import MultinomialNB # module phan loai Naive Bayes
classifier = MultinomialNB()
classifier.fit(x_train, y_train) # hoc theo thuat toan Naive Bayes cho tap x_train, y_train

print("\nGiai doan phan lop...")
y_pred = classifier.predict(x_test) # phan loai Naive Bayes cho tap x_test


print("\nTinh toan do chinh xac cua phan lop...")
from sklearn.metrics import confusion_matrix # module tinh toan do chinh xac phan loai
cm = confusion_matrix(y_test, y_pred) # ma tran nham lan

from sklearn.metrics import accuracy_score #module tinh toan do chinh xac
right_rate = accuracy_score(y_test, y_pred)
right_case = accuracy_score(y_test, y_pred,normalize=False)

print("\nResult:")
print("Accuracy: ", right_rate)
print("(" , right_case, "/", y_test.shape[0], " cases)")
precision = cm[0][0]/(cm[0][0]+cm[1][0])
recall = cm[0][0]/(cm[0][0]+cm[0][1])
f1score = 2*precision*recall/(precision+recall)
print("Precision(ham): ", precision)
print("Recall(ham): ", recall)
print("F1 score (ham): ", f1score)