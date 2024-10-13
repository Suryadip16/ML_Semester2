import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.metrics import classification_report

# import data
spam_df = pd.read_csv("spam_sms.csv")

#inspect data

print(spam_df)
# print(spam_df.columns)
# print(spam_df.groupby('v1').describe())

#Label Encoder
spam_y = spam_df['v1']
le = LabelEncoder()
le.fit(spam_y)
spam_y = le.transform(spam_y)
print(spam_y)

spam_x = spam_df['v2']

#create train-test split

x_train, x_test, y_train, y_test = train_test_split(spam_x, spam_y, test_size=0.3, shuffle=True, random_state=42)

#Count Vectorizer: Find word count and store data as a numerical vector

cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train.values).toarray()
x_test_count = cv.transform(x_test.values).toarray()
print(x_train_count)
print(x_test_count)

#Model Building

mnb = MultinomialNB()
gnb = GaussianNB()
cnb = ComplementNB()
clf = svm.SVC(kernel="rbf")

#Multinomial NB
mnb.fit(x_train_count, y_train)
y_pred_mnb = mnb.predict(x_test_count)
mnb_acc = accuracy_score(y_pred_mnb, y_test)
print(mnb_acc)

#gaussian NB
gnb.fit(x_train_count, y_train)
y_pred_gnb = gnb.predict(x_test_count)
gnb_acc = accuracy_score(y_pred_gnb, y_test)
print(gnb_acc)

#Complement NB
cnb.fit(x_train_count, y_train)
y_pred_cnb = cnb.predict(x_test_count)
cnb_acc = accuracy_score(y_pred_cnb, y_test)
print(cnb_acc)

#SVM Classifier
clf.fit(x_train_count, y_train)
y_pred_svm = clf.predict(x_test_count)
svm_acc = accuracy_score(y_pred_svm, y_test)
print(svm_acc)



#Testing with External data
# Phishing Scams
phishing_message_1 = "Your account has been compromised. Click here to reset your password."
phishing_message_2 = "You've won a free vacation! Claim your prize now."

# Fake Offers and Promotions
fake_offer_1 = "Get rich quick! Earn $1000 per day working from home."
fake_offer_2 = "Limited-time offer: Buy one, get one free!"

# Financial Scams
financial_scam_1 = "Invest in this amazing opportunity with guaranteed returns."
financial_scam_2 = "Urgent: Your bank account needs verification. Provide your details now."

# Health and Wellness Products
health_product_1 = "Miracle weight loss pill! Shed pounds in days."
health_product_2 = "New breakthrough supplement boosts energy and vitality."

# Unsolicited Advertisements
advertisement_1 = "Buy cheap prescription drugs online without a prescription."
advertisement_2 = "Enlarge your [body part] with our revolutionary product."

# Malicious Links and Attachments
malicious_link_1 = "Open this document to view important information."
malicious_link_2 = "Click here to track your package delivery status."

# List of spam messages
spam_messages = [
    phishing_message_1,
    phishing_message_2,
    fake_offer_1,
    fake_offer_2,
    financial_scam_1,
    financial_scam_2,
    health_product_1,
    health_product_2,
    advertisement_1,
    advertisement_2,
    malicious_link_1,
    malicious_link_2
]
# Normal (Non-Spam) Messages
normal_message_1 = "Hey, how are you doing?"
normal_message_2 = "Reminder: Our meeting is at 3 PM today."
normal_message_3 = "I'll pick up groceries on my way home."
normal_message_4 = "Congratulations on your promotion!"
normal_message_5 = "The weather is beautiful today."

# Friendly Conversations
conversation_1 = "Let's catch up over coffee sometime!"
conversation_2 = "What are your plans for the weekend?"

# Professional Communications
professional_message_1 = "Please review the attached document for our discussion."
professional_message_2 = "Thank you for your prompt response."

# Personal Updates
personal_update_1 = "I finished reading that book you recommended."
personal_update_2 = "Just wanted to say hello and check in."

# List of normal messages
normal_messages = [
    normal_message_1,
    normal_message_2,
    normal_message_3,
    normal_message_4,
    normal_message_5,
    conversation_1,
    conversation_2,
    professional_message_1,
    professional_message_2,
    personal_update_1,
    personal_update_2
]

spam_messages_count = cv.transform(spam_messages)
normal_messages_count = cv.transform(normal_messages)

y_pred_cnb_ext_spam = cnb.predict(spam_messages_count)
y_pred_cnb_ext_ham = cnb.predict(normal_messages_count)
print(y_pred_cnb_ext_spam)
print(y_pred_cnb_ext_ham)








