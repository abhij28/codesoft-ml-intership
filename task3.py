import pandas as pd


train_data = pd.read_csv("train_data.txt", delimiter='\t')
test_data = pd.read_csv("test_data.txt", delimiter='\t')

print(train_data.head())
print(test_data.head())


from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import re


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text



print("Training data columns:", train_data.columns)
print("Test data columns:", test_data.columns)

print(train_data.head())
print(test_data.head())



train_data = pd.read_csv("train_data.txt", delimiter=',')
test_data = pd.read_csv("test_data.txt", delimiter=',')



train_data['summary'] = train_data['summary'].apply(clean_text)
test_data['summary'] = test_data['summary'].apply(clean_text)





label_encoder = LabelEncoder()
train_data['genre'] = label_encoder.fit_transform(train_data['genre'])

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_data['plot'])
X_test = vectorizer.transform(test_data['plot'])
y_train = train_data['genre']

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC(kernel='linear')
}


for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)

    print(f"\n{model_name} Results on Training Data:")
    print("Accuracy:", accuracy_score(y_train, y_pred_train))
    print("Classification Report:\n", classification_report(y_train, y_pred_train, target_names=label_encoder.classes_))


best_model = models['Logistic Regression']
test_predictions = best_model.predict(X_test)


test_predictions_labels = label_encoder.inverse_transform(test_predictions)


output = pd.DataFrame({'plot': test_data['plot'], 'predicted_genre': test_predictions_labels})
output.to_csv("predicted_genres.txt", index=False)



test_solution = pd.read_csv("test_data_solution.txt", delimiter='\t')
test_solution['genre'] = label_encoder.transform(test_solution['genre'])


from sklearn.metrics import accuracy_score


accuracy = accuracy_score(test_solution['genre'], test_predictions)
print(f"Accuracy on Test Data: {accuracy}")


import pickle


with open('genre_classifier.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

with open('label_encoder.pkl', 'wb') as label_encoder_file:
    pickle.dump(label_encoder, label_encoder_file)


with open('genre_classifier.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

with open('label_encoder.pkl', 'rb') as label_encoder_file:
    loaded_label_encoder = pickle.load(label_encoder_file)


new_plot = ["A young wizard begins his journey in a magical world."]
new_plot_cleaned = clean_text(new_plot[0])
new_plot_vectorized = loaded_vectorizer.transform([new_plot_cleaned])
predicted_genre = loaded_model.predict(new_plot_vectorized)


predicted_genre_label = loaded_label_encoder.inverse_transform(predicted_genre)
print("Predicted Genre:", predicted_genre_label[0])











































































#dfghjkl