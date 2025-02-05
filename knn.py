import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier

def Knn(features):
    # Baca dataset glcm_features.csv menggunakan pandas
    data = pd.read_csv('glcm_rempah_512px+he+HSV_v2.csv')

    # Pisahkan fitur (X) dan label (y)
    X = data.drop('label', axis=1)
    y = data['label']

    # Bagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=150)

    # Buat model KNN
    model = KNeighborsClassifier()

    # Definisikan grid hyperparameter yang akan ditelusuri
    param_grid = {
        'n_neighbors': [5 , 7, 9, 11, 13],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    # Buat objek GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5)

    # Latih model dengan data latih
    grid_search.fit(X_train, y_train)

    # Dapatkan model terbaik setelah hyperparameter tuning
    best_model = grid_search.best_estimator_

    # Set feature names for the input data
    feature_names = X.columns
    features_df = pd.DataFrame(features, columns=feature_names)

    # Lakukan prediksi menggunakan model KNN terbaik
    hasil = best_model.predict(features_df)

    return hasil[0]