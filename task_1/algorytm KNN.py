import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


#Żeby ograniczyć losowość
np.random.seed(30)

def knn(X_train, y_train, X_test, k, dist):
    def classify_single(x):
        dists = [dist(x, i) for i in X_train]
        indices = np.argpartition(dists, k)[:k]
        return np.argmax(np.bincount(y_train[indices]))

    return [classify_single(x) for x in X_test]

def precision_recall(y_pred, y_test):
    class_precision_recall = []
    for c in np.unique(y_test):
        tp = len([i for i in range(len(y_pred)) if y_pred[i] == c and y_test[i] == c])
        fp = len([i for i in range(len(y_pred)) if y_pred[i] == c and y_test[i] != c])
        fn = len([i for i in range(len(y_test)) if y_pred[i] != c and y_test[i] == c])
        precision = tp / (tp + fp) if tp + fp > 0 else 0.
        recall = tp / (tp + fn) if tp + fn > 0 else 0.
        class_precision_recall.append((c, precision, recall))
    return class_precision_recall


def print_precision_recall(result):
    for c, precision, recall in result:
        print("class:", c, "\nprecision:", precision, "\nrecall:", recall, "\n")


def train_test_split(X, y, ratio=0.8):
    indices = np.random.permutation(X.shape[0])
    train_len = int(X.shape[0] * ratio)
    
    #Przypisywanie pociętych danych do zmiennych używając indeksów
    X_train = X[indices[:train_len]]
    X_test = X[indices[train_len:]]
    y_train = y[indices[:train_len]]
    y_test = y[indices[train_len:]]
    
    return X_train, y_train, X_test, y_test

def f1_score(precision, recall):
    if precision + recall == 0: return 0
    return 2 * (precision * recall) / (precision + recall)

def get_accuracy(y_pred, y_test):
    return np.mean(y_pred == y_test)

#Funkcja do wizualizacji
def plot_tsne(X, y_pred, title):
    tsne = TSNE(n_components=2, random_state=30)
    X_2d = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred, cmap='viridis', edgecolors='k')
    plt.colorbar(scatter, ticks=[0, 1, 2], label='Klasy')
    plt.title(title)
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.show()




if __name__ == '__main__':
    raw_data = np.genfromtxt('iris.data', delimiter=',', dtype=str)
    
    #Cechy to pierwsze 4 kolumny (zamieniamy na float)
    X = raw_data[:, :4].astype(float)
    
    #Etykiety to ostatnia kolumna (zamieniamy nazwy na numery 0, 1, 2)
    labels = raw_data[:, 4]
    unique_labels = np.unique(labels)
    label_to_int = {name: i for i, name in enumerate(unique_labels)}
    y = np.array([label_to_int[name] for name in labels])

    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.6)
    
    print("X_train: ", "\n")
    print(X_train)
    print("y_train: ")
    print(y_train)
    print("X_test", "\n")
    print(X_test)
    print("y_test", "\n")
    print(y_test, "\n")


#Definicja brakującej odległości
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))


#Przykładowa wartość dla k

wart_k = 4




#Wywołanie KNN
y_pred = knn(X_train, y_train, X_test, wart_k, euclidean_distance)

#Obliczenie i wyświetlenie miar
results = precision_recall(y_pred, y_test)
print_precision_recall(results)

acc = get_accuracy(y_pred, y_test)
results = precision_recall(y_pred, y_test)

print(f"Ogólne Accuracy: {acc:.2%}\n")
for c, p, r in results:
    f1 = f1_score(p, r)
    print(f"Klasa {c+1} -> Precision: {p:.2f}, Recall: {r:.2f}, F1: {f1:.2f}")

#Wykres
plot_tsne(X_test, y_pred, f"Wizualizacja t-SNE klasyfikacji KNN (Zbiór Testowy) dla k={wart_k}")