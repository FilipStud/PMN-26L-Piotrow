import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

#Żeby ograniczyć losowość
np.random.seed(30)

def analiza(X, y, feature_names, class_names):
    df = pd.DataFrame(X, columns=feature_names)
    df['Gatunek'] = [class_names[i] for i in y]

    print(f"\n{6*''} STATYSTYKI OPISOWE {6*''}")
    print(df.describe())
    
    print(f"\n{6*''} LICZEBNOŚĆ KLAS {6*''}")
    print(df['Gatunek'].value_counts())

    plt.figure(figsize=(8, 6))
    sns.heatmap(df.iloc[:, :4].corr(), annot=True, cmap='RdYlBu', fmt=".2f")
    plt.title("Korelacja cech (Macierz Pearsona)")
    plt.show()


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
def plot_tsne(X, y_pred,y_test, title):
    y_pred=np.array(y_pred)
    y_test=np.array(y_test)
    tsne = TSNE(n_components=2, random_state=30)
    X_2d = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 6))
    markers=['o', 's', '^']
    unique_classes = np.unique(y_test)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_classes)))
    for i, marker in enumerate(markers):
        if i not in y_test: continue
        mask=(y_test==i)
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[int(p)] for p in y_pred[mask]], marker=marker, s=100, edgecolors='k', label=(f"Prawdziwa klasa: {i}"), alpha=0.7)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'Model mówi: Klasa {i}', 
               markerfacecolor=colors[i], markersize=10) for i in range(len(unique_classes))
    ]
    first_legend = plt.legend(handles=legend_elements, loc="upper left", title="KOLORY (Predykcja)")
    plt.gca().add_artist(first_legend)
    plt.legend(loc="lower left", title="KSZTAŁTY (Prawda)")
    shape_elements = [
        Line2D([0], [0], marker=markers[0], color='w', label='Klasa 0', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker=markers[1], color='w', label='Klasa 1', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker=markers[2], color='w', label='Klasa 2', markerfacecolor='gray', markersize=10)
    ]

    plt.legend(handles=shape_elements, loc="lower left", title="Kształty (Prawda)")
    plt.title(title)
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.grid(True, linestyle='--', alpha=0.6)
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
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    analiza(X, y, feature_names, unique_labels)
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




wart_k = 9




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
plot_tsne(X_test, y_pred, y_test, f"Wizualizacja t-SNE klasyfikacji KNN (Zbiór Testowy) dla k={wart_k}")
