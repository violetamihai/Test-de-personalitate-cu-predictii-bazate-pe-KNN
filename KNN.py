import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# incarcare date din data.csv in old_table
old_table = pd.read_csv('data.csv', sep='\t', header=None)

# am denumit old_table datele preprocesate astfel: in loc de da=> 1, nu=>0
old_table.replace({'Da': 1, 'Nu': 0}, inplace=True)

# in functie de intrebari, am schimbat scorul la 1 dintre ele
# (cea cu te simti demotivat... deoarece raspunsul potrivit ar fi fost pe dos fata de celelalte)

for i in range(len(old_table)):
    for j in range(len(old_table.columns)):

        if (j == 1 or j == 3) and old_table.iloc[i, j] == 0:
            old_table.iloc[i, j] = 1

# pentru KNN am ales 2 trasaturi importante: factorii externi/ interni de decizie
# noua matrice new_table va avea 3 coloane (clasa + cele 2 trasaturi)
new_table = pd.DataFrame()

# clasa ramane la fel
new_table['Class'] = old_table.iloc[:, 0]

# a doua coloana reprezinta suma intrebarilor 1,4,5,6,10 (pe care le-am considerat ca tin de lumea exterioara)
#in chestionar, corespund intrebarilor 2,5,6,7,11
new_table['Factori Externi'] = old_table.iloc[:, [1,4,5,6,10]].sum(axis=1)

# a 3 a coloana reprezinta suma intrebarilor care au ramas
new_table['Factori Interni'] = old_table.iloc[:, [2,3,7,8,9]].sum(axis=1)

print("Old Table:")
print(old_table)

X = new_table.iloc[:, -2:]  # ultimele 2 coloane coloane din new_table vor fi trasaturile
y = new_table.iloc[:, 0]    # prima este clasa

print("\nNew Table:")
print(new_table)

# impartim datele in set de antrenare si testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# antrenam modelul
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# predictii
y_pred = knn.predict(X_test)

# acuratete
accuracy = accuracy_score(y_test, y_pred)
print(f'\nAccuracy: {accuracy * 100:.2f}%')



