import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# incarcare date din data.csv in old_table
df = pd.read_csv('data.csv', sep='\t', header=None)

# inlcouim cuvintele cu numere da=> 1, nu=>0
df.replace({'Da': 1, 'Nu': 0}, inplace=True)

y = df.iloc[:, 0]    # clasa va fi prima coloana din tabel
X = df.iloc[:, 1:11]  # trasaturile vor fi urmatoarele 10 (1-11 exclusiv)

# impartim datele in set de antrenare si testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# antrenam modelul
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

# predictii
y_pred = naive_bayes.predict(X_test)

# acuratete
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# QUIZ
user_response = input("Doriti sa faceti quiz-ul? (Da/Nu): ").strip().lower()

if user_response == 'da':

    answers = []
    questions = [
        "Tu esti persoana care a ales facultatea? (Da/Nu): ",
        "Inveti des ceva nou in legatura cu domeniul ales? (Da/Nu): ",
        "Sunt mai multe zilele in care te simti demotivat decat motivat sa inveti pentru admitere/ pentru temele de la facultate? (Da/Nu): ",
        "Vezi un viitor concret pentru tine dupa terminarea facultatii? (Da/Nu): ",
        "Programa ti se pare utila pentru planurile tale de viitor? (Da/Nu): ",
        "In domeniul in care vrei sa activezi este important sa ai o diploma? (Da/Nu): ",
        "Consideri mediul de la facultate unul propice dezvoltarii tale? (Da/Nu): ",
        "Consideri ca vreo materie pe care o faci/ o vei face la facultate este preferata ta? (Da/Nu): ",
        "Iti petreci timpul liber facand ceva ce are legatura cu domeniul studiat? (Da/Nu): ",
        "Consideri ca majoritatea timpului tau merita alocat domeniului ales? (Da/Nu): ",
    ]

    for question in questions:
        answer = input(question).strip().lower()
        answers.append(1 if answer == 'da' else 0)

    user_data = pd.DataFrame([answers])
    prediction = naive_bayes.predict(user_data)

    result = "DA" if prediction[0] == 1 else "NU"
    print(f"\nNu uita, testul nu ia in calcul decat anumiti factori. Bazat pe raspunsurile tale, raspunsul la intrebarea: Este facultatea aleasa potrivita pentru tine? este: {result}")
else:
    print("OK.")
