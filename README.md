# Kontakt

MS Teams

# Harmonogram zajęć grupy nr 104.BO 

| Id | termin     | 	temat                                                     | uwagi                                            |
|----|------------|------------------------------------------------------------|--------------------------------------------------|
| 1  | 11.10.2024 | Zagadnienie przeszukiwania i podstawowe podejścia do niego | Wprowadzenie                                     |
| 2  | 18.10.2024 | Zagadnienie przeszukiwania i podstawowe podejścia do niego | Konsultacje                                      |
| 3  | 25.10.2024 | Algorytmy ewolucyjne                                       | Ocena poprzedniego ćwiczenia                     |
| 4  | 8.11.2024  | Algorytmy ewolucyjne                                       | Konsultacje                                      |
| 5  | 15.11.2024 | Dwuosobowe gry deterministyczne                            | Ocena poprzedniego ćwiczenia                     |
| 6  | 22.11.2024 | Dwuosobowe gry deterministyczne                            | Konsultacje                                      |
| 7  | 29.11.2024 | Regresja i klasyfikacja                                    | Ocena poprzedniego ćwiczenia                     |
| 8  | 6.12.2024  | Regresja i klasyfikacja                                    | Konsultacje                                      |
| 9  | 13.12.2024 | Sztuczne sieci neuronowe                                   | Ocena poprzedniego ćwiczenia; Ćwiczenie w parach |
| 10 | 20.12.2024 | Sztuczne sieci neuronowe                                   | Brak spotkania stacjonarnego                     |
| 11 | 3.01.2025  | Uczenie się ze wzmocnieniem                                | Ocena poprzedniego ćwiczenia                     |
| 12 | 10.01.2025 | Uczenie się ze wzmocnieniem                                | Brak spotkania stacjonarnego                     |
| 13 | 17.01.2025 | Modele bayesowskie                                         | Ocena poprzedniego ćwiczenia                     |
| 14 | 24.01.2025 | Modele bayesowskie                                         | Ocena poprzedniego ćwiczenia                     |


Zamiast spotkań 10 i 12 chętnych na konsultacje zapraszam do umawiania się na spotkania indywidualne.

# Zasady ćwiczeń

[https://staff.elka.pw.edu.pl/~rbiedrzy/WSI/index.html](https://staff.elka.pw.edu.pl/~rbiedrzy/WSI/index.html)

## Doprecyzowanie dla grupy 104.BO

 - Głównym narzędziem organizacji ćwiczeń jest MS Teams. Na nim będą pojawiać się materiały z ćwiczeń oraz informacje organizacyjne.
 - Ćwiczenia oddajemy na zajęciach stacjonarnych. Prowadzący ocenia rozwiązanie (kod+sprawozdanie) w ciągu tygodnia od wysłania. 
Ewentualne pytania o pojawienie się oceny w USOSie można zadawać dopiero po tym czasie.
 - Na oddanie ćwiczenia przewidziane jest do 6 minut. W celu uniknięcia kolejek i kolizji termin oddania należy rezerwować w arkuszu https://wutwaw-my.sharepoint.com/:x:/g/personal/01143578_pw_edu_pl/EdH9VQOQPKdGpm2WCMrs7UUB93to8xV573ylCFiR5vLQPQ?e=baRrLZ. 
 - Obecność na zajęciach poza oddaniem nie jest obowiązkowa.
 - Ćwiczenia można oddawać tydzień wcześniej - w terminach konsultacji. W tym przypadku nie trzeba rezerwować terminu oddania.
 - Student powinien przygotować jedno repozytorium na wszystkie ćwiczenia z WSI.
Struktura przykładowego repozytorium znajduje się na [https://gitlab-stud.elka.pw.edu.pl/bolber/wsi-template.git](https://gitlab-stud.elka.pw.edu.pl/bolber/wsi-template.git). Można zrobić fork, lub kopię tego repozytorium.
Przykładowe repozytorium może ulegać modyfikacjom w trakcie semestru, w szczególności przed rozpoczęciem kolejnego tematu ćwiczeń.
 - Prowadzący powinien być dodany w repozytorium studenta z uprawnieniem do
komentowania merge requestów. Na każde zadanie należy utworzyć brancha.
W ramach oddawania zadania należy utworzyć merge request brancha zadania do
głównego brancha.
 - Branche i merge requesty należy nazywać krótkimi nazwami umożliwiającymi łatwą identyfikację zadania np. lab1, lab2 etc. 


# Uwagi ogólne dot. kodu

 - Językiem implementacji jest Python. Można korzystać z notatników Jupyter, ale należy zwrócić uwagę na reprodukowalność rozwiązania.
 - Zalecane jest (czasem konieczne) używanie pakietów spoza standardu języka
np. numpy, pandas, matplotlib. Użyte pakiety należy wypisać wraz z wersją w pliku `requirements.txt`
 - Przed wysłaniem rozwiązania należy sprawdzić, czy wykonuje się ono na czystym środowisku
    ```shell
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python nazwa_skryptu argumenty
    ```
 - Nie jest dozwolone wykorzystywać gotowych funkcji implementujących temat ćwiczenia
 - Czystość, czytelność kodu oraz raportu wpływa na ocenę ćwiczenia, 
wskazane jest korzystać z formatterów kodu np. https://github.com/psf/black
 - Parametry algorytmu podawać używając plików konfiguracyjnych (.json, .yaml...), lub, jeśli jest ich niewiele, jako argumenty wywołania skryptu (argparse).
Nie należy w tym celu używać magicznych stałych, ani zmiennych globalnych.
 - Im kod jest krótszy, tym, zazwyczaj, lepiej
 - Kod (nazwy zmiennych, plików, komentarze) piszemy po angielsku 

## Sugerowana struktura rozwiązania

1. Implementacja algorytmu w wydzielonej funkcji/klasie np.
    ```python
    class LogisticRegression:
        ...
        def fit(self, ...):
            ...
        
        def predict(self, ...):
            ...
    ```
2. Wydzielone funkcje czytające dane wejściowe i parametry algorytmu.
3. Funkcja main wczytuje dane, parametry, puszcza algorytm, zapisuje wyniki
4. Oddzielny moduł wizualizujący wyniki

# Uwagi ogólne dot. sprawozdania

 - Dokumentacja powinna być w pliku .pdf, .html lub .ipynb.
 - Proszę przeczytać sekcję *Uwagi* ze strony wykładowcy [https://staff.elka.pw.edu.pl/~rbiedrzy/WSI_CW/index.html](https://staff.elka.pw.edu.pl/~rbiedrzy/WSI_CW/index.html)
 - Tabele i wykresy muszą być przedstawione w formie łatwo zrozumiałej (np. wszystkie osie na wykresach powinny być podpisane)
 - Nie trzeba dokumentować kodu
 - W opisach tabel/wykresów oprócz wizualizowanej wartości zmiennych badanych należy też umieścić informacje o wartościach parametrów,
które są ustalone dla danego eksperymentu 
 - Sprawozdanie można pisać w języku polskim lub angielskim

## Sugerowana zawartość sprawozdania

 - Treść ćwiczenia
 - Doprecyzowanie (jeśli było konieczne)
 - Cel i opis eksperymentów (co jest badane, na jakim zbiorze danych, jaką metryką mierzymy jakość ...)
 - Instrukcja potrzebna do odtworzenia wyników (wraz z przygotowaniem środowiska, danych)
 - Wyniki w formie tabel i wykresów
 - Wnioski (komentarz wyników, próba własnej interpretacji)
