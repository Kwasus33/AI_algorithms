# Ćwiczenie nr 7

## Treść polecenia

1. Zapoznać się z jedną z bibliotek do tworzenia sieci Bayesowskich [pgmpy](https://github.com/pgmpy/pgmpy/tree/dev), 
[pomegranate](https://github.com/jmschrei/pomegranate), [bnlearn](https://github.com/erdogant/bnlearn)
2. Dla zbioru danych o zabójstwach w USA z lat 1980-2014 [https://www.kaggle.com/datasets/mrayushagrawal/us-crime-dataset](https://www.kaggle.com/datasets/mrayushagrawal/us-crime-dataset)
znaleźć strukturę sieci Bayesowskiej (structure learning), oraz estymować prawdopodobieństwa warunkowe
pomiędzy zmiennymi losowymi (parameter learning).
3. Zwizualizować i przeanalizować nauczoną sieć - jakie są rozkłady prawdopodobieństw pojedynczych zmiennych takich jak
 wiek, płeć, rasa i rodzaj znajomości ofiar i sprawców, jakie zależności pomiędzy wartościami zmiennych można zauważyć?
4. Przeprowadzić analogiczną analizę dla danych ograniczonych do jednej wybranej lokalizacji. 

## Uwagi 

- Proszę spróbować stworzyć jedną sieć dla danych globalnych tj. ze wszystkich lat, we wszystkich dostępnych miastach. 
Gdyby występowały problemy wydajnościowe (sieć się uczy za długo, brakuje pamięci), proszę ograniczyć się do jednej/kilku lokalizacji,
ewentualnie także zmniejszyć przedział czasowy.
- Z góry należy odrzucić cechy nieistotne/nieciekawe takie jak Record ID, Agency Code, Year itd... Usprawni to proces
tworzenia sieci i uprości jej analizę.
