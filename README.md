
# Projekt Strojenia Modeli Uczenia Maszynowego

## Opis projektu

Repozytorium zawiera kod do strojenia i oceny różnych modeli uczenia maszynowego. Projekt bada kilka algorytmów regresji, metod strojenia hiperparametrów oraz technik walidacji, aby zoptymalizować wydajność modeli. Używane zbiory danych pochodzą z dziedziny szacowania nakładów w projektach programistycznych.

### Zbiory danych

W projekcie wykorzystano różne zbiory danych związane z szacowaniem wysiłku w projektach programistycznych, w tym:
- Maxwell
- Desharnais
- Kitchenham
- Miyazaki94
- NASA93

Zbiory danych znajdują się w folderze `datasets`.

### Główne komponenty

1. **mlAlgorithms.py**: Zawiera funkcje do wybierania algorytmów uczenia maszynowego, technik strojenia hiperparametrów i słowników z nimi.
2. **BootstrapCV.py**: Implementuje niestandardową walidację krzyżową przy użyciu próbkowania Bootstrap.
3. **loadData.py**: Odpowiada za ładowanie zbiorów danych do ramek.
4. **SimulatedAnnealingAlgorithm.py**: Implementacja algorytmu symulowanego wyżarzania do optymalizacji hiperparametrów.
5. **TPEAlgorithm.py**: Implementacja optymalizacji TPE (Tree-structured Parzen Estimator).
6. **main.py**: Główny skrypt uruchamiający eksperymenty ze strojeniem i zapisujący wyniki.
7. **Pliki ml-tuning-*data*-splits-*n***: Wyniki eksperymentów

### Jak działa program

- **Przetwarzanie danych**: Dane wejściowe są ładowane i dzielone na zbiory treningowe i testowe za pomocą `train_test_split`.
- **Konfiguracja pipeline’u**: Pipeline skaluje dane przy użyciu `MinMaxScaler` i stosuje różne algorytmy regresji.
- **Trenowanie i ocena bazowego modelu**: Kilka algorytmów, takich jak CART, GBM, SVM, KNN, RF i ElasticNet (EN), jest trenowanych przy użyciu różnych technik walidacji:
- **Trenowanie i ocena strojonego modelu**: Wykorzystywane są różne metody strojenia hiperparametrów, takie jak grid search, random search, optymalizacja bayesowska, TPE i symulowane wyżarzanie (SA) oraz metod walidacji takie jak k-fold, repeated k-fold, hold0out, Monte Carlo i bootstrap.
- **Zapis wyników**: Wyniki są zapisywane w plikach CSV i obejmują metryki wydajności, takie jak MAE, MSE, MedAE, RMSE i R2.

### Uruchamianie projektu

Po sklonowaniu projektu można go uruchomić za pomocą jednej komendy:

`python main.py`

Przed uruchomieniem można zmienić usawienia ilości foldów, które będą zastosowane dla metod walidacji zmieniając wartość zmiennej `n_splits` w 62 linijce `main.py`.  Wyniki zostaną zapisane w odpowiednich folderach eksperymentów (np. ml-tuning-08-07-2024-splits-3).

### Wyniki
1. Metryki wydajności (MAE, MSE, RMSE, R2) są zapisywane dla modeli strojonych i niestrojonych.
2. Wyniki zawierają najlepsze znalezione hiperparametry, wyniki treningowe i testowe oraz czas strojenia.
3. Wyniki są zapisywane w formacie CSV dla każdego eksperymentu.
