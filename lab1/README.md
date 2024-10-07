# Ćwiczenie nr 1

## Treść polecenia

Znana jest funkcja celu

 1. $f(x) = Ax + Bsin(x)$ 
    
    $x \in{(-4\pi, 4\pi)}$
 2. $g(x, y) = \frac{Cxy}{e^{x^2 + y^2}}$
    
    $x \in{(-2, 2)}$
    
    $y \in{(-2, 2)}$

Zaimplementować algorytm spadku wzdłuż gradientu opisany na wykładzie.

Użyć zaimplementowanego algorytmu do wyznaczenia ekstremów funkcji.

Zbadać wpływ następujących parametrów na proces optymalizacji:
 - długość kroku uczącego
 - limit maksymalnej liczby kroków algorytmu
 - rozmieszczenie punktu startowego

Zinterpretować wyniki w kontekście kształtu badanej funkcji. 

## Uwagi

 - Punkty startowe algorytmu należy losować z zadanego przedziału.
 - Implementacja algorytmu powinna być jedna dla dowolnej zadanej funkcji i nazywać się `grad_descent`.
 - Do wizualizacji funkcji można użyć dowolnego narzędzia. Pisanie wizualizatora funkcji nie jest celem tego ćwiczenia.
