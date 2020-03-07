from mylib import *


def task2():
    x = float32(0.53125)
    expected = 5312500
    numbers = [x] * (10 ** 7)

    startTime = time.perf_counter()
    sum = float32(0.0)
    err = float32(0.0)
    for val in numbers:
        y = val - err           # mały błąd stosujemy do małej wartości, która chcemy dodać do naszej sumy, więc nie tracimy dokładności
        temp = sum + y          # sumujemy obecny element z suma, czyli dodajemu duza liczba do bardzo malej - tracimy dokladnosc
        err = (temp - sum) - y  # wyliczamy sobie błąd, ktory zostal popelniony, w idealnym swiecie powinno byc zawsze zerem
                                # wazna kolejnosc, odejmujemy duze liczby dostajemy mala liczbe i dopiero od niej odemujemy y otrzymujac wartosc bledu
        sum = temp              # aktualizuj sume

    printTime(startTime)
    printData(expected, sum)



if __name__ == "__main__":
    task2()
