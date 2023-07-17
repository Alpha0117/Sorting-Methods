import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import re
import random
import webbrowser
from sorting_algorithms import *


def mostrar_descripcion(metodo_seleccionado):
    descripcion = ""
    codigo = ""
    logica_matematica = ""

    if metodo_seleccionado == "Bubble Sort":
        descripcion = "Bubble Sort es un algoritmo de ordenamiento simple que repetidamente recorre la lista, compara elementos adyacentes y los intercambia si están en el orden incorrecto. El proceso se repite hasta que la lista esté completamente ordenada."
        codigo = "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]"
        logica_matematica = "La lógica matemática detrás de Bubble Sort se basa en la comparación de elementos adyacentes. En cada pasada, el algoritmo compara pares de elementos y los intercambia si están en el orden incorrecto. Este proceso se repite hasta que la lista esté completamente ordenada."

    elif metodo_seleccionado == "Insertion Sort":
        descripcion = "Insertion Sort es un algoritmo de ordenamiento que construye una lista ordenada un elemento a la vez. En cada iteración, el algoritmo toma un elemento de la lista de entrada y lo inserta en la posición correcta de la lista ordenada."
        codigo = "def insertion_sort(arr):\n    for i in range(1, len(arr)):\n        key = arr[i]\n        j = i - 1\n        while j >= 0 and arr[j] > key:\n            arr[j+1] = arr[j]\n            j -= 1\n        arr[j+1] = key"
        logica_matematica = "La lógica matemática detrás de Insertion Sort se basa en la inserción de elementos en una lista ordenada. En cada iteración, el algoritmo toma un elemento de la lista de entrada y lo inserta en la posición correcta de la lista ordenada mediante comparaciones e intercambios."
            
    elif metodo_seleccionado == "Selection Sort":
        descripcion = "Selection Sort es un algoritmo de ordenamiento que divide la lista en dos partes: una parte ordenada y otra parte sin ordenar. En cada iteración, encuentra el elemento mínimo de la parte sin ordenar y lo coloca al final de la parte ordenada."
        codigo = "def selection_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        min_idx = i\n        for j in range(i+1, n):\n            if arr[j] < arr[min_idx]:\n                min_idx = j\n        arr[i], arr[min_idx] = arr[min_idx], arr[i]"
        logica_matematica = "La lógica matemática detrás de Selection Sort se basa en la búsqueda del elemento mínimo en cada iteración. En cada paso, el algoritmo compara el elemento actual con el mínimo encontrado hasta el momento y realiza intercambios si es necesario. Esto asegura que el elemento mínimo se coloque en la posición correcta de la parte ordenada."
        
    elif metodo_seleccionado == "Merge Sort":
        descripcion = "Merge Sort es un algoritmo de ordenamiento basado en la técnica de dividir y conquistar. Divide la lista en dos mitades, ordena cada mitad por separado y luego combina las dos mitades ordenadas en una lista ordenada final."
        codigo = "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left_half = arr[:mid]\n    right_half = arr[mid:]\n    left_half = merge_sort(left_half)\n    right_half = merge_sort(right_half)\n    return merge(left_half, right_half)\n\ndef merge(left_half, right_half):\n    merged = []\n    while left_half and right_half:\n        if left_half[0] <= right_half[0]:\n            merged.append(left_half[0])\n            left_half = left_half[1:]\n        else:\n            merged.append(right_half[0])\n            right_half = right_half[1:]\n    merged += left_half\n    merged += right_half\n    return merged"
        logica_matematica = "La lógica matemática detrás de Merge Sort se basa en la combinación de listas ordenadas. El algoritmo divide recursivamente la lista en mitades hasta que cada mitad contenga un solo elemento. Luego, combina las mitades ordenadas de manera que se obtenga una lista ordenada completa."

    elif metodo_seleccionado == "Quick Sort":
        descripcion = "Quick Sort es un algoritmo de ordenamiento basado en la técnica de dividir y conquistar. Selecciona un elemento pivote de la lista y coloca todos los elementos más pequeños que el pivote a su izquierda y todos los elementos más grandes a su derecha. Luego, aplica el mismo proceso de manera recursiva en las sublistas resultantes."
        codigo = "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)"
        logica_matematica = "La lógica matemática detrás de Quick Sort se basa en la partición de la lista alrededor de un pivote. El algoritmo selecciona un elemento pivote y reorganiza los elementos de manera que los elementos más pequeños que el pivote estén a su izquierda y los elementos más grandes estén a su derecha. Luego, aplica el mismo proceso en las sublistas resultantes hasta obtener una lista ordenada."
        
    elif metodo_seleccionado == "Heap Sort":
        descripcion = "Heap Sort es un algoritmo de ordenamiento basado en la estructura de datos conocida como heap (montículo). Construye un montículo máximo a partir de la lista y luego extrae el elemento máximo repetidamente para obtener una lista ordenada."
        codigo = "def heap_sort(arr):\n    n = len(arr)\n    for i in range(n//2 - 1, -1, -1):\n        heapify(arr, n, i)\n    for i in range(n-1, 0, -1):\n        arr[i], arr[0] = arr[0], arr[i]\n        heapify(arr, i, 0)\n\ndef heapify(arr, n, i):\n    largest = i\n    left = 2 * i + 1\n    right = 2 * i + 2\n    if left < n and arr[left] > arr[largest]:\n        largest = left\n    if right < n and arr[right] > arr[largest]:\n        largest = right\n    if largest != i:\n        arr[i], arr[largest] = arr[largest], arr[i]\n        heapify(arr, n, largest)"
        logica_matematica = "La lógica matemática detrás de Heap Sort se basa en la propiedad de un montículo máximo. El algoritmo construye un montículo máximo a partir de la lista, donde el elemento en la posición `i` es mayor que los elementos en las posiciones `2i+1` y `2i+2`. Luego, extrae repetidamente el elemento máximo del montículo para obtener una lista ordenada."

    elif metodo_seleccionado == "Counting Sort":
        descripcion = "Counting Sort es un algoritmo de ordenamiento que utiliza la cuenta de ocurrencias de cada elemento para determinar su posición en la lista ordenada. Es eficiente cuando el rango de valores de entrada es pequeño en comparación con el tamaño de la lista."
        codigo = "def counting_sort(arr):\n    n = len(arr)\n    max_value = max(arr)\n    count = [0] * (max_value + 1)\n    sorted_arr = [0] * n\n\n    for num in arr:\n        count[num] += 1\n\n    for i in range(1, max_value + 1):\n        count[i] += count[i - 1]\n\n    for i in range(n - 1, -1, -1):\n        num = arr[i]\n        sorted_arr[count[num] - 1] = num\n        count[num] -= 1\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de Counting Sort se basa en el conteo de ocurrencias de cada elemento en la lista de entrada. El algoritmo crea un arreglo de conteo donde el índice `i` representa un valor posible en la lista y el valor en el índice `i` indica cuántas veces aparece ese valor. Luego, utiliza este arreglo de conteo para determinar la posición de cada elemento en la lista ordenada."

    elif metodo_seleccionado == "Bucket Sort":
        descripcion = "Bucket Sort es un algoritmo de ordenamiento que distribuye los elementos en diferentes 'cubetas' o baldes basados en sus valores. Luego, aplica otro algoritmo de ordenamiento (generalmente Insertion Sort) en cada cubeta para ordenar los elementos internamente."
        codigo = "def bucket_sort(arr):\n    n = len(arr)\n    buckets = [[] for _ in range(n)]\n    sorted_arr = []\n\n    for num in arr:\n        index = int(n * num)\n        buckets[index].append(num)\n\n    for bucket in buckets:\n        insertion_sort(bucket)\n        sorted_arr.extend(bucket)\n\n    return sorted_arr\n\n\ndef insertion_sort(arr):\n    for i in range(1, len(arr)):\n        key = arr[i]\n        j = i - 1\n        while j >= 0 and arr[j] > key:\n            arr[j + 1] = arr[j]\n            j -= 1\n        arr[j + 1] = key"
        logica_matematica = "La lógica matemática detrás de Bucket Sort se basa en la distribución de elementos en diferentes cubetas o baldes según sus valores. Cada cubeta se ordena internamente utilizando otro algoritmo de ordenamiento, como Insertion Sort. Luego, se concatenan las cubetas ordenadas para obtener la lista ordenada final."

    elif metodo_seleccionado == "Radix Sort":
        descripcion = "Radix Sort es un algoritmo de ordenamiento que utiliza la posición de los dígitos para ordenar los elementos. Comienza por los dígitos menos significativos y avanza hacia los más significativos. Se aplica una pasada de ordenamiento para cada dígito, utilizando un algoritmo de ordenamiento estable como Counting Sort."
        codigo = "def radix_sort(arr):\n    max_value = max(arr)\n    exp = 1\n\n    while max_value // exp > 0:\n        counting_sort_by_digit(arr, exp)\n        exp *= 10\n\n    return arr\n\n\ndef counting_sort_by_digit(arr, exp):\n    n = len(arr)\n    count = [0] * 10\n    sorted_arr = [0] * n\n\n    for num in arr:\n        digit = (num // exp) % 10\n        count[digit] += 1\n\n    for i in range(1, 10):\n        count[i] += count[i - 1]\n\n    for i in range(n - 1, -1, -1):\n        num = arr[i]\n        digit = (num // exp) % 10\n        sorted_arr[count[digit] - 1] = num\n        count[digit] -= 1\n\n    for i in range(n):\n        arr[i] = sorted_arr[i]"
        logica_matematica = "La lógica matemática detrás de Radix Sort se basa en la comparación y ordenamiento de los elementos según los dígitos en diferentes posiciones. Se aplica una pasada de ordenamiento para cada dígito, utilizando Counting Sort para ordenar los elementos en cada posición específica de los dígitos."

    elif metodo_seleccionado == "Shell Sort":
        descripcion = "Shell Sort es un algoritmo de ordenamiento que mejora el rendimiento del algoritmo de inserción (Insertion Sort) al reducir el número de comparaciones y movimientos de elementos. Divide la lista en subgrupos más pequeños y los ordena utilizando Insertion Sort. Luego, combina los subgrupos y realiza otro paso de ordenamiento. Este proceso se repite hasta que la lista completa esté ordenada."
        codigo = "def shell_sort(arr):\n    n = len(arr)\n    gap = n // 2\n\n    while gap > 0:\n        for i in range(gap, n):\n            temp = arr[i]\n            j = i\n\n            while j >= gap and arr[j - gap] > temp:\n                arr[j] = arr[j - gap]\n                j -= gap\n\n            arr[j] = temp\n\n        gap //= 2\n\n    return arr"
        logica_matematica = "La lógica matemática detrás de Shell Sort se basa en la división de la lista en subgrupos más pequeños y el uso del algoritmo de inserción (Insertion Sort) para ordenar los elementos en cada subgrupo. Luego, se realiza un paso de ordenamiento en el cual se combina y ordena los subgrupos en la lista completa. Este proceso se repite con un tamaño de subgrupo más pequeño hasta que la lista completa esté ordenada."

    elif metodo_seleccionado == "Binary Tree Sort":
        descripcion = "Binary Tree Sort es un algoritmo de ordenamiento que utiliza un árbol binario de búsqueda para ordenar los elementos. Cada elemento se inserta en el árbol y luego se realiza un recorrido inorden para obtener los elementos ordenados."
        codigo = "class Node:\n    def __init__(self, key):\n        self.key = key\n        self.left = None\n        self.right = None\n\n\ndef binary_tree_sort(arr):\n    root = None\n\n    for key in arr:\n        root = insert(root, key)\n\n    sorted_arr = []\n    in_order(root, sorted_arr)\n    return sorted_arr\n\n\ndef insert(root, key):\n    if root is None:\n        return Node(key)\n\n    if key < root.key:\n        root.left = insert(root.left, key)\n    else:\n        root.right = insert(root.right, key)\n\n    return root\n\n\ndef in_order(root, sorted_arr):\n    if root:\n        in_order(root.left, sorted_arr)\n        sorted_arr.append(root.key)\n        in_order(root.right, sorted_arr)"
        logica_matematica = "La lógica matemática detrás de Binary Tree Sort se basa en la estructura de un árbol binario de búsqueda. Cada elemento se inserta en el árbol y luego se realiza un recorrido inorden para obtener los elementos ordenados."

    elif metodo_seleccionado == "Optimal Merge Sort":
        descripcion = "Optimal Merge Sort es un algoritmo de ordenamiento basado en el Merge Sort que optimiza la utilización de la memoria y el número de comparaciones. Divide la lista en bloques y realiza una mezcla óptima de los bloques utilizando el concepto de árbol de fusiones."
        codigo = "def optimal_merge_sort(arr):\n    n = len(arr)\n    block_size = 1\n\n    while block_size < n:\n        for start in range(0, n, 2 * block_size):\n            mid = min(start + block_size - 1, n - 1)\n            end = min(start + 2 * block_size - 1, n - 1)\n            merge(arr, start, mid, end)\n\n        block_size *= 2\n\n    return arr\n\n\ndef merge(arr, start, mid, end):\n    merged = []\n    i = start\n    j = mid + 1\n\n    while i <= mid and j <= end:\n        if arr[i] <= arr[j]:\n            merged.append(arr[i])\n            i += 1\n        else:\n            merged.append(arr[j])\n            j += 1\n\n    while i <= mid:\n        merged.append(arr[i])\n        i += 1\n\n    while j <= end:\n        merged.append(arr[j])\n        j += 1\n\n    for k in range(len(merged)):\n        arr[start + k] = merged[k]"
        logica_matematica = "La lógica matemática detrás de Optimal Merge Sort se basa en dividir la lista en bloques y realizar una mezcla óptima de los bloques utilizando el concepto de árbol de fusiones. Se combinan los bloques de manera eficiente, minimizando el número de comparaciones y optimizando el uso de la memoria."

    elif metodo_seleccionado == "Natural Merge Sort":
        descripcion = "Natural Merge Sort es un algoritmo de ordenamiento que utiliza el concepto de mezcla natural para ordenar los elementos. Se basa en identificar las secuencias ordenadas en la lista y fusionarlas de manera eficiente."
        codigo = "def natural_merge_sort(arr):\n    n = len(arr)\n    left = 0\n    right = n - 1\n    sorted_arr = arr.copy()\n\n    while left < right:\n        mid = find_mid(arr, left, right)\n        merge(arr, left, mid, right, sorted_arr)\n        left = right + 1\n        right = find_right(arr, left, n - 1)\n        merge(sorted_arr, left, right, n - 1, arr)\n        left = right + 1\n        right = find_right(sorted_arr, left, n - 1)\n\n    return sorted_arr\n\n\ndef find_mid(arr, left, right):\n    mid = left\n\n    while mid < right and arr[mid] <= arr[mid + 1]:\n        mid += 1\n\n    return mid\n\n\ndef find_right(arr, left, right):\n    if left == right:\n        return right\n\n    while right < len(arr) - 1 and arr[right] <= arr[right + 1]:\n        right += 1\n\n    return right\n\n\ndef merge(arr, left, mid, right, sorted_arr):\n    i = left\n    j = mid + 1\n    k = left\n\n    while i <= mid and j <= right:\n        if arr[i] <= arr[j]:\n            sorted_arr[k] = arr[i]\n            i += 1\n        else:\n            sorted_arr[k] = arr[j]\n            j += 1\n        k += 1\n\n    while i <= mid:\n        sorted_arr[k] = arr[i]\n        i += 1\n        k += 1\n\n    while j <= right:\n        sorted_arr[k] = arr[j]\n        j += 1\n        k += 1"
        logica_matematica = "La lógica matemática detrás de Natural Merge Sort se basa en identificar las secuencias ordenadas en la lista y fusionarlas de manera eficiente. Se utiliza la búsqueda de puntos de división para identificar las secuencias y luego se realiza una mezcla de las secuencias para obtener la lista ordenada."

    elif metodo_seleccionado == "Pigeonhole Sort":
        descripcion = "Pigeonhole Sort es un algoritmo de ordenamiento que utiliza la técnica de clasificar los elementos en un conjunto de 'agujeros' o 'palomas'. Cada elemento se asigna a un agujero y luego se extraen en orden para obtener la lista ordenada."
        codigo = "def pigeonhole_sort(arr):\n    min_val = min(arr)\n    max_val = max(arr)\n    range_size = max_val - min_val + 1\n    pigeonholes = [0] * range_size\n\n    for num in arr:\n        pigeonholes[num - min_val] += 1\n\n    sorted_arr = []\n    for i in range(range_size):\n        while pigeonholes[i] > 0:\n            sorted_arr.append(i + min_val)\n            pigeonholes[i] -= 1\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de Pigeonhole Sort se basa en clasificar los elementos en un conjunto de 'agujeros' o 'palomas'. Cada elemento se asigna a un agujero de acuerdo a su valor y luego se extraen en orden para obtener la lista ordenada."

    elif metodo_seleccionado == "Card Sort":
        descripcion = "Card Sort es un algoritmo de ordenamiento que se basa en distribuir los elementos en una estructura de tarjetas y luego recogerlas en orden. Es especialmente eficiente para ordenar listas pequeñas o listas con elementos repetidos."
        codigo = "def card_sort(arr):\n    sorted_arr = []\n\n    for num in arr:\n        sorted_arr.append(num)\n\n    sorted_arr.sort()\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de Card Sort es simple. Se distribuyen los elementos en una estructura de tarjetas y luego se recogen en orden. La función `sort` incorporada en Python se utiliza para ordenar las tarjetas y obtener la lista ordenada."

    elif metodo_seleccionado == "Shellsort":
        descripcion = "Shellsort, también conocido como ordenamiento por inserción de Shell, es una variante del ordenamiento por inserción que divide la lista en subgrupos y luego los ordena utilizando el ordenamiento por inserción. A medida que los subgrupos se vuelven más pequeños, el ordenamiento se vuelve más eficiente."
        codigo = "def shellsort(arr):\n    n = len(arr)\n    gap = n // 2\n\n    while gap > 0:\n        for i in range(gap, n):\n            temp = arr[i]\n            j = i\n\n            while j >= gap and arr[j - gap] > temp:\n                arr[j] = arr[j - gap]\n                j -= gap\n\n            arr[j] = temp\n\n        gap //= 2\n\n    return arr"
        logica_matematica = "La lógica matemática detrás de Shellsort se basa en dividir la lista en subgrupos y aplicar el ordenamiento por inserción en cada subgrupo. A medida que los subgrupos se vuelven más pequeños, el ordenamiento se vuelve más eficiente debido a que los elementos están más cerca de su posición final."

    elif metodo_seleccionado == "Smoothsort":
        descripcion = "Smoothsort es un algoritmo de ordenamiento que combina las técnicas de ordenamiento por inserción y ordenamiento por árbol binario. Utiliza una estructura de datos llamada árbol de Dijkstra para realizar las comparaciones y las operaciones de intercambio de manera eficiente."
        codigo = "def smoothsort(arr):\n    def down_heap(arr, start, end, r):\n        # Implementación de down-heap\n        # ...\n\n    def sift(arr, start, size, r):\n        # Implementación de sift\n        # ...\n\n    def trinkle(arr, start, p, r):\n        # Implementación de trinkle\n        # ...\n\n    def semitrinkle(arr, start, p, r):\n        # Implementación de semitrinkle\n        # ...\n\n    def smooth_sort_helper(arr, start, size, r):\n        # Implementación de smooth_sort_helper\n        # ...\n\n    sorted_arr = arr.copy()\n    r = 0\n    smooth_sort_helper(sorted_arr, 0, len(sorted_arr), r)\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de Smoothsort se basa en combinar las técnicas de ordenamiento por inserción y ordenamiento por árbol binario. Utiliza una estructura de datos llamada árbol de Dijkstra para realizar las comparaciones y las operaciones de intercambio de manera eficiente."

    elif metodo_seleccionado == "Most Significant Digit Radix Sort":
        descripcion = "Most Significant Digit (MSD) Radix Sort es un algoritmo de ordenamiento basado en la clasificación de los elementos de una lista según los dígitos más significativos de sus representaciones en una base numérica. Se utiliza un enfoque de ordenamiento estable y se aplica recursivamente a cada sublista generada por los dígitos más significativos."
        codigo = "def msd_radix_sort(arr):\n    def counting_sort(arr, exp):\n        # Implementación de counting_sort\n        # ...\n\n    def msd_radix_sort_helper(arr, low, high, exp):\n        # Implementación de msd_radix_sort_helper\n        # ...\n\n    sorted_arr = arr.copy()\n    low = 0\n    high = len(sorted_arr) - 1\n    exp = 1\n    msd_radix_sort_helper(sorted_arr, low, high, exp)\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de Most Significant Digit Radix Sort se basa en clasificar los elementos de una lista según los dígitos más significativos de sus representaciones en una base numérica. Se utiliza un enfoque recursivo y estable para ordenar los elementos en cada sublista generada por los dígitos más significativos."

    elif metodo_seleccionado == "Least Significant Digit Radix Sort":
        descripcion = "Least Significant Digit (LSD) Radix Sort es un algoritmo de ordenamiento basado en la clasificación de los elementos de una lista según los dígitos menos significativos de sus representaciones en una base numérica. Se utiliza un enfoque de ordenamiento estable y se aplica recursivamente a cada sublista generada por los dígitos menos significativos."
        codigo = "def lsd_radix_sort(arr):\n    def counting_sort(arr, exp):\n        # Implementación de counting_sort\n        # ...\n\n    def lsd_radix_sort_helper(arr, low, high, exp):\n        # Implementación de lsd_radix_sort_helper\n        # ...\n\n    sorted_arr = arr.copy()\n    low = 0\n    high = len(sorted_arr) - 1\n    exp = 1\n    lsd_radix_sort_helper(sorted_arr, low, high, exp)\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de Least Significant Digit Radix Sort se basa en clasificar los elementos de una lista según los dígitos menos significativos de sus representaciones en una base numérica. Se utiliza un enfoque recursivo y estable para ordenar los elementos en cada sublista generada por los dígitos menos significativos."

    elif metodo_seleccionado == "Comparison Sort":
        descripcion = "Comparison Sort es una categoría de algoritmos de ordenamiento que se basa en comparar los elementos de una lista y realizar intercambios o movimientos según un criterio de comparación definido. Algunos ejemplos populares de Comparison Sort son Bubble Sort, Insertion Sort, Merge Sort y Quick Sort."
        codigo = "def comparison_sort(arr):\n    # Implementación del algoritmo de Comparison Sort\n    # ...\n\n    sorted_arr = arr.copy()\n    # Lógica de ordenamiento\n    # ...\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de los algoritmos de Comparison Sort se basa en comparar los elementos de una lista y realizar intercambios o movimientos según un criterio de comparación definido. Cada algoritmo de Comparison Sort tiene su propia estrategia y complejidad en términos de comparaciones y movimientos."

    elif metodo_seleccionado == "Bitonic Sort":
        descripcion = "Bitonic Sort es un algoritmo de ordenamiento paralelo que se basa en la construcción de secuencias bitónicas. Una secuencia bitónica es una secuencia que primero aumenta y luego disminuye o viceversa. El algoritmo utiliza una estrategia de división y fusión para ordenar los elementos en una secuencia bitónica."
        codigo = "def bitonic_sort(arr):\n    def bitonic_merge(arr, low, count, direction):\n        # Implementación de bitonic_merge\n        # ...\n\n    def bitonic_sort_helper(arr, low, count, direction):\n        # Implementación de bitonic_sort_helper\n        # ...\n\n    sorted_arr = arr.copy()\n    low = 0\n    count = len(sorted_arr)\n    direction = 1  # 1 para orden ascendente, -1 para orden descendente\n    bitonic_sort_helper(sorted_arr, low, count, direction)\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de Bitonic Sort se basa en la construcción y fusión de secuencias bitónicas. El algoritmo utiliza la propiedad de que las secuencias bitónicas pueden dividirse en mitades que son secuencias bitónicas en sí mismas. Luego, se realiza la fusión de estas secuencias hasta obtener una secuencia bitónica ordenada."

    elif metodo_seleccionado == "Block Sort":
        descripcion = "Block Sort es un algoritmo de ordenamiento que divide una lista en bloques de tamaño fijo y luego ordena cada bloque individualmente. Después de ordenar los bloques, se realiza una fusión para combinar los bloques en una única lista ordenada."
        codigo = "def block_sort(arr):\n    def insertion_sort(arr, low, high):\n        # Implementación de insertion_sort\n        # ...\n\n    def block_sort_helper(arr, low, high):\n        # Implementación de block_sort_helper\n        # ...\n\n    sorted_arr = arr.copy()\n    low = 0\n    high = len(sorted_arr) - 1\n    block_sort_helper(sorted_arr, low, high)\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de Block Sort se basa en dividir una lista en bloques de tamaño fijo y luego ordenar cada bloque individualmente utilizando un algoritmo de ordenamiento eficiente, como Insertion Sort. Después de ordenar los bloques, se realiza una fusión para combinar los bloques en una única lista ordenada."

    elif metodo_seleccionado == "Flash Sort":
        descripcion = "Flash Sort es un algoritmo de ordenamiento que utiliza la distribución de los elementos en una lista para lograr un ordenamiento eficiente. El algoritmo divide la lista en regiones y realiza una secuencia de redistribuciones basadas en el valor de los elementos. Flash Sort es especialmente eficiente para listas con elementos repetidos."
        codigo = "def flash_sort(arr):\n    # Implementación del algoritmo de Flash Sort\n    # ...\n\n    sorted_arr = arr.copy()\n    # Lógica de ordenamiento\n    # ...\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de Flash Sort se basa en la distribución y redistribución de los elementos en una lista. El algoritmo utiliza información sobre la distribución de los elementos para determinar su posición en la lista ordenada final, lo que evita comparaciones innecesarias y logra un ordenamiento eficiente."

    elif metodo_seleccionado == "Three-Way Merge Sort":
        descripcion = "Three-Way Merge Sort es una variante del algoritmo de Merge Sort que divide la lista en tres partes en lugar de dos. El proceso de ordenamiento se realiza de forma recursiva fusionando las tres partes en una única lista ordenada. Three-Way Merge Sort puede ser más eficiente que Merge Sort tradicional para ciertos tipos de datos."
        codigo = "def three_way_merge_sort(arr):\n    # Implementación del algoritmo de Three-Way Merge Sort\n    # ...\n\n    sorted_arr = arr.copy()\n    # Lógica de ordenamiento\n    # ...\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de Three-Way Merge Sort se basa en la división de la lista en tres partes y la fusión recursiva de estas partes en una única lista ordenada. El algoritmo aprovecha la capacidad de fusionar tres partes en lugar de dos para lograr un ordenamiento eficiente."

    elif metodo_seleccionado == "Counting Distribution Sort":
        descripcion = "Counting Distribution Sort es un algoritmo de ordenamiento que utiliza la técnica de distribución para ordenar los elementos de una lista. El algoritmo cuenta el número de ocurrencias de cada elemento y luego distribuye los elementos en la lista en función de esta información. Counting Distribution Sort es eficiente para listas con un rango limitado de elementos."
        codigo = "def counting_distribution_sort(arr):\n    # Implementación del algoritmo de Counting Distribution Sort\n    # ...\n\n    sorted_arr = arr.copy()\n    # Lógica de ordenamiento\n    # ...\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de Counting Distribution Sort se basa en contar el número de ocurrencias de cada elemento en una lista y distribuir los elementos en la lista en función de esta información. El algoritmo utiliza información sobre las ocurrencias de los elementos para determinar su posición en la lista ordenada final."

    elif metodo_seleccionado == "Index Sort":
        descripcion = "Index Sort es un algoritmo de ordenamiento que utiliza un arreglo de índices para ordenar los elementos de una lista. El algoritmo crea un arreglo de índices que representa el orden de los elementos y luego utiliza este arreglo para construir la lista ordenada."
        codigo = "def index_sort(arr):\n    # Implementación del algoritmo de Index Sort\n    # ...\n\n    sorted_arr = arr.copy()\n    # Lógica de ordenamiento\n    # ...\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de Index Sort se basa en la creación de un arreglo de índices que representa el orden de los elementos en una lista. El algoritmo utiliza este arreglo de índices para construir la lista ordenada final."

    elif metodo_seleccionado == "Stability Sort":
        descripcion = "Stability Sort es un algoritmo de ordenamiento que mantiene el orden relativo de elementos iguales durante el proceso de ordenamiento. Esto significa que si dos elementos son iguales antes del ordenamiento, se mantendrá el mismo orden relativo después del ordenamiento."
        codigo = "def stability_sort(arr):\n    # Implementación del algoritmo de Stability Sort\n    # ...\n\n    sorted_arr = arr.copy()\n    # Lógica de ordenamiento\n    # ...\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de Stability Sort se basa en el mantenimiento del orden relativo de elementos iguales durante el proceso de ordenamiento. El algoritmo utiliza técnicas especiales para garantizar que los elementos iguales conserven su orden relativo en la lista ordenada final."

    elif metodo_seleccionado == "Library Sort":
        descripcion = "Library Sort es un algoritmo de ordenamiento que utiliza una estructura de datos de biblioteca para organizar los elementos de una lista. El algoritmo utiliza técnicas de búsqueda y ordenamiento para construir y mantener la estructura de biblioteca, lo que permite un ordenamiento eficiente."
        codigo = "def library_sort(arr):\n    # Implementación del algoritmo de Library Sort\n    # ...\n\n    sorted_arr = arr.copy()\n    # Lógica de ordenamiento\n    # ...\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de Library Sort se basa en el uso de una estructura de datos de biblioteca para organizar los elementos de una lista. El algoritmo utiliza técnicas de búsqueda y ordenamiento para construir y mantener la estructura de biblioteca, lo que permite un ordenamiento eficiente."

    elif metodo_seleccionado == "Cocktail Sort":
        descripcion = "Cocktail Sort, también conocido como Bubble Sort bidireccional, es una variante del algoritmo de Bubble Sort que realiza un ordenamiento en ambas direcciones. El algoritmo compara y intercambia elementos adyacentes en ambas direcciones hasta que la lista esté completamente ordenada."
        codigo = "def cocktail_sort(arr):\n    # Implementación del algoritmo de Cocktail Sort\n    # ...\n\n    sorted_arr = arr.copy()\n    # Lógica de ordenamiento\n    # ...\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de Cocktail Sort se basa en la comparación y el intercambio de elementos adyacentes en ambas direcciones para lograr un ordenamiento. El algoritmo repite este proceso hasta que la lista esté completamente ordenada."

    elif metodo_seleccionado == "Distribution Sort":
        descripcion = "Distribution Sort es un algoritmo de ordenamiento que utiliza técnicas de distribución para organizar los elementos de una lista. El algoritmo distribuye los elementos en diferentes grupos o segmentos según sus valores y luego ordena cada grupo de forma independiente."
        codigo = "def distribution_sort(arr):\n    # Implementación del algoritmo de Distribution Sort\n    # ...\n\n    sorted_arr = arr.copy()\n    # Lógica de ordenamiento\n    # ...\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de Distribution Sort se basa en la distribución de los elementos en diferentes grupos o segmentos según sus valores. El algoritmo ordena cada grupo de forma independiente y luego combina los grupos en una única lista ordenada final."

    elif metodo_seleccionado == "Binary Insertion Sort":
        descripcion = "Binary Insertion Sort es una variante del algoritmo de Insertion Sort que utiliza una búsqueda binaria para encontrar la posición correcta de un elemento en la lista ordenada mientras se realiza el proceso de inserción. Esto resulta en un ordenamiento más eficiente en comparación con el Insertion Sort tradicional."
        codigo = "def binary_insertion_sort(arr):\n    # Implementación del algoritmo de Binary Insertion Sort\n    # ...\n\n    sorted_arr = arr.copy()\n    # Lógica de ordenamiento\n    # ...\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de Binary Insertion Sort se basa en la búsqueda binaria para encontrar la posición correcta de un elemento en la lista ordenada mientras se realiza el proceso de inserción. El algoritmo utiliza la propiedad de ordenamiento de la lista parcialmente ordenada para realizar una búsqueda más eficiente."

    elif metodo_seleccionado == "Cube Sort":
        descripcion = "Cube Sort es un algoritmo de ordenamiento que utiliza un enfoque tridimensional para organizar los elementos de una lista. El algoritmo divide la lista en cubos y luego los ordena en función de sus coordenadas tridimensionales. Esto permite un ordenamiento más eficiente en comparación con otros algoritmos."
        codigo = "def cube_sort(arr):\n    # Implementación del algoritmo de Cube Sort\n    # ...\n\n    sorted_arr = arr.copy()\n    # Lógica de ordenamiento\n    # ...\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de Cube Sort se basa en el enfoque tridimensional para organizar los elementos de una lista. El algoritmo divide la lista en cubos y utiliza las coordenadas tridimensionales para realizar un ordenamiento eficiente."

    elif metodo_seleccionado == "Polyphase Merge Sort":
        descripcion = "Polyphase Merge Sort es un algoritmo de ordenamiento externo que utiliza múltiples fases y fusiones para ordenar grandes conjuntos de datos que no caben en la memoria principal. El algoritmo divide el conjunto de datos en bloques y realiza fusiones parciales y finales para obtener la lista ordenada."
        codigo = "def polyphase_merge_sort(arr):\n    # Implementación del algoritmo de Polyphase Merge Sort\n    # ...\n\n    sorted_arr = arr.copy()\n    # Lógica de ordenamiento\n    # ...\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de Polyphase Merge Sort se basa en el uso de múltiples fases y fusiones para ordenar grandes conjuntos de datos. El algoritmo divide los datos en bloques, realiza fusiones parciales y finales y utiliza técnicas de ordenamiento por mezcla para obtener la lista ordenada."

    elif metodo_seleccionado == "Shaker Sort":
        descripcion = "Shaker Sort, también conocido como Cocktail Shaker Sort, es una variante del algoritmo de Bubble Sort que realiza múltiples pasadas de ordenamiento en ambas direcciones. El algoritmo compara y intercambia elementos adyacentes en ambas direcciones en cada pasada hasta que la lista esté completamente ordenada."
        codigo = "def shaker_sort(arr):\n    # Implementación del algoritmo de Shaker Sort\n    # ...\n\n    sorted_arr = arr.copy()\n    # Lógica de ordenamiento\n    # ...\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de Shaker Sort se basa en la comparación y el intercambio de elementos adyacentes en ambas direcciones en múltiples pasadas. El algoritmo repite este proceso hasta que la lista esté completamente ordenada."

    elif metodo_seleccionado == "Straight Merge Sort":
        descripcion = "Straight Merge Sort es una variante del algoritmo de Merge Sort que utiliza una estrategia de ordenamiento más eficiente para listas parcialmente ordenadas. El algoritmo identifica subsecuencias ordenadas y realiza fusiones directas entre ellas, evitando así el proceso de partición y fusión en cada etapa."
        codigo = "def straight_merge_sort(arr):\n    # Implementación del algoritmo de Straight Merge Sort\n    # ...\n\n    sorted_arr = arr.copy()\n    # Lógica de ordenamiento\n    # ...\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de Straight Merge Sort se basa en la identificación de subsecuencias ordenadas y la realización de fusiones directas entre ellas. El algoritmo utiliza técnicas para identificar y fusionar subsecuencias de forma eficiente, evitando el proceso de partición y fusión en cada etapa."

    elif metodo_seleccionado == "Stick Sort":
        descripcion = "Stick Sort es un algoritmo de ordenamiento que utiliza un enfoque basado en la construcción de estructuras de palitos para organizar los elementos de una lista. El algoritmo crea palitos de diferentes longitudes y los ordena en función de los elementos, generando así la lista ordenada."
        codigo = "def stick_sort(arr):\n    # Implementación del algoritmo de Stick Sort\n    # ...\n\n    sorted_arr = arr.copy()\n    # Lógica de ordenamiento\n    # ...\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de Stick Sort se basa en la construcción de estructuras de palitos de diferentes longitudes para organizar los elementos de una lista. El algoritmo asigna los elementos a los palitos correspondientes y genera la lista ordenada a partir de la longitud y posición de los palitos."

    elif metodo_seleccionado == "Red Sort":
        descripcion = "Red Sort es un algoritmo de ordenamiento que utiliza un enfoque basado en intercambios condicionales para organizar los elementos de una lista. El algoritmo realiza múltiples pasadas de intercambio y utiliza condiciones especiales para determinar los intercambios entre elementos."
        codigo = "def red_sort(arr):\n    # Implementación del algoritmo de Red Sort\n    # ...\n\n    sorted_arr = arr.copy()\n    # Lógica de ordenamiento\n    # ...\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de Red Sort se basa en el uso de intercambios condicionales para organizar los elementos de una lista. El algoritmo repite múltiples pasadas de intercambio y utiliza condiciones especiales para determinar los intercambios entre elementos."

    elif metodo_seleccionado == "Spiral Sort":
        descripcion = "Spiral Sort es un algoritmo de ordenamiento que utiliza una estrategia basada en espirales para organizar los elementos de una lista. El algoritmo crea espirales en función de los elementos y los ordena en un patrón espiral, generando así la lista ordenada."
        codigo = "def spiral_sort(arr):\n    # Implementación del algoritmo de Spiral Sort\n    # ...\n\n    sorted_arr = arr.copy()\n    # Lógica de ordenamiento\n    # ...\n\n    return sorted_arr"
        logica_matematica = "La lógica matemática detrás de Spiral Sort se basa en la creación de espirales en función de los elementos y el ordenamiento de los elementos en un patrón espiral. El algoritmo utiliza técnicas para generar las espirales y ordenar eficientemente los elementos."


    ventana_descripcion = tk.Toplevel(ventana)
    ventana_descripcion.title("Descripción del método de ordenamiento")
    ventana_descripcion.geometry("800x600")

    label_descripcion = ttk.Label(ventana_descripcion, text="Descripción:")
    label_descripcion.pack()

    text_descripcion = tk.Text(ventana_descripcion, height=10, width=80)
    text_descripcion.insert(tk.END, descripcion)
    text_descripcion.pack()

    label_codigo = ttk.Label(ventana_descripcion, text="Código:")
    label_codigo.pack()

    text_codigo = tk.Text(ventana_descripcion, height=15, width=80)
    text_codigo.insert(tk.END, codigo)
    text_codigo.pack()

    label_logica_matematica = ttk.Label(ventana_descripcion, text="Lógica Matemática:")
    label_logica_matematica.pack()

    text_logica_matematica = tk.Text(ventana_descripcion, height=20, width=80)
    text_logica_matematica.insert(tk.END, logica_matematica)
    text_logica_matematica.pack()

def validate_input(input_str):
    # Verificar si el input contiene solo números enteros separados por espacios
    if re.match(r'^(\d+\s+)*\d+$', input_str):
        return True
    else:
        return False

def generar_numeros_aleatorios():
    numeros_aleatorios = random.sample(range(1, 101), 15)  # Genera 15 números aleatorios del 1 al 100
    entrada_vector.delete(0, tk.END)  # Borra el contenido actual del ttk.Entry
    entrada_vector.insert(tk.END, " ".join(map(str, numeros_aleatorios)))  # Inserta los números aleatorios en el ttk.Entry

def ordenar():
    metodo_seleccionado = opcion_var.get()
    vector = entrada_vector.get()

    # Validar el campo de entrada
    if not validate_input(vector):
        messagebox.showerror("Error", "Ingresa solo números enteros separados por espacios")
        return

    # Convertir la cadena de entrada en una lista de números enteros
    vector = [int(num) for num in vector.split()]

    if metodo_seleccionado == "":
        messagebox.showerror("Error", "Selecciona un método de ordenamiento")
        return

    if metodo_seleccionado == "Bubble Sort":
        sorted_vector, sorted_steps = bubble_sort(vector)
    elif metodo_seleccionado == "Insertion Sort":
        sorted_vector, sorted_steps = insertion_sort(vector)
    elif metodo_seleccionado == "Selection Sort":
        sorted_vector, sorted_steps = selection_sort(vector)
    elif metodo_seleccionado == "Merge Sort":
        sorted_vector, sorted_steps = merge_sort(vector)
    elif metodo_seleccionado == "Quick Sort":
        sorted_vector, sorted_steps = quick_sort(vector)
    elif metodo_seleccionado == "Heap Sort":
        sorted_vector, sorted_steps = heap_sort(vector)
    elif metodo_seleccionado == "Counting Sort":
        sorted_vector, sorted_steps = counting_sort(vector)
    elif metodo_seleccionado == "Bucket Sort":
        sorted_vector, sorted_steps = bucket_sort(vector)
    elif metodo_seleccionado == "Radix Sort":
        sorted_vector, sorted_steps = radix_sort(vector)
    elif metodo_seleccionado == "Shell Sort":
        sorted_vector, sorted_steps = shell_sort(vector)
    elif metodo_seleccionado == "Binary Tree Sort":
        sorted_vector, sorted_steps = binary_tree_sort(vector)
    elif metodo_seleccionado == "Optimal Merge Sort":
        sorted_vector, sorted_steps = optimal_merge_sort(vector)
    elif metodo_seleccionado == "Natural Merge Sort":
        sorted_vector, sorted_steps = natural_merge_sort(vector)
    elif metodo_seleccionado == "Pigeonhole Sort":
        sorted_vector, sorted_steps = pigeonhole_sort(vector)
    elif metodo_seleccionado == "Card Sort":
        sorted_vector, sorted_steps = card_sort(vector)
    elif metodo_seleccionado == "Shellsort":
        sorted_vector, sorted_steps = shellsort(vector)
    elif metodo_seleccionado == "Smoothsort":
        sorted_vector, sorted_steps = smoothsort(vector)
    elif metodo_seleccionado == "Most Significant Digit Radix Sort":
        sorted_vector, sorted_steps = msd_radix_sort(vector)
    elif metodo_seleccionado == "Least Significant Digit Radix Sort":
        sorted_vector, sorted_steps = lsd_radix_sort(vector)
    elif metodo_seleccionado == "Comparison Sort":
        sorted_vector, sorted_steps = comparison_sort(vector)
    elif metodo_seleccionado == "Bitonic Sort":
        sorted_vector, sorted_steps = bitonic_sort(vector)
    elif metodo_seleccionado == "Block Sort":
        sorted_vector, sorted_steps = block_sort(vector)
    elif metodo_seleccionado == "Flash Sort":
        sorted_vector, sorted_steps = flash_sort(vector)
    elif metodo_seleccionado == "Three-Way Merge Sort":
        sorted_vector, sorted_steps = three_way_merge_sort(vector)
    elif metodo_seleccionado == "Counting Distribution Sort":
        sorted_vector, sorted_steps = counting_distribution_sort(vector)
    elif metodo_seleccionado == "Index Sort":
        sorted_vector, sorted_steps = index_sort(vector)
    elif metodo_seleccionado == "Stability Sort":
        sorted_vector, sorted_steps = stability_sort(vector)
    elif metodo_seleccionado == "Library Sort":
        sorted_vector, sorted_steps = library_sort(vector)
    elif metodo_seleccionado == "Cocktail Sort":
        sorted_vector, sorted_steps = cocktail_sort(vector)
    elif metodo_seleccionado == "Distribution Sort":
        sorted_vector, sorted_steps = distribution_sort(vector)
    elif metodo_seleccionado == "Binary Insertion Sort":
        sorted_vector, sorted_steps = binary_insertion_sort(vector)
    elif metodo_seleccionado == "Cube Sort":
        sorted_vector, sorted_steps = cube_sort(vector)
    elif metodo_seleccionado == "Polyphase Merge Sort":
        sorted_vector, sorted_steps = polyphase_merge_sort(vector)
    elif metodo_seleccionado == "Shaker Sort":
        sorted_vector, sorted_steps = shaker_sort(vector)
    elif metodo_seleccionado == "Straight Merge Sort":
        sorted_vector, sorted_steps = straight_merge_sort(vector)
    elif metodo_seleccionado == "Stick Sort":
        sorted_vector, sorted_steps = stick_sort(vector)
    elif metodo_seleccionado == "Red Sort":
        sorted_vector, sorted_steps = red_sort(vector)
    elif metodo_seleccionado == "Spiral Sort":
        sorted_vector, sorted_steps = spiral_sort(vector)

    messagebox.showinfo("Resultado", f"Vector ordenado: {sorted_vector}")
    etiqueta_detalle.config(text=metodo_seleccionado)
    vector_ordenado_str = ', '.join(str(elemento) for elemento in sorted_vector)
    etiqueta_resultado.config(text="Resultado: [" + vector_ordenado_str + "]")

    lista_pasos.delete(0, tk.END)
    for paso in sorted_steps:
        lista_pasos.insert(tk.END, str(paso))
        lista_pasos.yview(tk.END)

# Ventana principal
ventana = tk.Tk()
ventana.title("Ordenamiento")
ventana.geometry("1280x720")
ventana.configure(bg="#222222")
estilo = ttk.Style()
estilo.theme_use("clam")
estilo.configure(".", background="white", foreground="black", font=("Arial", 12))
estilo.configure("TLabel", foreground="black")
estilo.configure("TEntry", foreground="black")

# Funciones de ayuda y créditos
def mostrar_ayuda():
    messagebox.showinfo("Ayuda", "Esta aplicación te permite ordenar una lista de números utilizando diferentes métodos de ordenamiento.")

def mostrar_creditos():
    def redirigir_enlace(enlace):
        webbrowser.open(enlace)

    ventana_creditos = tk.Toplevel(ventana)
    ventana_creditos.title("Créditos")
    ventana_creditos.geometry("400x200")
    
    mensaje = "Esta aplicación fue desarrollada por Ruben David Renteria Cruz.\n\n"
    mensaje += "Redes de contacto:\n"

    etiqueta_creditos = tk.Label(ventana_creditos, text=mensaje, justify="left")
    etiqueta_creditos.pack(pady=10)

    etiqueta_linkedin = tk.Label(ventana_creditos, text="LinkedIn", fg="blue", cursor="hand2")
    etiqueta_linkedin.pack()
    etiqueta_linkedin.bind("<Button-1>", lambda event: redirigir_enlace("https://www.linkedin.com/in/rub%C3%A9n-david-renter%C3%ADa-cruz-1a329a207/"))

    etiqueta_github = tk.Label(ventana_creditos, text="GitHub", fg="blue", cursor="hand2")
    etiqueta_github.pack()
    etiqueta_github.bind("<Button-1>", lambda event: redirigir_enlace("https://github.com/Alpha0117"))
       
def mostrar_manual():
    manual_text = """
    Manual de Usuario: Programa de Ordenamiento
    Descripción
    El Programa de Ordenamiento es una aplicación que te permite ordenar una lista de números utilizando diferentes métodos de ordenamiento. La interfaz gráfica de la aplicación te permite seleccionar el método de ordenamiento deseado, ingresar la lista de números y ver el resultado ordenado paso a paso.

    Requisitos del sistema
    Sistema operativo: Windows, macOS o Linux.
    Python 3 instalado en el sistema.
    Librerías de Python: tkinter, ttk y sorting_algorithms.

    Instalación
    1. Descarga el archivo del programa de ordenamiento en tu computadora.
    2. Asegúrate de tener Python 3 instalado en tu sistema. Puedes descargarlo desde el sitio web oficial de Python.
    3. Instala las librerías necesarias ejecutando los siguientes comandos en la terminal:
        - pip install tkinter
        - pip install ttk
        - pip install sorting_algorithms
    4. Descomprime el archivo descargado y navega hasta la carpeta del programa.

    Iniciar el programa
    1. Abre una terminal en la carpeta del programa.
    2. Ejecuta el siguiente comando para iniciar la aplicación:
        - python programa_ordenamiento.py

    Uso del programa
    - Una vez que hayas iniciado el programa, verás la interfaz gráfica con las siguientes opciones:
        - Elije el método de ordenamiento: Selecciona el método de ordenamiento que deseas utilizar en el menú desplegable.
        - Por favor ingresar el vector: Ingresa los números que deseas ordenar, separados por espacios.
        - Generar: El botón "Generar" es una funcionalidad que te permite generar automáticamente una lista de 15 números aleatorios para ser ordenados. 
          Esta funcionalidad es útil cuando deseas probar el programa de ordenamiento con una lista predefinida de números aleatorios en lugar de ingresarlos manualmente. 
          Al presionar el botón "Generar", se generarán 15 números aleatorios y se colocarán automáticamente en el campo de entrada.
        - Ordenar: Haz clic en el botón "Ordenar" para iniciar el proceso de ordenamiento.
        - Resultado: Después de ordenar la lista, verás el resultado en la sección de "Resultado".
        - Pasos: En la sección de "Pasos", se muestra una lista de los pasos realizados durante el proceso de ordenamiento.
        - Menú de Ayuda: En la barra de menú, encontrarás la opción "Ayuda" que brinda información general sobre el programa.
        - Menú By: En la barra de menú, encontrarás la opción "By" que muestra los créditos y la información del desarrollador.
        - Menú Manual de Usuario: En la opción "Ayuda" del menú, encontrarás la opción "Manual de usuario" que muestra este manual.
        - Salir: En la barra de herramientas, hay un botón "Salir" para cerrar la aplicación.

    Consideraciones adicionales
    - Asegúrate de ingresar los números correctamente, separados por espacios, sin ningún otro carácter especial.
    - Si cometes un error, puedes borrar la lista y volver a ingresarla antes de hacer clic en "Ordenar".
    - Algunos métodos de ordenamiento pueden requerir más tiempo o recursos dependiendo del tamaño de la lista ingresada.

    ¡Disfruta utilizando el Programa de Ordenamiento y explora los diferentes métodos disponibles para ordenar tus listas de números de manera eficiente! Si tienes alguna pregunta o necesitas más información, consulta la sección de "Ayuda" o "Manual de usuario" en la aplicación.
    """

    messagebox.showinfo("Manual de usuario", manual_text)

def salir():
    ventana.quit()

# Barra de menú
barra_menu = tk.Menu(ventana)
menu_ayuda = tk.Menu(barra_menu, tearoff=0)
menu_ayuda.add_command(label="Ayuda", command=mostrar_ayuda)
menu_ayuda.add_command(label="Manual de usuario", command=mostrar_manual)
barra_menu.add_cascade(label="Ayuda", menu=menu_ayuda)

menu_by = tk.Menu(barra_menu, tearoff=0)
menu_by.add_command(label="By", command=mostrar_creditos)
barra_menu.add_cascade(label="By", menu=menu_by)

boton_salir = tk.Menu(barra_menu, tearoff=0)
boton_salir.add_command(label="Salir", command=salir)
barra_menu.add_cascade(label="Salir", menu=boton_salir)

ventana.config(menu=barra_menu)

#------------------------------------------------------------

#------------------------------------------------------------
frame_entrada = tk.Frame(ventana, bg="#222222")
frame_entrada.pack(pady=50)

etiqueta_metodo = ttk.Label(frame_entrada, text="Elije el método de ordenamiento:")
etiqueta_metodo.pack()

# Crear el ttk.OptionMenu
opcion_var = tk.StringVar()
opcion_menu = ttk.OptionMenu(frame_entrada, opcion_var, "", "", "Bubble Sort", "Insertion Sort", "Selection Sort", "Merge Sort", "Quick Sort", "Heap Sort", "Counting Sort", "Bucket Sort", "Radix Sort", "Shell Sort", "Binary Tree Sort", "Optimal Merge Sort", "Natural Merge Sort", "Pigeonhole Sort", "Card Sort", "Shellsort", "Smoothsort", "Most Significant Digit Radix Sort", "Least Significant Digit Radix Sort", "Comparison Sort", "Bitonic Sort", "Block Sort", "Flash Sort", "Three-Way Merge Sort", "Counting Distribution Sort", "Index Sort", "Stability Sort", "Library Sort", "Cocktail Sort", "Distribution Sort", "Binary Insertion Sort", "Cube Sort", "Polyphase Merge Sort", "Shaker Sort", "Straight Merge Sort", "Stick Sort", "Red Sort", "Spiral Sort")
opcion_menu.pack(pady=30)

# Crear el ttk.Label
label_vector = ttk.Label(frame_entrada, text="Por favor ingresar el vector \n(los numeros deben ir separados por espacios)")
label_vector.pack(pady=20)

entrada_vector = ttk.Entry(frame_entrada)
entrada_vector.pack(pady=10)

boton_generar = ttk.Button(frame_entrada, text="Generar", command=generar_numeros_aleatorios)
boton_generar.pack(side=tk.LEFT, padx=10)

boton_ordenar = ttk.Button(frame_entrada, text="Ordenar", command=ordenar)
boton_ordenar.pack()

frame_resultado = tk.Frame(ventana, bg="#222222")
frame_resultado.pack(pady=10)

etiqueta_resultado = ttk.Label(frame_resultado, text="Resultado:")
etiqueta_resultado.pack()

frame_pasos = tk.Frame(ventana, bg="#222222")
frame_pasos.pack(pady=10)

etiqueta_pasos = ttk.Label(frame_pasos, text="Pasos:")
etiqueta_pasos.pack()

lista_pasos = tk.Listbox(frame_pasos)
lista_pasos.pack()
lista_pasos.configure(width=0)  # Ajustar el ancho de la listbox al contenido

etiqueta_detalle = ttk.Label(ventana, text="")
etiqueta_detalle.pack(pady=10)

boton_descripcion = ttk.Button(ventana, text="Ver Descripción", command=lambda: mostrar_descripcion(opcion_var.get()))
boton_descripcion.pack()

ventana.mainloop()
