def bubble_sort(arr):
    n = len(arr)
    steps = []
    for i in range(n-1):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                steps.append(list(arr))
    return arr, steps

def insertion_sort(arr):
    n = len(arr)
    steps = []
    for i in range(1, n):
        key = arr[i]
        j = i-1
        while j >= 0 and arr[j] > key:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key
        steps.append(list(arr))
    return arr, steps

def selection_sort(arr):
    n = len(arr)
    steps = []
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
        steps.append(list(arr))
    return arr, steps

def merge_sort(arr):
    steps = []

    def merge(arr, left, mid, right):
        n1 = mid - left + 1
        n2 = right - mid

        L = [0] * n1
        R = [0] * n2

        for i in range(n1):
            L[i] = arr[left + i]

        for j in range(n2):
            R[j] = arr[mid + 1 + j]

        i = j = 0
        k = left

        while i < n1 and j < n2:
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
            steps.append(list(arr))

        while i < n1:
            arr[k] = L[i]
            i += 1
            k += 1
            steps.append(list(arr))

        while j < n2:
            arr[k] = R[j]
            j += 1
            k += 1
            steps.append(list(arr))

    def merge_sort_helper(arr, left, right):
        if left < right:
            mid = (left + right) // 2
            merge_sort_helper(arr, left, mid)
            merge_sort_helper(arr, mid + 1, right)
            merge(arr, left, mid, right)

    merge_sort_helper(arr, 0, len(arr) - 1)

    return arr, steps


def quick_sort(arr):
    steps = []

    def partition(arr, low, high):
        i = low - 1
        pivot = arr[high]

        for j in range(low, high):
            if arr[j] < pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
            steps.append(list(arr))

        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        steps.append(list(arr))

        return i + 1

    def quick_sort_helper(arr, low, high):
        if low < high:
            pi = partition(arr, low, high)
            quick_sort_helper(arr, low, pi - 1)
            quick_sort_helper(arr, pi + 1, high)

    quick_sort_helper(arr, 0, len(arr) - 1)

    return arr, steps


def heapify(arr, n, i):
    steps = []
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[i] < arr[left]:
        largest = left

    if right < n and arr[largest] < arr[right]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        steps.append(list(arr))
        steps.extend(heapify(arr, n, largest))

    return steps


def heap_sort(arr):
    steps = []
    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):
        steps.extend(heapify(arr, n, i))

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        steps.append(list(arr))
        steps.extend(heapify(arr, i, 0))

    return arr, steps


def counting_sort(arr):
    max_val = max(arr)
    counts = [0] * (max_val + 1)
    steps = []

    for num in arr:
        counts[num] += 1
        steps.append(list(arr))

    sorted_arr = []
    for i in range(len(counts)):
        sorted_arr.extend([i] * counts[i])
        steps.append(list(sorted_arr))

    return sorted_arr, steps


def bucket_sort(arr):
    min_val = min(arr)
    max_val = max(arr)
    bucket_range = int((max_val - min_val) / len(arr)) + 1
    steps = []

    buckets = [[] for _ in range(len(arr))]
    for num in arr:
        index = int((num - min_val) // bucket_range)
        buckets[index].append(num)
        steps.append(list(arr))

    sorted_arr = []
    for bucket in buckets:
        sorted_arr.extend(sorted(bucket))
        steps.append(list(sorted_arr))

    return sorted_arr, steps


def radix_sort(arr):
    max_val = max(arr)
    exp = 1
    steps = []

    while max_val // exp > 0:
        counting_sort_by_digit(arr, exp, steps)
        exp *= 10

    return arr, steps


def counting_sort_by_digit(arr, exp, steps):
    counts = [0] * 10

    for num in arr:
        digit = (num // exp) % 10
        counts[digit] += 1
        steps.append(list(arr))

    for i in range(1, 10):
        counts[i] += counts[i - 1]

    sorted_arr = [0] * len(arr)
    for i in range(len(arr) - 1, -1, -1):
        digit = (arr[i] // exp) % 10
        sorted_arr[counts[digit] - 1] = arr[i]
        counts[digit] -= 1
        steps.append(list(sorted_arr))

    for i in range(len(arr)):
        arr[i] = sorted_arr[i]


def shell_sort(arr):
    n = len(arr)
    gap = n // 2
    steps = []

    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
                steps.append(list(arr))
            arr[j] = temp
        gap //= 2

    return arr, steps


def binary_tree_sort(arr):
    class TreeNode:
        def __init__(self, val):
            self.val = val
            self.left = None
            self.right = None

    def insert(root, node):
        if root is None:
            return node
        if root.val < node.val:
            root.right = insert(root.right, node)
        else:
            root.left = insert(root.left, node)
        return root

    def inorder_traversal(root, sorted_arr, steps):
        if root:
            inorder_traversal(root.left, sorted_arr, steps)
            sorted_arr.append(root.val)
            steps.append(list(sorted_arr))
            inorder_traversal(root.right, sorted_arr, steps)

    steps = []
    sorted_arr = []
    root = None

    for num in arr:
        root = insert(root, TreeNode(num))
        steps.append(list(arr))

    inorder_traversal(root, sorted_arr, steps)

    return sorted_arr, steps


def merge(arr, left, mid, right):
    steps = []
    n1 = mid - left + 1
    n2 = right - mid

    L = [0] * n1
    R = [0] * n2

    for i in range(n1):
        L[i] = arr[left + i]

    for j in range(n2):
        R[j] = arr[mid + 1 + j]

    i = j = 0
    k = left

    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1
        steps.append(list(arr))

    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1
        steps.append(list(arr))

    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1
        steps.append(list(arr))

    return steps


def optimal_merge_sort(arr):
    steps = []

    def merge_sort(arr, left, right):
        if left < right:
            mid = (left + right) // 2
            merge_sort(arr, left, mid)
            merge_sort(arr, mid + 1, right)
            steps.extend(merge(arr, left, mid, right))

    merge_sort(arr, 0, len(arr) - 1)

    return arr, steps


def natural_merge_sort(arr):
    steps = []

    def merge(arr, left, mid, right):
        if mid >= right:
            return []

        temp = [0] * (right - left + 1)
        i = left
        j = mid + 1
        k = 0

        while i <= mid and j <= right:
            if arr[i] <= arr[j]:
                temp[k] = arr[i]
                i += 1
            else:
                temp[k] = arr[j]
                j += 1
            k += 1

        while i <= mid:
            temp[k] = arr[i]
            i += 1
            k += 1

        while j <= right:
            temp[k] = arr[j]
            j += 1
            k += 1

        for p in range(len(temp)):
            arr[left + p] = temp[p]
            steps.append(list(arr))

    def find_runs(arr):
        runs = []
        n = len(arr)
        i = 1
        while i < n:
            if arr[i - 1] > arr[i]:
                break
            i += 1

        if i == n:
            runs.append((0, n - 1))
            return runs

        start = 0
        while i < n:
            if arr[i - 1] <= arr[i]:
                i += 1
            else:
                runs.append((start, i - 1))
                start = i
                i += 1

        if start < n:
            runs.append((start, n - 1))

        return runs

    def merge_runs(arr, runs):
        while len(runs) > 1:
            new_runs = []
            i = 0
            while i < len(runs):
                if i + 1 < len(runs):
                    merge(arr, runs[i][0], runs[i][1], runs[i + 1][1])
                    new_runs.append((runs[i][0], runs[i + 1][1]))
                    i += 2
                else:
                    new_runs.append(runs[i])
                    i += 1
            runs = new_runs
            steps.append(list(arr))

    runs = find_runs(arr)
    merge_runs(arr, runs)

    return arr, steps


def pigeonhole_sort(arr):
    min_val = min(arr)
    max_val = max(arr)
    size = max_val - min_val + 1
    steps = []

    holes = [0] * size

    for num in arr:
        holes[num - min_val] += 1
        steps.append(list(arr))

    sorted_arr = []
    for i in range(size):
        sorted_arr.extend([i + min_val] * holes[i])
        steps.append(list(sorted_arr))

    return sorted_arr, steps


def card_sort(arr):
    steps = []

    def insertion_sort(arr, start, gap):
        for i in range(start + gap, len(arr), gap):
            key = arr[i]
            j = i - gap
            while j >= start and arr[j] > key:
                arr[j + gap] = arr[j]
                j -= gap
            arr[j + gap] = key
            steps.append(list(arr))

    n = len(arr)
    gap = n // 2

    while gap > 0:
        for i in range(gap):
            insertion_sort(arr, i, gap)
        gap //= 2

    return arr, steps


def shellsort(arr):
    steps = []
    n = len(arr)
    gap = n // 2

    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
            steps.append(list(arr))
        gap //= 2

    return arr, steps

def smoothsort(arr):
    def down_heap(start, end, r):
        nonlocal sorted_arr, steps
        while 2 * start + 1 < end:
            child = 2 * start + 1
            if child + 1 < end and sorted_arr[child] < sorted_arr[child + 1]:
                child += 1
            if sorted_arr[start] < sorted_arr[child]:
                sorted_arr[start], sorted_arr[child] = sorted_arr[child], sorted_arr[start]
                steps.append(list(sorted_arr[:]))  # Agregar una copia de la lista ordenada
                start = child
            else:
                break

    def sift(start, size, r):
        nonlocal sorted_arr
        while start + r < size:
            down_heap(start, start + r + 1, r)
            start += 1

    def trinkle(start, p, r):
        nonlocal sorted_arr
        while p > 0:
            while p & 1 == 1:
                p >>= 1
                r += 1
            sift(start - r, start, r)
            r -= 1
            p >>= 1

    def semitrinkle(start, p, r):
        nonlocal sorted_arr
        if p & 1 == 1:
            sift(start - r, start, r)
        trinkle(start, p // 2, r)

    def smooth_sort_helper(start, size, r):
        nonlocal sorted_arr, steps
        if size < 2:
            return

        q = size // 2
        p = 1
        while p <= q:
            p <<= 1
            r += 1
        p >>= 1
        q = p

        while q > 1:
            if p > 3 and (p - 1) % 3 == 0:
                semitrinkle(start, p - 1, r)
            else:
                trinkle(start, p - 1, r)
            p = q
            q >>= 1
            r -= 1

        sift(start, start + p, r)
        steps.append(list(sorted_arr[:]))  # Agregar una copia de la lista ordenada al final

    steps = []
    r = 0
    sorted_arr = arr[:]
    smooth_sort_helper(0, len(sorted_arr), r)

    return sorted_arr, steps


def msd_radix_sort(arr):
    steps = []

    def counting_sort_by_digit(arr, exp, steps):
        counts = [0] * 10

        for num in arr:
            digit = (num // exp) % 10
            counts[digit] += 1
            steps.append(list(arr))

        for i in range(1, 10):
            counts[i] += counts[i - 1]

        sorted_arr = [0] * len(arr)
        for i in range(len(arr) - 1, -1, -1):
            digit = (arr[i] // exp) % 10
            sorted_arr[counts[digit] - 1] = arr[i]
            counts[digit] -= 1
            steps.append(list(sorted_arr))

        for i in range(len(arr)):
            arr[i] = sorted_arr[i]

    max_val = max(arr)
    exp = 1
    while max_val // exp > 0:
        counting_sort_by_digit(arr, exp, steps)
        exp *= 10

    return arr, steps


def lsd_radix_sort(arr):
    steps = []

    def counting_sort_by_digit(arr, exp, steps):
        counts = [0] * 10

        for num in arr:
            digit = (num // exp) % 10
            counts[digit] += 1
            steps.append(list(arr))

        for i in range(1, 10):
            counts[i] += counts[i - 1]

        sorted_arr = [0] * len(arr)
        for i in range(len(arr) - 1, -1, -1):
            digit = (arr[i] // exp) % 10
            sorted_arr[counts[digit] - 1] = arr[i]
            counts[digit] -= 1
            steps.append(list(sorted_arr))

        for i in range(len(arr)):
            arr[i] = sorted_arr[i]

    max_val = max(arr)
    exp = 1
    while max_val // exp > 0:
        counting_sort_by_digit(arr, exp, steps)
        exp *= 10

    return arr, steps


def comparison_sort(arr):
    steps = []
    sorted_arr = list(arr)
    n = len(sorted_arr)

    for i in range(n - 1):
        for j in range(i + 1, n):
            steps.append((sorted_arr[i], sorted_arr[j]))  # Agregar la comparación actual
            if sorted_arr[i] > sorted_arr[j]:
                sorted_arr[i], sorted_arr[j] = sorted_arr[j], sorted_arr[i]

    return sorted_arr, steps


def bitonic_sort(arr):
    steps = []

    def bitonic_merge(arr, low, cnt, direction, steps):
        if cnt > 1:
            k = cnt // 2
            for i in range(low, low + k):
                if direction == 1:
                    if arr[i] > arr[i + k]:
                        arr[i], arr[i + k] = arr[i + k], arr[i]
                        steps.append(list(arr))
                else:
                    if arr[i] < arr[i + k]:
                        arr[i], arr[i + k] = arr[i + k], arr[i]
                        steps.append(list(arr))
            bitonic_merge(arr, low, k, direction, steps)
            bitonic_merge(arr, low + k, k, direction, steps)

    def bitonic_sort_helper(arr, low, cnt, direction, steps):
        if cnt > 1:
            k = cnt // 2
            bitonic_sort_helper(arr, low, k, 1, steps)
            bitonic_sort_helper(arr, low + k, k, 0, steps)
            bitonic_merge(arr, low, cnt, direction, steps)

    n = len(arr)
    bitonic_sort_helper(arr, 0, n, 1, steps)

    return arr, steps


def block_sort(arr):
    steps = []
    n = len(arr)
    block_size = int(n ** 0.5)
    
    for i in range(0, n, block_size):
        if i + block_size < n:
            arr[i:i+block_size] = sorted(arr[i:i+block_size])
        else:
            arr[i:] = sorted(arr[i:])
        steps.append(list(arr))

    return arr, steps


def flash_sort(arr):
    steps = []

    def insertion_sort(arr, start, end):
        for i in range(start + 1, end + 1):
            key = arr[i]
            j = i - 1
            while j >= start and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

    n = len(arr)
    m = int(0.45 * n)
    k = 0
    steps.append(list(arr))

    if n < 2:
        return arr, steps

    max_val = arr[0]
    min_val = arr[0]
    for i in range(1, n):
        if arr[i] > max_val:
            max_val = arr[i]
        if arr[i] < min_val:
            min_val = arr[i]

    c = (m - 1) / (max_val - min_val)
    count = [0] * m
    for i in range(n):
        index = int(c * (arr[i] - min_val))
        count[index] += 1

    for i in range(1, m):
        count[i] += count[i - 1]

    while k < m - 1 and count[k] <= n:
        k += 1

    for i in range(m):
        count[i] -= 1

    j = int(c * (n - 1))
    while k > 0:
        while j >= count[k]:
            j -= 1
            index = int(c * (arr[j] - min_val))
            arr[count[index]] += arr[j]
            arr[j] = arr[count[index]] - arr[j]
            arr[count[index]] -= arr[j]
            steps.append(list(arr))
        k -= 1

    for i in range(1, m):
        insertion_sort(arr, count[i - 1], count[i])
        steps.append(list(arr))

    return arr, steps


def three_way_merge_sort(arr):
    def merge(arr, left, mid1, mid2, right):
        n1 = mid1 - left + 1
        n2 = mid2 - mid1
        n3 = right - mid2

        L = arr[left:mid1+1]
        M = arr[mid1+1:mid2+1]
        R = arr[mid2+1:right+1]

        i = j = k = 0
        p = q = r = 0

        while i < n1 and j < n2 and k < n3:
            if L[i] <= M[j] and L[i] <= R[k]:
                arr[left+p] = L[i]
                i += 1
            elif M[j] <= L[i] and M[j] <= R[k]:
                arr[left+p] = M[j]
                j += 1
            else:
                arr[left+p] = R[k]
                k += 1
            p += 1

        while i < n1 and j < n2:
            if L[i] <= M[j]:
                arr[left+p] = L[i]
                i += 1
            else:
                arr[left+p] = M[j]
                j += 1
            p += 1

        while i < n1 and k < n3:
            if L[i] <= R[k]:
                arr[left+p] = L[i]
                i += 1
            else:
                arr[left+p] = R[k]
                k += 1
            p += 1

        while j < n2 and k < n3:
            if M[j] <= R[k]:
                arr[left+p] = M[j]
                j += 1
            else:
                arr[left+p] = R[k]
                k += 1
            p += 1

        while i < n1:
            arr[left+p] = L[i]
            i += 1
            p += 1

        while j < n2:
            arr[left+p] = M[j]
            j += 1
            p += 1

        while k < n3:
            arr[left+p] = R[k]
            k += 1
            p += 1

    def merge_sort(arr, left, right):
        if left < right:
            mid1 = left + (right - left) // 3
            mid2 = left + 2 * (right - left) // 3

            merge_sort(arr, left, mid1)
            merge_sort(arr, mid1+1, mid2)
            merge_sort(arr, mid2+1, right)

            merge(arr, left, mid1, mid2, right)
            steps.append(list(arr))  # Registrar el estado del vector en cada paso

    steps = []
    merge_sort(arr, 0, len(arr) - 1)
    return arr, steps


def counting_distribution_sort(arr):
    steps = []
    min_val = min(arr)
    max_val = max(arr)
    counts = [0] * (max_val - min_val + 1)
    sorted_arr = [0] * len(arr)

    for num in arr:
        counts[num - min_val] += 1
        steps.append(list(arr))

    for i in range(1, len(counts)):
        counts[i] += counts[i - 1]

    for num in reversed(arr):
        index = counts[num - min_val] - 1
        sorted_arr[index] = num
        counts[num - min_val] -= 1
        steps.append(list(sorted_arr))

    return sorted_arr, steps


def index_sort(arr):
    steps = []
    min_val = min(arr)
    max_val = max(arr)
    range_val = max_val - min_val + 1
    counts = [0] * range_val
    sorted_arr = [0] * len(arr)

    for num in arr:
        counts[num - min_val] += 1
        steps.append(list(arr))

    for i in range(1, range_val):
        counts[i] += counts[i - 1]

    for num in arr:
        index = counts[num - min_val] - 1
        sorted_arr[index] = num
        counts[num - min_val] -= 1
        steps.append(list(sorted_arr))

    return sorted_arr, steps


def stability_sort(arr):
    steps = []
    max_val = max(arr)
    min_val = min(arr)
    range_val = max_val - min_val + 1
    counts = [0] * range_val
    sorted_arr = [0] * len(arr)

    for num in arr:
        counts[num - min_val] += 1
        steps.append(list(arr))

    for i in range(1, range_val):
        counts[i] += counts[i - 1]

    for i in range(len(arr) - 1, -1, -1):
        index = counts[arr[i] - min_val] - 1
        sorted_arr[index] = arr[i]
        counts[arr[i] - min_val] -= 1
        steps.append(list(sorted_arr))

    return sorted_arr, steps


def library_sort(arr):
    steps = []
    n = len(arr)
    sorted_arr = sorted(arr)
    steps.append(list(sorted_arr))

    while arr != sorted_arr:
        for i in range(n - 1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                steps.append(list(arr))

    return arr, steps


def cocktail_sort(arr):
    steps = []
    n = len(arr)
    swapped = True
    start = 0
    end = n - 1

    while swapped:
        swapped = False
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                steps.append(list(arr))
                swapped = True

        if not swapped:
            break

        swapped = False
        end -= 1

        for i in range(end - 1, start - 1, -1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                steps.append(list(arr))
                swapped = True

        start += 1

    return arr, steps


def distribution_sort(arr):
    steps = []
    min_val = min(arr)
    max_val = max(arr)
    counts = [0] * (max_val - min_val + 1)
    sorted_arr = [0] * len(arr)

    for num in arr:
        counts[num - min_val] += 1
        steps.append(list(arr))

    sorted_index = 0
    for i in range(len(counts)):
        for _ in range(counts[i]):
            sorted_arr[sorted_index] = i + min_val
            sorted_index += 1
            steps.append(list(sorted_arr))

    return sorted_arr, steps

# Binary Insertion Sort
def binary_insertion_sort(arr):
    steps = []
    for i in range(1, len(arr)):
        key = arr[i]
        left = 0
        right = i - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] > key:
                right = mid - 1
            else:
                left = mid + 1
        for j in range(i, left, -1):
            arr[j] = arr[j - 1]
        arr[left] = key
        steps.append(list(arr))
    return arr, steps

# Cube Sort
def cube_sort(arr):
    steps = []
    def swap(a, b):
        arr[a], arr[b] = arr[b], arr[a]
        steps.append(list(arr))
    n = len(arr)
    gap = n // 2
    while gap >= 1:
        for i in range(gap, n):
            for j in range(i - gap, -1, -gap):
                if arr[j] > arr[j + gap]:
                    swap(j, j + gap)
        gap //= 2
    return arr, steps

# Polyphase Merge Sort
def polyphase_merge_sort(arr):
    steps = []
    def merge(a, b, left, right):
        i = left
        j = right
        merged = []
        while i < len(a) and j < len(b):
            if a[i] < b[j]:
                merged.append(a[i])
                i += 1
            else:
                merged.append(b[j])
                j += 1
        merged += a[i:]
        merged += b[j:]
        return merged

    def merge_pass(arr):
        n = len(arr)
        merged = []
        i = 0
        while i < n:
            if i + 1 < n:
                merged += merge(arr[i], arr[i + 1], 0, 0)
                i += 2
            else:
                merged += arr[i]
                i += 1
        return merged

    n = len(arr)
    k = 1
    while k < n:
        groups = [arr[i:i + k] for i in range(0, n, k)]
        arr = merge_pass(groups)
        steps.append(list(arr))
        k *= 2
    return arr, steps

# Shaker Sort
def shaker_sort(arr):
    steps = []
    n = len(arr)
    left = 0
    right = n - 1
    while left <= right:
        for i in range(left, right):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                steps.append(list(arr))
        right -= 1
        for i in range(right, left, -1):
            if arr[i] < arr[i - 1]:
                arr[i], arr[i - 1] = arr[i - 1], arr[i]
                steps.append(list(arr))
        left += 1
    return arr, steps

# Straight Merge Sort
def straight_merge_sort(arr):
    steps = []
    def merge(left, right, start, mid, end):
        merged = []
        i = start
        j = mid
        while i < mid and j < end:
            if arr[i] <= arr[j]:
                merged.append(arr[i])
                i += 1
            else:
                merged.append(arr[j])
                j += 1
        merged += arr[i:mid]
        merged += arr[j:end]
        for i in range(len(merged)):
            arr[start + i] = merged[i]
        steps.append(list(arr))

    def merge_pass(n, length):
        i = 0
        while i + 2 * length <= n:
            merge(0, 2 * length, i, i + length, i + 2 * length)
            i += 2 * length
        if i + length < n:
            merge(0, length, i, i + length, n)
        else:
            for j in range(i, n):
                arr[j] = arr[j]
            steps.append(list(arr))

    n = len(arr)
    length = 1
    while length < n:
        merge_pass(n, length)
        length *= 2
    return arr, steps


# Stick Sort
def stick_sort(arr):
    steps = []
    n = len(arr)
    i = 0
    while i < n - 1:
        if arr[i] <= arr[i + 1]:
            i += 1
        else:
            arr[i], arr[i + 1] = arr[i + 1], arr[i]
            steps.append(list(arr))
            if i > 0:
                i -= 1
    return arr, steps

# Red Sort
def red_sort(arr):
    steps = []

    def swap(i, j):
        arr[i], arr[j] = arr[j], arr[i]
        steps.append(list(arr))

    def partition(arr, low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                swap(i, j)
        swap(i + 1, high)
        return i + 1

    def red_quick_sort(arr, low, high):
        if low < high:
            pi = partition(arr, low, high)
            red_quick_sort(arr, low, pi - 1)
            red_quick_sort(arr, pi + 1, high)

    red_quick_sort(arr, 0, len(arr) - 1)
    return arr, steps


# Spiral Sort
def spiral_sort(arr):
    steps = []

    def spiral(arr, left, top, right, bottom):
        if left >= right or top >= bottom:
            return

        # Move from left to right
        for i in range(left, right + 1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                steps.append(list(arr))

        # Move from top to bottom
        for i in range(top, bottom + 1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                steps.append(list(arr))

        # Move from right to left
        for i in range(right, left - 1, -1):
            if arr[i] > arr[i - 1]:
                arr[i], arr[i - 1] = arr[i - 1], arr[i]
                steps.append(list(arr))

        # Move from bottom to top
        for i in range(bottom, top - 1, -1):
            if arr[i] > arr[i - 1]:
                arr[i], arr[i - 1] = arr[i - 1], arr[i]
                steps.append(list(arr))

        # Recurse on inner spiral
        spiral(arr, left + 1, top + 1, right - 1, bottom - 1)

    spiral(arr, 0, 0, len(arr) - 1, len(arr) - 1)
    return arr, steps
