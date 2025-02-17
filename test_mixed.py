# test_mixed.py

def simple_function(x, y):
    """Fonction avec docstring correcte."""
    return x + y

def complex_function(a, b):
    # Fonction sans docstring, avec une complexité accrue
    result = 0
    for i in range(10):
        if i % 2 == 0:
            result += i * a
        else:
            result -= i * b
        for j in range(5):
            if j % 2 == 0:
                result += j
            else:
                result -= j
    return result

class SampleClass:
    """Classe avec docstring et quelques méthodes."""
    def method1(self):
        return "Méthode 1"

    def method2(self):
        # Ajout de complexité dans la méthode
        total = 0
        for i in range(3):
            for j in range(3):
                total += i * j
        return total
