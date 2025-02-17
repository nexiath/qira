# test_class.py

class TooManyMethods:
    # Classe sans docstring avec trop de méthodes pour tester la refactorisation

    def method1(self):
        pass

    def method2(self):
        pass

    def method3(self):
        pass

    def method4(self):
        pass

    def method5(self):
        pass

    def method6(self):
        # Méthode avec un peu de complexité
        result = 0
        for i in range(5):
            if i % 2 == 0:
                result += i
            else:
                result -= i
        return result
