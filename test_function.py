# test_function.py

def too_long_function(x):
    # Fonction sans docstring qui est trop longue pour tester l'extraction et la complexité
    result = 0
    # Ajout de nombreuses lignes pour dépasser le seuil de longueur (par défaut 30 lignes)
    a = 0
    b = 1
    c = 2
    d = 3
    e = 4
    f = 5
    g = 6
    h = 7
    i = 8
    j = 9
    k = 10
    l = 11
    m = 12
    n = 13
    o = 14
    p = 15
    q = 16
    r = 17
    s = 18
    t = 19

    # Boucle pour ajouter de la complexité et des lignes
    for index in range(10):
        if index % 2 == 0:
            result += index
        else:
            result -= index
        # Ajout de lignes supplémentaires
        x_val = index * 2
        y_val = index + 3
        z_val = x_val + y_val
        print("Index:", index, "x:", x_val, "y:", y_val, "z:", z_val)

    # Quelques lignes supplémentaires pour allonger la fonction
    a = 100
    b = 200
    c = a + b
    d = c * 2
    e = d - a
    f = e // 3
    print("Final result:", result, f)
    return result
