"""
Un exemple de code bien structuré avec documentation.
"""

class Calculator:
    """Une classe simple de calculatrice."""
    
    def __init__(self):
        """Initialise la calculatrice."""
        self.result = 0
    
    def add(self, a, b):
        """Addition de deux nombres."""
        return a + b
    
    def subtract(self, a, b):
        """Soustraction de deux nombres."""
        return a - b

def say_hello(name):
    """Dit bonjour à quelqu'un."""
    return f"Hello, {name}!"

# tests/test_files/bad_code.py
class ComplexCalculator:
    def __init__(self):
        self.memory = []
        self.result = 0
        self.operations = []
        self.history = {}
        self.error_count = 0
    
    def complex_operation(self, x, y, operation_type, precision=2, retry_count=3, 
                         save_history=True, validate=True, apply_rounding=True):
        if validate:
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                self.error_count += 1
                if save_history:
                    self.history['errors'] = self.history.get('errors', 0) + 1
                if retry_count > 0:
                    return self.complex_operation(float(x), float(y), operation_type,
                                               precision, retry_count - 1)
                else:
                    return None
        
        result = None
        if operation_type == 'add':
            result = x + y
        elif operation_type == 'subtract':
            result = x - y
        elif operation_type == 'multiply':
            result = x * y
        elif operation_type == 'divide':
            if y != 0:
                result = x / y
            else:
                if save_history:
                    self.history['div_by_zero'] = self.history.get('div_by_zero', 0) + 1
                return None
        
        if result is not None and apply_rounding:
            result = round(result, precision)
        
        if save_history:
            self.memory.append(result)
            self.operations.append({
                'x': x,
                'y': y,
                'operation': operation_type,
                'result': result,
                'precision': precision
            })
        
        return result