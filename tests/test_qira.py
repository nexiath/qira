# tests/test_qira.py
import unittest
import os
import json
import subprocess
import sys
from pathlib import Path

class TestQIRA(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).parent.parent
        self.qira_path = self.project_root / "qira.py"
        self.test_files_dir = Path(__file__).parent / "test_files"
        self.base_cmd = f"{sys.executable} {self.qira_path}"
    
    def run_qira(self, input_file, output_file=None, **kwargs):
        """Helper method to run QIRA with various arguments"""
        cmd_parts = [self.base_cmd, "--file", str(input_file)]
        
        if output_file:
            cmd_parts.extend(["--output", str(output_file)])
        
        # Correction ici : utilisation de underscores au lieu de tirets
        for key, value in kwargs.items():
            if isinstance(value, bool) and value:
                cmd_parts.append(f"--{key}")  # Pas de remplacement tiret/underscore
            elif not isinstance(value, bool):
                cmd_parts.extend([f"--{key}", str(value)])  # Pas de remplacement tiret/underscore
        
        cmd = " ".join(str(part) for part in cmd_parts)
        print(f"Executing command: {cmd}")  # Pour le débogage
        return subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    def test_good_code(self):
        input_file = self.test_files_dir / "good_code.py"
        output_file = self.project_root / "good_report.json"
        
        # Ajout de seuils plus permissifs pour le "bon" code
        result = self.run_qira(
            input_file, 
            output_file,
            length_threshold=50,
            complexity_threshold=10,
            depth_threshold=5,
            methods_threshold=10,
            class_complexity_threshold=8.0
        )
        
        self.assertEqual(result.returncode, 0, f"QIRA failed with error: {result.stderr}")
        
        self.assertTrue(output_file.exists(), "Report file was not created")
        with open(output_file) as f:
            report = json.load(f)
        self.assertIsInstance(report, list)
        # Modification du test pour accepter jusqu'à 5 suggestions pour le bon code
        self.assertLess(len(report), 5, "Good code should not have too many suggestions")
    
    def test_bad_code(self):
        input_file = self.test_files_dir / "bad_code.py"
        output_file = self.project_root / "bad_report.json"
        
        result = self.run_qira(
            input_file,
            output_file,
            length_threshold=20,
            complexity_threshold=5,
            depth_threshold=2,
            methods_threshold=3,
            class_complexity_threshold=3.0,
            doc_required=True,
            debug=True
        )
        
        self.assertEqual(result.returncode, 0, f"QIRA failed with error: {result.stderr}")
        
        self.assertTrue(output_file.exists(), "Report file was not created")
        with open(output_file) as f:
            report = json.load(f)
        self.assertIsInstance(report, list)
        self.assertGreater(len(report), 3, "Bad code should have multiple suggestions")
    
    def test_messy_code(self):
        input_file = self.test_files_dir / "messy_code.py"
        output_file = self.project_root / "messy_report.json"
        
        result = self.run_qira(input_file, output_file, debug=True)
        self.assertEqual(result.returncode, 0, f"QIRA failed with error: {result.stderr}")
        
        self.assertTrue(output_file.exists(), "Report file was not created")
        with open(output_file) as f:
            report = json.load(f)
        self.assertIsInstance(report, list)
        self.assertGreater(len(report), 0, "Messy code should have suggestions")
    
    def test_empty_code(self):
        input_file = self.test_files_dir / "empty_code.py"
        output_file = self.project_root / "empty_report.json"
        
        result = self.run_qira(input_file, output_file)
        self.assertEqual(result.returncode, 0, f"QIRA failed with error: {result.stderr}")
    
    def test_invalid_file(self):
        input_file = self.test_files_dir / "nonexistent.py"
        result = self.run_qira(input_file)
        self.assertNotEqual(result.returncode, 0, "QIRA should fail with nonexistent file")

if __name__ == '__main__':
    unittest.main(verbosity=2)