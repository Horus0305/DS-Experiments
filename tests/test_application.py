"""
Test suite for DS-Experiments Streamlit application
"""
import pytest
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEnvironment:
    """Test environment and dependencies"""
    
    def test_python_version(self):
        """Test Python version is 3.8 or higher"""
        assert sys.version_info >= (3, 8), "Python 3.8+ is required"
    
    def test_required_modules(self):
        """Test that all required modules can be imported"""
        required_modules = [
            'streamlit',
            'pandas',
            'numpy',
            'plotly',
            'sklearn',
            'scipy',
            'seaborn',
            'matplotlib'
        ]
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                pytest.fail(f"Required module '{module}' not found")


class TestDataFiles:
    """Test data file availability"""
    
    def test_readme_exists(self):
        """Test README.md exists"""
        assert os.path.exists('README.md'), "README.md not found"
    
    def test_requirements_exists(self):
        """Test requirements.txt exists"""
        assert os.path.exists('requirements.txt'), "requirements.txt not found"
    
    def test_main_app_exists(self):
        """Test main app.py exists"""
        assert os.path.exists('app.py'), "app.py not found"
    
    def test_pages_directory_exists(self):
        """Test pages directory exists"""
        assert os.path.isdir('pages'), "pages directory not found"
    
    def test_all_pages_exist(self):
        """Test all required page files exist"""
        required_pages = [
            '1_Introduction.py',
            '2_Data_Cleaning.py',
            '3_EDA_and_Statistical_Analysis.py',
            '4_ML_Modeling_and_Tracking.py',
            '5_Explainability_and_Fairness.py',
            '6_API_Deployment_and_Containerization.py'
        ]
        
        for page in required_pages:
            page_path = os.path.join('pages', page)
            assert os.path.exists(page_path), f"Page {page} not found"


class TestModelFiles:
    """Test model files availability"""
    
    def test_model_directory_exists(self):
        """Test model directory exists"""
        model_dir = 'dsmodelpickl+preprocessor'
        if not os.path.exists(model_dir):
            pytest.skip(f"Model directory {model_dir} not found (may be pulled by DVC)")
    
    def test_preprocessor_exists(self):
        """Test preprocessor pickle file exists"""
        preprocessor_path = os.path.join('dsmodelpickl+preprocessor', 'preprocessor.pkl')
        if not os.path.exists(preprocessor_path):
            pytest.skip("Preprocessor not found (may be pulled by DVC)")
    
    def test_model_files_exist(self):
        """Test that model pickle files exist"""
        model_dir = 'dsmodelpickl+preprocessor'
        if not os.path.exists(model_dir):
            pytest.skip("Model directory not found (may be pulled by DVC)")
        
        expected_models = [
            'knn_model.pkl',
            'ann_dnn_model.pkl',
            'logistic_regression_model.pkl',
            'naive_bayes_model.pkl',
            'svm_model.pkl',
            'lda_model.pkl'
        ]
        
        for model_file in expected_models:
            model_path = os.path.join(model_dir, model_file)
            if not os.path.exists(model_path):
                pytest.skip(f"Model {model_file} not found (may be pulled by DVC)")


class TestApplicationStructure:
    """Test application code structure"""
    
    def test_app_py_syntax(self):
        """Test app.py has valid Python syntax"""
        try:
            with open('app.py', 'r', encoding='utf-8') as f:
                compile(f.read(), 'app.py', 'exec')
        except SyntaxError as e:
            pytest.fail(f"Syntax error in app.py: {e}")
    
    def test_pages_syntax(self):
        """Test all page files have valid Python syntax"""
        pages_dir = 'pages'
        if not os.path.exists(pages_dir):
            pytest.skip("Pages directory not found")
        
        for filename in os.listdir(pages_dir):
            if filename.endswith('.py'):
                filepath = os.path.join(pages_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        compile(f.read(), filepath, 'exec')
                except SyntaxError as e:
                    pytest.fail(f"Syntax error in {filename}: {e}")


class TestMLflowIntegration:
    """Test MLflow database integration"""
    
    def test_mlruns_db_exists(self):
        """Test MLflow database exists"""
        if not os.path.exists('mlruns.db'):
            pytest.skip("mlruns.db not found (may be pulled by DVC)")
    
    def test_mlruns_db_readable(self):
        """Test MLflow database is readable"""
        if not os.path.exists('mlruns.db'):
            pytest.skip("mlruns.db not found")
        
        try:
            import sqlite3
            conn = sqlite3.connect('mlruns.db')
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            conn.close()
            
            # Check for essential MLflow tables
            table_names = [t[0] for t in tables]
            assert 'runs' in table_names, "MLflow 'runs' table not found"
            assert 'metrics' in table_names, "MLflow 'metrics' table not found"
            assert 'params' in table_names, "MLflow 'params' table not found"
        except Exception as e:
            pytest.fail(f"Error reading MLflow database: {e}")


class TestDataIntegrity:
    """Test data file integrity"""
    
    def test_csv_data_readable(self):
        """Test CSV data files are readable"""
        csv_files = [
            'uncleanedfirstdatasetcsv.csv',
            'data/WholeTruthFoodDataset-combined.csv'
        ]
        
        import pandas as pd
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    assert len(df) > 0, f"{csv_file} is empty"
                except Exception as e:
                    pytest.fail(f"Error reading {csv_file}: {e}")


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
