import unittest
import sys

def run_tests():
    """Run all test cases in the project."""
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = '.'  # Start in the current directory
    suite = loader.discover(start_dir, pattern="test_*.py")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return 0 if successful, 1 otherwise (for CI systems)
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_tests())
