"""Simple test runner that executes test functions without pytest.
This allows running tests even when pytest isn't installed in the environment.
"""
import importlib
import sys
import traceback

TEST_MODULES = [
    "tests.test_utils",
    "tests.test_network",
]

failed = 0
passed = 0

for mod_name in TEST_MODULES:
    try:
        mod = importlib.import_module(mod_name)
    except Exception:
        print(f"Failed to import {mod_name}")
        traceback.print_exc()
        failed += 1
        continue
    for attr in dir(mod):
        if attr.startswith("test_") and callable(getattr(mod, attr)):
            fn = getattr(mod, attr)
            try:
                fn()
                print(f"PASS: {mod_name}.{attr}")
                passed += 1
            except AssertionError:
                print(f"FAIL: {mod_name}.{attr} (AssertionError)")
                traceback.print_exc()
                failed += 1
            except Exception:
                print(f"ERROR: {mod_name}.{attr} (Exception)")
                traceback.print_exc()
                failed += 1

print(f"\nSummary: {passed} passed, {failed} failed")
if failed:
    sys.exit(2)

