import os

tests = ["greeting", "wit", "authority", "comfort", "technical", "peak"]

for test in tests:
    os.system(f"python test_jin.py default {test}")
