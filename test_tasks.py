#!/usr/bin/env python
"""Test script to run tasks.py directly without Celery."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

from Backend.tasks import process_loan_case
from Backend.db import init_db

if __name__ == "__main__":
    try:
        init_db()
        print("✅ Database initialized")
        
        # Create a mock self object for the task
        class MockSelf:
            def update_state(self, state=None, meta=None):
                print(f"  [Task State] {state} - {meta}")
        
        # Run the task directly
        case_id = "CASE001"
        print(f"\n▶ Running process_loan_case('{case_id}')...")
        result = process_loan_case.run(case_id)
        
        print("\n✅ Task completed successfully!")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
