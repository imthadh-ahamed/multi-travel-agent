"""
System Status Check for Multi-Agent Travel Assistant
Quick verification that all components are working after fixes
"""

import requests
import time

def check_streamlit():
    """Check if Streamlit is running"""
    try:
        response = requests.get("http://localhost:8503", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_api():
    """Check if API is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    print("🔍 System Status Check")
    print("=" * 30)
    
    # Check API
    if check_api():
        print("✅ FastAPI Backend: Running (http://localhost:8000)")
    else:
        print("❌ FastAPI Backend: Not accessible")
    
    # Check Streamlit
    if check_streamlit():
        print("✅ Streamlit UI: Running (http://localhost:8503)")
    else:
        print("❌ Streamlit UI: Not accessible")
    
    print("\n🎯 **Ready to Test:**")
    print("1. Open http://localhost:8503 in your browser")
    print("2. Login with demo/demo123")
    print("3. Test travel planning features")
    print("4. Check dashboard and admin features")

if __name__ == "__main__":
    main()
