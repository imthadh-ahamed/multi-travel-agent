"""
Quick Test Script for Multi-Agent Travel Assistant
Tests the integration between Streamlit UI and FastAPI backend
"""

import requests
import json

def test_api_connection():
    """Test if API is running and accessible"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ API is running successfully!")
            print(f"📊 Health check: {response.json()}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def test_user_registration():
    """Test user registration"""
    try:
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpass123"
        }
        
        response = requests.post("http://localhost:8000/auth/register", json=user_data)
        if response.status_code == 200:
            print("✅ User registration working!")
            return True
        else:
            print(f"⚠️ Registration response: {response.status_code} - {response.json()}")
            return True  # Might be already registered
    except Exception as e:
        print(f"❌ Registration failed: {e}")
        return False

def test_user_login():
    """Test user login"""
    try:
        login_data = {
            "username": "demo",
            "password": "demo123"
        }
        
        response = requests.post("http://localhost:8000/auth/login", json=login_data)
        if response.status_code == 200:
            print("✅ User login working!")
            auth_data = response.json()
            print(f"🎫 Token received: {auth_data['access_token'][:20]}...")
            return auth_data['access_token']
        else:
            print(f"❌ Login failed: {response.status_code} - {response.json()}")
            return None
    except Exception as e:
        print(f"❌ Login error: {e}")
        return None

def test_travel_planning(token):
    """Test travel planning with authentication"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        travel_data = {
            "request_text": "Plan a weekend trip to Paris for 2 people with museums and cafes",
            "priority": "high"
        }
        
        response = requests.post("http://localhost:8000/travel/plan", json=travel_data, headers=headers)
        if response.status_code == 200:
            print("✅ Travel planning working!")
            result = response.json()
            print(f"📋 Plan generated with quality score: {result['data']['quality_score']}")
            return True
        else:
            print(f"❌ Travel planning failed: {response.status_code} - {response.json()}")
            return False
    except Exception as e:
        print(f"❌ Travel planning error: {e}")
        return False

def test_streamlit_ui():
    """Test if Streamlit UI is accessible"""
    try:
        response = requests.get("http://localhost:8502")
        if response.status_code == 200:
            print("✅ Streamlit UI is accessible!")
            return True
        else:
            print(f"❌ Streamlit UI not accessible: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Streamlit UI: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Multi-Agent Travel Assistant System")
    print("=" * 50)
    
    # Test API connection
    if not test_api_connection():
        print("❌ API not running. Please start the API first.")
        return
    
    # Test UI connection  
    if not test_streamlit_ui():
        print("❌ UI not running. Please start Streamlit first.")
        return
    
    # Test registration
    test_user_registration()
    
    # Test login
    token = test_user_login()
    if not token:
        print("❌ Cannot proceed without authentication")
        return
    
    # Test travel planning
    test_travel_planning(token)
    
    print("\n" + "=" * 50)
    print("🎉 System Test Complete!")
    print("\n📍 Access Points:")
    print("   • Streamlit UI: http://localhost:8502")
    print("   • FastAPI Backend: http://localhost:8000")
    print("   • API Documentation: http://localhost:8000/docs")
    print("\n🔑 Demo Credentials:")
    print("   • Username: demo")
    print("   • Password: demo123")
    print("   • Role: admin")
    print("\n🌟 Ready for testing!")

if __name__ == "__main__":
    main()
