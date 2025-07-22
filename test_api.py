"""
Minimal FastAPI backend for testing the Streamlit UI
This is a simplified version that mocks the main API functionality
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import jwt
import time
from datetime import datetime, timedelta
import uvicorn

# Configuration
SECRET_KEY = "test-secret-key-for-development"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# FastAPI app
app = FastAPI(
    title="Multi-Agent Travel Assistant API",
    description="Production-ready travel planning API with AI agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8502", "http://localhost:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class TravelRequest(BaseModel):
    request_text: str
    priority: str = "medium"

class User(BaseModel):
    id: int
    username: str
    email: str
    role: str = "user"

# Mock database
users_db = {
    "demo": {
        "id": 1,
        "username": "demo",
        "email": "demo@example.com",
        "password": "demo123",  # In production, this would be hashed
        "role": "admin"
    },
    "user": {
        "id": 2,
        "username": "user",
        "email": "user@example.com", 
        "password": "user123",
        "role": "user"
    }
}

travel_requests_db = []

# Utility functions
def create_access_token(data: dict):
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
        return username
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

def get_current_user(username: str = Depends(verify_token)):
    """Get current authenticated user"""
    user = users_db.get(username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    return user

# API Routes

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "uptime": 3600,  # Mock uptime
        "request_count": len(travel_requests_db)
    }

@app.post("/auth/register")
async def register(user: UserCreate):
    """Register a new user"""
    if user.username in users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # In production, password would be hashed
    users_db[user.username] = {
        "id": len(users_db) + 1,
        "username": user.username,
        "email": user.email,
        "password": user.password,
        "role": "user"
    }
    
    return {"message": "User registered successfully"}

@app.post("/auth/login")
async def login(user: UserLogin):
    """Login user and return access token"""
    db_user = users_db.get(user.username)
    if not db_user or db_user["password"] != user.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token = create_access_token(data={"sub": user.username})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": db_user["id"],
            "username": db_user["username"],
            "email": db_user["email"],
            "role": db_user["role"]
        }
    }

@app.post("/travel/plan")
async def plan_travel(
    request: TravelRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a travel planning request"""
    
    # Mock travel planning response
    travel_response = {
        "id": len(travel_requests_db) + 1,
        "user_id": current_user["id"],
        "request_text": request.request_text,
        "priority": request.priority,
        "status": "completed",
        "response_text": f"""
# 🌍 Your Travel Plan

Thank you for your travel request! Here's a personalized itinerary based on your requirements:

## 📋 Trip Summary
**Request:** {request.request_text}

## ✈️ Suggested Itinerary

### Day 1-2: Arrival & Exploration
- **Accommodation:** Premium hotel in city center
- **Activities:** Welcome dinner, city walking tour
- **Highlights:** Local attractions and cultural sites

### Day 3-4: Adventure & Culture  
- **Activities:** Museum visits, local food experiences
- **Transportation:** Local guided tours
- **Special:** Sunset viewing at scenic location

### Day 5-6: Relaxation & Shopping
- **Activities:** Spa treatments, shopping districts
- **Dining:** Recommended restaurants and cafes
- **Leisure:** Free time for personal exploration

## 💰 Budget Breakdown
- **Accommodation:** $150-250/night
- **Activities:** $50-100/day
- **Meals:** $40-80/day
- **Transportation:** $20-50/day

## 📞 Support
Our AI agents have crafted this plan using advanced algorithms and real-time data. 
For modifications, please create a new request or contact support.

*Generated by Multi-Agent Travel Assistant v1.0*
        """,
        "quality_score": 0.92,
        "processing_time": 2.5,
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": datetime.utcnow().isoformat()
    }
    
    travel_requests_db.append(travel_response)
    
    return {
        "success": True,
        "request_id": travel_response["id"],
        "message": "Travel plan generated successfully",
        "data": travel_response
    }

@app.get("/travel/requests")
async def get_travel_requests(current_user: dict = Depends(get_current_user)):
    """Get user's travel requests"""
    user_requests = [req for req in travel_requests_db if req["user_id"] == current_user["id"]]
    return user_requests

@app.get("/admin/system-status")
async def get_system_status(current_user: dict = Depends(get_current_user)):
    """Get system status (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return {
        "system_health": "excellent",
        "uptime": 7200,
        "total_requests": len(travel_requests_db),
        "active_sessions": 3,
        "performance_metrics": {
            "avg_response_time": 2.1,
            "success_rate": 0.98,
            "total_users": len(users_db),
            "requests_per_minute": 5.2,
            "system_load": 0.45
        }
    }

@app.post("/admin/knowledge")
async def add_knowledge(
    title: str,
    content: str,
    category: str = "general",
    current_user: dict = Depends(get_current_user)
):
    """Add knowledge to the system (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return {"message": f"Knowledge '{title}' added successfully to {category} category"}

if __name__ == "__main__":
    print("🚀 Starting Multi-Agent Travel Assistant API...")
    print("📍 API will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔄 CORS enabled for Streamlit UI")
    
    uvicorn.run(
        "test_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
