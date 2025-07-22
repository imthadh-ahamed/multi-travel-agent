"""
Task 12: FastAPI Backend for Production Deployment
Goal: Deploy system using FastAPI backend with role-based access and logging
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import jwt
import bcrypt
from fastapi import FastAPI, HTTPException, Depends, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Database imports (using SQLAlchemy with async support)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, DateTime, Text, Float, Integer, Boolean, select
import redis.asyncio as redis

# Import our Phase 4 components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.phase4.production_coordinator import ProductionCoordinator, TaskPriority
from agents.phase4.rag_knowledge_base import RAGKnowledgeBase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class Settings:
    SECRET_KEY = os.getenv("SECRET_KEY", "travel-agent-secret-key-change-in-production")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost/travel_agent")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Rate limiting
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # 1 hour

settings = Settings()

# Database Models
class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(20), default="guest")  # guest, user, admin
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

class TravelRequest(Base):
    __tablename__ = "travel_requests"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer)
    session_id: Mapped[str] = mapped_column(String(100))
    request_text: Mapped[str] = mapped_column(Text)
    response_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    processing_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

# Pydantic Models
class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: str = Field(min_length=5, max_length=100)
    password: str = Field(min_length=6, max_length=100)

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None

class TravelRequestCreate(BaseModel):
    request_text: str = Field(min_length=10, max_length=5000)
    priority: str = Field(default="medium", pattern="^(low|medium|high|urgent|critical)$")

class TravelRequestResponse(BaseModel):
    id: int
    session_id: str
    request_text: str
    response_text: Optional[str] = None
    status: str
    quality_score: Optional[float] = None
    processing_time: Optional[float] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user: UserResponse

class SystemStatusResponse(BaseModel):
    status: str
    uptime: float
    total_requests: int
    active_sessions: int
    system_health: str
    performance_metrics: Dict[str, Any]

# Global application state
app_state = {
    'coordinator': None,
    'start_time': datetime.utcnow(),
    'request_count': 0,
    'redis_client': None
}

# Database setup
engine = create_async_engine(settings.DATABASE_URL, echo=False)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)

async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Travel Agent API...")
    
    # Initialize database
    await init_db()
    
    # Initialize Redis
    app_state['redis_client'] = redis.from_url(settings.REDIS_URL)
    
    # Initialize Production Coordinator
    app_state['coordinator'] = ProductionCoordinator(enable_auto_improvement=True)
    await app_state['coordinator'].initialize_system()
    
    logger.info("✅ API startup complete")
    
    yield
    
    # Shutdown
    logger.info("📴 Shutting down API...")
    if app_state['redis_client']:
        await app_state['redis_client'].close()

# Create FastAPI app
app = FastAPI(
    title="Multi-Agent Travel Assistant API",
    description="Production-ready travel planning API with AI agents",
    version="4.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiting middleware"""
    if app_state['redis_client']:
        client_ip = request.client.host
        key = f"rate_limit:{client_ip}"
        
        try:
            current = await app_state['redis_client'].get(key)
            if current is None:
                await app_state['redis_client'].setex(key, settings.RATE_LIMIT_WINDOW, 1)
            else:
                current_count = int(current)
                if current_count >= settings.RATE_LIMIT_REQUESTS:
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Rate limit exceeded. Try again later."}
                    )
                await app_state['redis_client'].incr(key)
        except Exception as e:
            logger.warning(f"Rate limiting error: {e}")
    
    response = await call_next(request)
    return response

# Request logging middleware
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()
    app_state['request_count'] += 1
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response

# Authentication utilities
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password: str) -> str:
    """Generate password hash"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    
    async with async_session_maker() as session:
        result = await session.execute(select(User).filter(User.username == username))
        user = result.scalar_one_or_none()
        
        if user is None:
            raise credentials_exception
        
        return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def require_role(required_role: str):
    """Dependency to require specific role"""
    def role_checker(current_user: User = Depends(get_current_active_user)):
        role_hierarchy = {"guest": 0, "user": 1, "admin": 2}
        user_level = role_hierarchy.get(current_user.role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return role_checker

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Multi-Agent Travel Assistant API",
        "version": "4.0.0",
        "status": "operational",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    uptime = (datetime.utcnow() - app_state['start_time']).total_seconds()
    
    # Check system components
    coordinator_status = "healthy" if app_state['coordinator'] else "unavailable"
    redis_status = "healthy" if app_state['redis_client'] else "unavailable"
    
    return {
        "status": "healthy",
        "uptime": uptime,
        "components": {
            "coordinator": coordinator_status,
            "redis": redis_status,
            "database": "healthy"  # Assume healthy if we got this far
        },
        "request_count": app_state['request_count'],
        "timestamp": datetime.utcnow()
    }

# Authentication endpoints
@app.post("/auth/register", response_model=UserResponse)
async def register_user(user_data: UserCreate):
    """Register a new user"""
    async with async_session_maker() as session:
        # Check if user exists
        result = await session.execute(
            select(User).filter(
                (User.username == user_data.username) | (User.email == user_data.email)
            )
        )
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Username or email already registered"
            )
        
        # Create new user
        hashed_password = get_password_hash(user_data.password)
        new_user = User(
            username=user_data.username,
            email=user_data.email,
            hashed_password=hashed_password,
            role="user"  # Default role
        )
        
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)
        
        logger.info(f"New user registered: {user_data.username}")
        
        return UserResponse(
            id=new_user.id,
            username=new_user.username,
            email=new_user.email,
            role=new_user.role,
            is_active=new_user.is_active,
            created_at=new_user.created_at,
            last_login=new_user.last_login
        )

@app.post("/auth/login", response_model=TokenResponse)
async def login_user(user_credentials: UserLogin):
    """Login user and return access token"""
    async with async_session_maker() as session:
        result = await session.execute(select(User).filter(User.username == user_credentials.username))
        user = result.scalar_one_or_none()
        
        if not user or not verify_password(user_credentials.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is disabled"
            )
        
        # Update last login
        user.last_login = datetime.utcnow()
        await session.commit()
        
        # Create access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        
        logger.info(f"User logged in: {user.username}")
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse(
                id=user.id,
                username=user.username,
                email=user.email,
                role=user.role,
                is_active=user.is_active,
                created_at=user.created_at,
                last_login=user.last_login
            )
        )

@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user information"""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        role=current_user.role,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )

# Travel planning endpoints
@app.post("/travel/plan", response_model=TravelRequestResponse)
async def create_travel_plan(
    request_data: TravelRequestCreate,
    current_user: User = Depends(get_current_active_user)
):
    """Create a new travel plan request"""
    if not app_state['coordinator']:
        raise HTTPException(
            status_code=503,
            detail="Travel planning service is currently unavailable"
        )
    
    # Map priority string to enum
    priority_mapping = {
        "low": TaskPriority.LOW,
        "medium": TaskPriority.MEDIUM,
        "high": TaskPriority.HIGH,
        "urgent": TaskPriority.URGENT,
        "critical": TaskPriority.CRITICAL
    }
    priority = priority_mapping.get(request_data.priority, TaskPriority.MEDIUM)
    
    # Create database record
    async with async_session_maker() as session:
        travel_request = TravelRequest(
            user_id=current_user.id,
            session_id=f"session_{current_user.id}_{int(time.time())}",
            request_text=request_data.request_text,
            status="processing"
        )
        
        session.add(travel_request)
        await session.commit()
        await session.refresh(travel_request)
        
        request_id = travel_request.id
        session_id = travel_request.session_id
    
    try:
        # Process with coordinator
        logger.info(f"Processing travel request {request_id} for user {current_user.username}")
        
        result = await app_state['coordinator'].process_travel_request(
            request_data.request_text,
            user_id=str(current_user.id),
            session_id=session_id,
            priority=priority
        )
        
        # Update database with results
        async with async_session_maker() as session:
            result_obj = await session.get(TravelRequest, request_id)
            if result_obj:
                result_obj.response_text = result.get('answer', 'No response generated')
                result_obj.status = "completed" if result.get('success', False) else "failed"
                result_obj.quality_score = result.get('quality_scores', {}).get('average', 0.8)
                result_obj.processing_time = result.get('processing_time', 0.0)
                result_obj.completed_at = datetime.utcnow()
                
                await session.commit()
                await session.refresh(result_obj)
                
                return TravelRequestResponse(
                    id=result_obj.id,
                    session_id=result_obj.session_id,
                    request_text=result_obj.request_text,
                    response_text=result_obj.response_text,
                    status=result_obj.status,
                    quality_score=result_obj.quality_score,
                    processing_time=result_obj.processing_time,
                    created_at=result_obj.created_at,
                    completed_at=result_obj.completed_at
                )
        
    except Exception as e:
        logger.error(f"Failed to process travel request {request_id}: {e}")
        
        # Update database with error
        async with async_session_maker() as session:
            result_obj = await session.get(TravelRequest, request_id)
            if result_obj:
                result_obj.status = "failed"
                result_obj.response_text = f"Processing failed: {str(e)}"
                result_obj.completed_at = datetime.utcnow()
                await session.commit()
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process travel request: {str(e)}"
        )

@app.get("/travel/requests", response_model=List[TravelRequestResponse])
async def get_user_travel_requests(
    current_user: User = Depends(get_current_active_user),
    limit: int = 10,
    offset: int = 0
):
    """Get user's travel requests"""
    async with async_session_maker() as session:
        result = await session.execute(
            select(TravelRequest)
            .filter(TravelRequest.user_id == current_user.id)
            .order_by(TravelRequest.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        requests = result.scalars().all()
        
        return [
            TravelRequestResponse(
                id=req.id,
                session_id=req.session_id,
                request_text=req.request_text,
                response_text=req.response_text,
                status=req.status,
                quality_score=req.quality_score,
                processing_time=req.processing_time,
                created_at=req.created_at,
                completed_at=req.completed_at
            )
            for req in requests
        ]

@app.get("/travel/request/{request_id}", response_model=TravelRequestResponse)
async def get_travel_request(
    request_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """Get specific travel request"""
    async with async_session_maker() as session:
        result = await session.execute(
            select(TravelRequest).filter(
                TravelRequest.id == request_id,
                TravelRequest.user_id == current_user.id
            )
        )
        travel_request = result.scalar_one_or_none()
        
        if not travel_request:
            raise HTTPException(status_code=404, detail="Travel request not found")
        
        return TravelRequestResponse(
            id=travel_request.id,
            session_id=travel_request.session_id,
            request_text=travel_request.request_text,
            response_text=travel_request.response_text,
            status=travel_request.status,
            quality_score=travel_request.quality_score,
            processing_time=travel_request.processing_time,
            created_at=travel_request.created_at,
            completed_at=travel_request.completed_at
        )

# Admin endpoints
@app.get("/admin/system-status", response_model=SystemStatusResponse)
async def get_system_status(current_user: User = Depends(require_role("admin"))):
    """Get comprehensive system status (admin only)"""
    if not app_state['coordinator']:
        raise HTTPException(status_code=503, detail="Coordinator unavailable")
    
    uptime = (datetime.utcnow() - app_state['start_time']).total_seconds()
    
    # Get coordinator status
    coordinator_status = await app_state['coordinator'].get_system_status()
    
    # Get active sessions from Redis
    active_sessions = 0
    if app_state['redis_client']:
        try:
            # Count active rate limit keys as proxy for active sessions
            keys = await app_state['redis_client'].keys("rate_limit:*")
            active_sessions = len(keys)
        except Exception:
            active_sessions = 0
    
    return SystemStatusResponse(
        status="operational",
        uptime=uptime,
        total_requests=app_state['request_count'],
        active_sessions=active_sessions,
        system_health=coordinator_status.get('system_health', 'unknown'),
        performance_metrics=coordinator_status.get('performance_metrics', {})
    )

@app.post("/admin/knowledge")
async def add_knowledge(
    title: str,
    content: str,
    category: str = "custom",
    current_user: User = Depends(require_role("admin"))
):
    """Add knowledge to the system (admin only)"""
    if not app_state['coordinator']:
        raise HTTPException(status_code=503, detail="Coordinator unavailable")
    
    try:
        doc_id = await app_state['coordinator'].add_knowledge_source(title, content, category)
        logger.info(f"Admin {current_user.username} added knowledge: {title}")
        
        return {
            "message": "Knowledge added successfully",
            "document_id": doc_id,
            "title": title,
            "category": category
        }
        
    except Exception as e:
        logger.error(f"Failed to add knowledge: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add knowledge: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )

@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
