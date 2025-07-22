"""
Streamlit UI for Multi-Agent Travel Assistant
Modern, interactive web interface for the production system
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64

# Configure page
st.set_page_config(
    page_title="Multi-Agent Travel Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
STREAMLIT_THEME = os.getenv("UI_THEME", "light")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Session state initialization
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
if 'user_info' not in st.session_state:
    st.session_state.user_info = None
if 'travel_requests' not in st.session_state:
    st.session_state.travel_requests = []
if 'current_request' not in st.session_state:
    st.session_state.current_request = None

# Custom CSS
def load_css():
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .travel-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        text-align: center;
    }
    
    .status-completed {
        background-color: #d4edda;
        color: #155724;
    }
    
    .status-processing {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .status-failed {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    
    .assistant-message {
        background-color: #f3e5f5;
        margin-right: 20%;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #5a6fd8 0%, #6a4190 100%);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)

# API Helper Functions
class APIClient:
    def __init__(self, base_url: str, token: str = None):
        self.base_url = base_url
        self.headers = {'Content-Type': 'application/json'}
        if token:
            self.headers['Authorization'] = f'Bearer {token}'
    
    def post(self, endpoint: str, data: dict) -> requests.Response:
        return requests.post(f"{self.base_url}{endpoint}", json=data, headers=self.headers)
    
    def get(self, endpoint: str) -> requests.Response:
        return requests.get(f"{self.base_url}{endpoint}", headers=self.headers)

def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticate user and return token info"""
    try:
        client = APIClient(API_BASE_URL)
        response = client.post("/auth/login", {"username": username, "password": password})
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Login failed: {response.json().get('detail', 'Unknown error')}")
            return None
            
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None

def register_user(username: str, email: str, password: str) -> bool:
    """Register a new user"""
    try:
        client = APIClient(API_BASE_URL)
        response = client.post("/auth/register", {
            "username": username,
            "email": email,
            "password": password
        })
        
        if response.status_code == 200:
            st.success("Registration successful! Please log in.")
            return True
        else:
            st.error(f"Registration failed: {response.json().get('detail', 'Unknown error')}")
            return False
            
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return False

def create_travel_request(request_text: str, priority: str = "medium") -> Optional[Dict]:
    """Create a new travel planning request"""
    try:
        client = APIClient(API_BASE_URL, st.session_state.access_token)
        response = client.post("/travel/plan", {
            "request_text": request_text,
            "priority": priority
        })
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Request failed: {response.json().get('detail', 'Unknown error')}")
            return None
            
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None

def get_travel_requests() -> List[Dict]:
    """Get user's travel requests"""
    try:
        client = APIClient(API_BASE_URL, st.session_state.access_token)
        response = client.get("/travel/requests")
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch requests: {response.json().get('detail', 'Unknown error')}")
            return []
            
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return []

def get_system_status() -> Optional[Dict]:
    """Get system status (admin only)"""
    try:
        client = APIClient(API_BASE_URL, st.session_state.access_token)
        response = client.get("/admin/system-status")
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except Exception as e:
        return None

# UI Components
def render_header():
    """Render main header"""
    st.markdown("""
    <div class="main-header">
        <h1>✈️ Multi-Agent Travel Assistant</h1>
        <p>Powered by AI Agents with Auto-Improvement & Knowledge Base</p>
    </div>
    """, unsafe_allow_html=True)

def render_login_page():
    """Render login/registration page"""
    st.title("🔐 Authentication")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login to Your Account")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit and username and password:
                auth_data = authenticate_user(username, password)
                if auth_data:
                    st.session_state.access_token = auth_data['access_token']
                    st.session_state.user_info = auth_data['user']
                    st.success("Login successful!")
                    st.rerun()
    
    with tab2:
        st.subheader("Create New Account")
        with st.form("register_form"):
            reg_username = st.text_input("Username", key="reg_username")
            reg_email = st.text_input("Email", key="reg_email")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_confirm = st.text_input("Confirm Password", type="password")
            submit_reg = st.form_submit_button("Register")
            
            if submit_reg:
                if reg_password != reg_confirm:
                    st.error("Passwords do not match")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters")
                elif reg_username and reg_email and reg_password:
                    register_user(reg_username, reg_email, reg_password)

def render_sidebar():
    """Render sidebar navigation"""
    if st.session_state.user_info:
        st.sidebar.markdown(f"**Welcome, {st.session_state.user_info['username']}!**")
        st.sidebar.markdown(f"Role: {st.session_state.user_info['role']}")
        
        if st.sidebar.button("Logout"):
            st.session_state.access_token = None
            st.session_state.user_info = None
            st.session_state.travel_requests = []
            st.rerun()
        
        st.sidebar.markdown("---")
        
        # Navigation menu
        menu_options = ["New Request", "My Requests", "Dashboard", "Admin Panel"] if st.session_state.user_info.get('role') == 'admin' else ["New Request", "My Requests", "Dashboard"]
        selected = st.sidebar.selectbox(
            "🧭 Navigation",
            options=menu_options,
            index=0,
            key="nav_menu"
        )
        
        return selected
    
    return None

def render_new_request_page():
    """Render new travel request page"""
    st.header("🌍 Plan Your Next Adventure")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Tell us about your travel plans")
        
        # Quick templates
        st.markdown("**Quick Templates:**")
        templates = {
            "Beach Vacation": "Plan a relaxing 7-day beach vacation for 2 people with a budget of $3000",
            "City Break": "Plan a 3-day city break with cultural activities and good restaurants",
            "Adventure Trip": "Plan an adventure-packed 10-day trip with hiking and outdoor activities",
            "Business Travel": "Plan efficient business travel with comfortable accommodation and meeting facilities"
        }
        
        template_cols = st.columns(len(templates))
        selected_template = ""
        for i, (name, template) in enumerate(templates.items()):
            with template_cols[i]:
                if st.button(name, key=f"template_{i}"):
                    selected_template = template
        
        # Main request form
        with st.form("travel_request_form"):
            request_text = st.text_area(
                "Describe your travel needs:",
                value=selected_template or st.session_state.get('request_template', ''),
                height=150,
                placeholder="E.g., I want to plan a 5-day trip to Japan for 2 people in April. We love cultural sites and great food. Budget is around $4000..."
            )
            
            col_priority, col_submit = st.columns([1, 1])
            with col_priority:
                priority = st.selectbox("Priority", ["low", "medium", "high", "urgent"])
            
            with col_submit:
                st.markdown("<br>", unsafe_allow_html=True)
                submit = st.form_submit_button("🚀 Plan My Trip", use_container_width=True)
            
            if submit and request_text:
                with st.spinner("🤖 Our AI agents are working on your travel plan..."):
                    result = create_travel_request(request_text, priority)
                    
                    if result:
                        st.success("Travel request submitted successfully!")
                        st.session_state.current_request = result
                        
                        # Display result
                        st.markdown("---")
                        render_travel_response(result)
    
    with col2:
        st.subheader("💡 Tips for Better Results")
        
        tips = [
            "🎯 Be specific about your destination and dates",
            "💰 Include your budget range",
            "👥 Mention number of travelers",
            "🎨 Share your interests and preferences",
            "📅 Specify trip duration",
            "🏨 Mention accommodation preferences",
            "✈️ Include departure location for flights"
        ]
        
        for tip in tips:
            st.markdown(f"• {tip}")
        
        st.markdown("---")
        st.subheader("🔄 System Status")
        
        # Try to get system health
        try:
            health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if health_response.status_code == 200:
                health_data = health_response.json()
                st.success("✅ System Online")
                st.metric("Uptime", f"{health_data.get('uptime', 0):.0f}s")
                st.metric("Requests Processed", health_data.get('request_count', 0))
            else:
                st.warning("⚠️ System Status Unknown")
        except:
            st.error("❌ System Offline")

def render_travel_response(response: Dict):
    """Render a travel planning response"""
    st.markdown('<div class="travel-card">', unsafe_allow_html=True)
    
    # Status badge
    status = response.get('status', 'unknown')
    status_class = f"status-{status}"
    st.markdown(f'<span class="status-badge {status_class}">{status.upper()}</span>', unsafe_allow_html=True)
    
    # Response content
    if response.get('response_text'):
        st.markdown("### 📋 Your Travel Plan")
        st.markdown(response['response_text'])
    
    # Metadata
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if response.get('processing_time'):
            st.metric("Processing Time", f"{response['processing_time']:.1f}s")
    
    with col2:
        if response.get('quality_score'):
            st.metric("Quality Score", f"{response['quality_score']:.2f}")
    
    with col3:
        created_at = response.get('created_at', '')
        if created_at:
            st.metric("Created", created_at[:10])  # Show just date
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_my_requests_page():
    """Render user's travel requests page"""
    st.header("📋 My Travel Requests")
    
    # Fetch requests
    if st.button("🔄 Refresh Requests"):
        st.session_state.travel_requests = get_travel_requests()
    
    if not st.session_state.travel_requests:
        st.session_state.travel_requests = get_travel_requests()
    
    if st.session_state.travel_requests:
        # Summary metrics
        total_requests = len(st.session_state.travel_requests)
        completed_requests = len([r for r in st.session_state.travel_requests if r['status'] == 'completed'])
        avg_quality = sum(r.get('quality_score', 0) for r in st.session_state.travel_requests if r.get('quality_score')) / max(1, len([r for r in st.session_state.travel_requests if r.get('quality_score')]))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Requests", total_requests)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Completed", completed_requests)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg Quality", f"{avg_quality:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Filter options
        col_filter, col_sort = st.columns(2)
        with col_filter:
            status_filter = st.selectbox("Filter by Status", ["all", "completed", "processing", "failed"])
        
        with col_sort:
            sort_by = st.selectbox("Sort by", ["created_at", "quality_score", "processing_time"])
        
        # Apply filters
        filtered_requests = st.session_state.travel_requests
        if status_filter != "all":
            filtered_requests = [r for r in filtered_requests if r['status'] == status_filter]
        
        # Sort requests
        filtered_requests = sorted(filtered_requests, key=lambda x: x.get(sort_by, 0), reverse=True)
        
        # Display requests
        for request in filtered_requests:
            render_travel_response(request)
            
            # Show request text
            with st.expander(f"Original Request - {request['created_at'][:10]}"):
                st.text(request['request_text'])
    
    else:
        st.info("🌟 No travel requests yet. Create your first travel plan!")
        if st.button("➕ Create New Request"):
            st.info("✨ Click 'New Request' in the sidebar to create your first travel plan!")

def render_dashboard_page():
    """Render analytics dashboard"""
    st.header("📊 Travel Analytics Dashboard")
    
    if not st.session_state.travel_requests:
        st.session_state.travel_requests = get_travel_requests()
    
    if st.session_state.travel_requests:
        df = pd.DataFrame(st.session_state.travel_requests)
        
        # Convert datetime columns
        df['created_at'] = pd.to_datetime(df['created_at'])
        if 'completed_at' in df.columns:
            df['completed_at'] = pd.to_datetime(df['completed_at'])
        
        # Status distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Request Status Distribution")
            status_counts = df['status'].value_counts()
            fig_pie = px.pie(values=status_counts.values, names=status_counts.index, 
                           title="Request Status Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("⚡ Processing Time Analysis")
            if 'processing_time' in df.columns and df['processing_time'].notna().any():
                fig_hist = px.histogram(df, x='processing_time', nbins=10,
                                      title="Processing Time Distribution")
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No processing time data available")
        
        # Quality scores over time
        if 'quality_score' in df.columns and df['quality_score'].notna().any():
            st.subheader("🎯 Quality Scores Over Time")
            fig_quality = px.line(df, x='created_at', y='quality_score',
                                title="Quality Scores Trend")
            st.plotly_chart(fig_quality, use_container_width=True)
        
        # Request volume over time
        st.subheader("📅 Request Volume Over Time")
        df['date'] = df['created_at'].dt.date
        daily_requests = df.groupby('date').size().reset_index(name='count')
        fig_volume = px.bar(daily_requests, x='date', y='count',
                          title="Daily Request Volume")
        st.plotly_chart(fig_volume, use_container_width=True)
        
    else:
        st.info("📊 No data available for analytics. Create some travel requests first!")

def render_admin_panel():
    """Render admin panel (admin only)"""
    if st.session_state.user_info.get('role') != 'admin':
        st.error("🚫 Access denied. Admin privileges required.")
        return
    
    st.header("⚙️ Admin Panel")
    
    # System status
    st.subheader("🖥️ System Status")
    system_status = get_system_status()
    
    if system_status:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("System Health", system_status.get('system_health', 'unknown'))
        
        with col2:
            st.metric("Uptime", f"{system_status.get('uptime', 0):.0f}s")
        
        with col3:
            st.metric("Total Requests", system_status.get('total_requests', 0))
        
        with col4:
            st.metric("Active Sessions", system_status.get('active_sessions', 0))
        
        # Performance metrics
        if 'performance_metrics' in system_status:
            st.subheader("📊 Performance Metrics")
            metrics = system_status['performance_metrics']
            
            col1, col2 = st.columns(2)
            with col1:
                st.json(metrics)
            
            with col2:
                # Create performance chart
                if metrics:
                    metric_names = list(metrics.keys())
                    metric_values = list(metrics.values())
                    
                    # Filter numeric values
                    numeric_metrics = [(k, v) for k, v in zip(metric_names, metric_values) if isinstance(v, (int, float))]
                    
                    if numeric_metrics:
                        names, values = zip(*numeric_metrics)
                        fig = go.Figure(data=[go.Bar(x=names, y=values)])
                        fig.update_layout(title="Performance Metrics")
                        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("⚠️ Unable to fetch system status")
    
    st.markdown("---")
    
    # Knowledge management
    st.subheader("📚 Knowledge Base Management")
    
    with st.form("add_knowledge_form"):
        title = st.text_input("Knowledge Title")
        category = st.selectbox("Category", ["general", "flights", "hotels", "documentation", "custom"])
        content = st.text_area("Content", height=200)
        
        if st.form_submit_button("Add Knowledge"):
            if title and content:
                try:
                    client = APIClient(API_BASE_URL, st.session_state.access_token)
                    response = client.post(f"/admin/knowledge?title={title}&content={content}&category={category}", {})
                    
                    if response.status_code == 200:
                        st.success("Knowledge added successfully!")
                    else:
                        st.error(f"Failed to add knowledge: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

def main():
    """Main application function"""
    load_css()
    
    # Check authentication
    if not st.session_state.access_token:
        render_header()
        render_login_page()
        return
    
    # Render main application
    render_header()
    selected_page = render_sidebar()
    
    # Route to selected page
    if selected_page == "New Request":
        render_new_request_page()
    elif selected_page == "My Requests":
        render_my_requests_page()
    elif selected_page == "Dashboard":
        render_dashboard_page()
    elif selected_page == "Admin Panel":
        render_admin_panel()
    else:
        render_new_request_page()  # Default

if __name__ == "__main__":
    main()
