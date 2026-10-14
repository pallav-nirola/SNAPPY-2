import os
import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv

# Load variables from the root .env file if present
load_dotenv()

# Try fetching from Streamlit secrets first, then fallback to environment variables (.env)
supabase_url = None
supabase_key = None

try:
    supabase_url = st.secrets.get("SUPABASE_URL")
    supabase_key = st.secrets.get("SUPABASE_KEY")
except Exception:
    pass

# Fallback to environment variables if st.secrets isn't configured or empty
if not supabase_url:
    supabase_url = os.getenv("SUPABASE_URL", "https://rgvuanqaoeqwpuaqxshb.supabase.co")
if not supabase_key:
    supabase_key = os.getenv("SUPABASE_KEY", "sb_publishable_G-NkeO3o1znDatX3WqCwuA_iBpLBweJ")

# Initialize Supabase client
supabase: Client = create_client(supabase_url, supabase_key)