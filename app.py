import streamlit as st

from src.screens.student_screen import student_screen
from src.screens.teacher_screen import teacher_screen
from src.screens.home_screen import home_screen

st.set_page_config(
    page_title="ClassLens",
    page_icon="https://i.ibb.co/YTYGn5qV/logo.png",
    # layout="wide",
    # initial_sidebar_state="expanded"
)

def main():
    
    if 'login_type' not in st.session_state:
        st.session_state['login_type'] = None
        
    match st.session_state['login_type']:
        case 'teacher':
            teacher_screen()
            
        case 'student':
            student_screen()
        
        case None:
            home_screen()
    
main()