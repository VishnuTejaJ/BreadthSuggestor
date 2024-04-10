import streamlit as st
import pandas as pd
import numpy as np
from priority_selector import get_priority
from gpa_order import get_sorted
from tfidf_1 import career_func

# Title
st.markdown("<h1 style='text-align: center;'>Breadth Suggestor</h1>", unsafe_allow_html=True)

# Tabs for different sections
active_tab = st.radio("Select a tab:", ["Interest", "Career", "GPA"], key="active_tab", horizontal=True)

# Initialize DataFrame outside the if statement
df = pd.DataFrame()

# Use session variable to track table visibility
if "table_visible" not in st.session_state:
    st.session_state.table_visible = False

# Clear table visibility state when the active tab changes
if "prev_tab" not in st.session_state:
    st.session_state.prev_tab = active_tab

if st.session_state.prev_tab != active_tab:
    st.session_state.prev_tab = active_tab
    st.session_state.table_visible = False

# Determine the active tab and get user choice accordingly
if active_tab == "Interest":
    st.header("Get your breadth based on your academic interest")
    interestchoice = st.selectbox('Input your Academic interests (Branch name)', ['','AE','AG','CS','EC', 'EE', 'MA', 'IE', 'IM','HS',])
    carrerchoice = None
    gpachoice = False
elif active_tab == "Career":
    st.header("Get your breadth based on your career interest")
    carrerchoice = st.selectbox('Input your career interest (Job role)', ['', 'Mechanical Engineering','Engineering Design Process','Environmental Engineering','Physics and Engineering','Finance'])
    interestchoice = None
    gpachoice = False
else:
    st.header("Get your breadth based on Grading of the course")
    gpachoice = st.checkbox("Input your GPA")
    interestchoice = None
    carrerchoice = None

# Validation checks and button to submit choices
if not (interestchoice or carrerchoice or gpachoice):
    st.warning("Please fill out all fields.")
else:
    if st.button("Submit Choices", type="primary"):
        # Print the choices
        # print(semester, choice)
        st.session_state.table_visible = True

# Display the table if the button is clicked and active tab is not changed
if st.session_state.table_visible:
    # Table with pagination
    st.write("### Course Suggestions")
    if interestchoice:
        df = get_priority(interestchoice)
    
    if carrerchoice:
        df = career_func(carrerchoice) 

    if gpachoice:
        df = get_sorted()
    # Display table without the extra column
    if not df.empty:
        page_size = st.slider("Items per page:", min_value=1, max_value=len(df), value=5)
        page_number = st.number_input("Page number:", min_value=1, max_value=len(df) // page_size + 1, value=1)

        start_index = (page_number - 1) * page_size
        end_index = start_index + page_size

        st.table(df[start_index:end_index].reset_index(drop=True))  # Reset index and remove the index column from display
    else:
        st.write("No courses found for the selected department.")
