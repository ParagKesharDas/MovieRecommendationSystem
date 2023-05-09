import streamlit as st
import subprocess
import csv
from streamlit import cache
# @st.cache(suppress_st_warning=True, allow_output_mutation=True)

def run_app(app_file, username, user_id=None):
    args = ["streamlit", "run", app_file, "--", "--username", username]
    if user_id:
        args += ["--user_id", user_id]
    subprocess.Popen(args)

def main():
    st.set_page_config(page_title="Login Page")

    st.title("Login Page")
    st.write('Welcome to the Movie Recommendation App! Please login or register to access its features.')
    # Create input fields for username and password
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    st.session_state["UserName"] = ""
    st.session_state["UserID"] = ""   
    
     
    # Create a login button
    if st.button("Login"):
        # Open the login.csv file
        with open('login.csv', encoding="cp437", errors='ignore') as login_file:
            csv_reader = csv.reader(login_file)
            found_match = False

            # Loop through the rows in the login.csv file
            for row in csv_reader:
                # Check if the username and password match
                if row[1] == username and row[2] == password:
                    st.success("Logged in successfully!")
                    user_id = row[0]
                    found_match = True
                    break

            if not found_match:
                st.error("Invalid username or password")
            else:
                st.session_state["UserID"] = user_id
                st.session_state["UserName"]=username
    
    if st.checkbox("Admin"):
        password1 = st.text_input("Admin Password", type="password")
        if st.button("Show Login Details"):
            if password1=="Admin@123":
                # Display the login.csv file as a table
                with open('login.csv', mode='r') as login_file:
                    csv_reader = csv.reader(login_file)
                    st.table(csv_reader)
                

if __name__ == '__main__':
    main()
