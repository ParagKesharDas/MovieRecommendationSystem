import streamlit as st
import csv

def get_last_user_id_rating():
    with open('ratings.csv', mode='r') as rating_file:
        csv_reader = csv.reader(rating_file)
        last_row = list(csv_reader)[-1]
        last_user_id = int(last_row[0])
        return last_user_id

def get_last_user_id_login():
    with open('login.csv', mode='r') as rating_file:
        csv_reader = csv.reader(rating_file)
        last_row = list(csv_reader)[-1]
        last_user_id = int(last_row[0])
        return last_user_id

def get_last_user_id():
    lur=get_last_user_id_rating()
    lul=get_last_user_id_login()
    if lul<lur:
        return lur
    else:
        return lul
    
def isPresent(username):
    with open('login.csv', encoding="cp437", errors='ignore') as login_file:
            csv_reader = csv.reader(login_file)
            found_match = False

            # Loop through the rows in the login.csv file
            for row in csv_reader:
                # Check if the username and password match
                if row[1] == username:
                    found_match = True
                    return found_match

            if not found_match:
                return found_match   
def main():
    st.set_page_config(page_title="Registration Page")
    st.session_state["UserName"] = ""
    st.session_state["UserID"] = ""   
    
    # Create or open the login.csv file
    with open('login.csv', mode='a', newline='') as login_file:
        login_writer = csv.writer(login_file)

        st.title("Registration Page")

        # Create input fields for username and password
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        # Create a register button
        if st.button("Register"):
            # Check if username and password are not empty
            if username != "" and password != "":
                # Get the last user ID from the rating.csv file
                last_user_id = get_last_user_id()
                # Increment the last user ID to get a new user ID
                new_user_id = last_user_id + 1


                isThere=isPresent(username)

                if isThere :
                    st.error("Username "+username+" Already Exists")  
                else:    
                    # Write the username, password, and user ID to the login.csv file
                    login_writer.writerow([new_user_id,username,password])

                    st.success("Registered successfully!")
                    st.success("Now "+username+" can login")

            else:
                st.error("Please enter both username and password")  
# Display the login.csv file as a table
        # with open('login.csv', mode='r') as login_file:
        #     csv_reader = csv.reader(login_file)
        #     st.table(csv_reader)

if __name__ == '__main__':
    main()