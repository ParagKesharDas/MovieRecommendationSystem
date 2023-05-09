import streamlit as st

st.set_page_config(page_title="My App", page_icon=":guardsman:", layout="centered")

st.session_state["UserName"] = ""
st.session_state["UserID"] = ""   
st.header("Wellcome to Movie Recommendation Streamlit App !!")
st.write('''
🎥 Are you tired of scrolling endlessly through streaming platforms trying to find the perfect movie to watch?

🤔 Do you wish you had a personalized movie recommendation system that understands your preferences?

👀 Look no further! Our app offers a seamless user experience with personalized recommendations tailored to your viewing history and feedback.

🔎 With our app, you can easily search for your desired movie and get all the details you need, including the cast, plot, and reviews.

🌟 You can also rate movies on a scale of one to five and provide your personal feedback to help improve our recommendation algorithm.

📈 Once registered, you can receive personalized movie recommendations based on your activity within the app and keep track of your movie-watching history.

🤖 Our machine learning algorithms ensure that you get the best recommendations possible, making your movie-watching experience more enjoyable and effortless than ever before.

🎉 So why wait? Register now and join our community of movie enthusiasts!
''')
