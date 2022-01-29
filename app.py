import streamlit as st
from multiapp import MultiApp
from apps import allstocks, begin # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("allstocks", allstocks.app)
app.add_app("prediction", begin.app)
# app.add_app("Model", model.app)
# The main app
app.run()