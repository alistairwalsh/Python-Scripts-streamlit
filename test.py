import streamlit as st
st.title("NMF")

st.image('nmf/1920px-Restricted_Boltzmann_machine.svg.png')

st.write('NMF as a probabilistic graphical model: visible units (V) are connected to hidden units (H) through weights W, so that V is generated from a probability distribution with mean ')

st.latex('{\displaystyle \sum _{a}W_{ia}h_{a}}\sum _{a}W_{ia}h_{a}')