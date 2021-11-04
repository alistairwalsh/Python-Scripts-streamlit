import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF

st.title("NMF")

st.image('nmf/1920px-Restricted_Boltzmann_machine.svg.png')

st.write('NMF as a probabilistic graphical model: visible units (V) are connected to hidden units (H) through weights W, so that V is generated from a probability distribution with mean ')

st.latex('{\displaystyle \sum _{a}W_{ia}h_{a}}\sum _{a}W_{ia}h_{a}')


#######
# Load
#######

uploaded_file = st.file_uploader("Upload Files",type=['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file , sep = ';')
    data.dropna(axis = 'columns', how = 'all', inplace = True)

    st.dataframe(data)

    st.write(data.isna().sum())

    newdata = data.select_dtypes(include=np.number)

    st.write(data.equals(newdata))

    st.table(data.var())

    X = data.iloc[:,1:]

    component_err = {}

    for c in range(1,len(X.columns)):
        model = NMF(n_components=c, init='random', max_iter=100000)
        W = model.fit_transform(X)
        H = model.components_
        Err = model.reconstruction_err_
        component_err[c] = Err

    st.write('reduction in error per component')
    st.bar_chart(pd.DataFrame({k:[v] for k, v in component_err.items()}).T)
    num_comp = st.slider('Select number of components', min_value = 1, max_value = len(X.columns) - 1)

    model = NMF(n_components=num_comp, init='random', max_iter=100000, random_state=43)
    W = model.fit_transform(X)
    H = model.components_
    Err = model.reconstruction_err_

    hm_cols =  X.columns
    st.write(pd.DataFrame(H, columns = hm_cols))
    st.header('Components')
    fig, ax = plt.subplots()
    df_H = pd.DataFrame(data = H, columns = hm_cols, index = ['comp_' + str(i) for i in range(H.shape[0])])
    ax = sns.heatmap(df_H)
    plt.yticks(rotation=0)
    st.pyplot(fig)