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

    st.dataframe(data)

    st.write(data.isna().sum())

    newdata = data.select_dtypes(include=np.number)

    st.write(data.equals(newdata))

    X = data.iloc[:,1:]

    var_table = X.var()
    st.header("Variability")
    st.write(var_table)
    st.header('Items with no variability')
    no_var_list = var_table[var_table == 0].index.tolist()
    st.write(no_var_list)

    X.drop(columns = no_var_list, inplace = True)

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

    #st.bar_chart(df_H)

    st.header('Weights')
    st.write(W)
    fig, ax = plt.subplots()
    df_W = pd.DataFrame(W, columns = ['comp_' + str(i) for i in range(W.shape[1])])
    ax = sns.heatmap(df_W)
    st.pyplot(fig)

    from sklearn.cluster import KMeans

    st.header('Weight on component')

    st.write(W)

    number_clusters = 3

    number_clusters = st.number_input('Number of clusters', step = 1, min_value=2)

    kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit(W)

    st.header('KMeans clustering')
    st.write(kmeans.labels_)

    df_W['group'] = kmeans.labels_

    st.header('Cluster distribution')

    fig = sns.pairplot(df_W, hue = 'group')

    st.pyplot(fig)