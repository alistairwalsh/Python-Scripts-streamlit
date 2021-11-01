import pandas as pd
import streamlit as st
import numpy as np
import pyarrow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from streamlit.proto.RootContainer_pb2 import SIDEBAR

#st.set_page_config(layout="wide")

########
# Header
########
st.title('Non-negative Matrix Factorisation')
versions = f'streamlit {st.__version__},pyarrow {pyarrow.__version__},numpy {np.__version__}'
st.caption(versions)

#######
# Load
#######
@st.cache
def load_data():
    # Cache the conversion to prevent computation on every rerun
    return pd.read_pickle('2018_04_04_START_data.p')


big_dict = load_data()

st.sidebar.header('Options')

######
#select timepoint
######

timepoint_options = {k.split('local_docker_MySQL/data/START ')[-1].split('(')[0].strip():k for k in big_dict.keys()}
tp_options = pd.Series(timepoint_options.keys())
tp_selected = st.sidebar.radio('Select timepoint',tp_options, index = 4)
t = timepoint_options[tp_selected]


#######
# select participants
#######

mask = big_dict['local_docker_MySQL/data/START day 90 (20170203).xlsx']['MRI']['conducted'] == 1
participant_index = {
    'mri': big_dict['local_docker_MySQL/data/START day 90 (20170203).xlsx']['MRI'][mask].index,
    'no_mri': big_dict['local_docker_MySQL/data/START day 90 (20170203).xlsx']['MRI'][~mask].index,
    'all': big_dict['local_docker_MySQL/data/START day 90 (20170203).xlsx']['MRI'].index
    }


use_participants = st.sidebar.radio('Choose participants', ['mri', 'all', 'no_mri'])

st.write('timepoint: ', tp_selected)
st.write('participants:', use_participants)

#######
# select tests
#######

test_selected = st.sidebar.multiselect('select test',[k for k in big_dict[t]],'identity')

if not test_selected:
    test_selected = ['identity']
if 'identity' not in test_selected:
    test_selected.insert(0, 'identity')

d = {test:big_dict[t][test] for test in test_selected}
df = pd.concat(d.values(), axis=1, keys=d.keys())
data = df.loc[participant_index[use_participants],:]

st.header('Raw data')
st.dataframe(data)

#######
# Download option
#######

@st.cache
def convert_df(data):
  # Cache the conversion to prevent computation on every rerun
      return data.to_csv().encode('utf-8')

csv = convert_df(data)

st.download_button(
    label="Press to Download",
    data=csv,
    file_name='start_df.csv',
    mime='text/csv'
)
#------------------------------------------------------------------------------------------------
#######
# drop object columns in tests
#######

drop_cols = []

check_df = data.dtypes
check_df = pd.DataFrame({'data_type':check_df})

st.header('Data cleaning options')

st.markdown('### Column types')

#st, _ = st.columns((1,1))

st.write('''"object" data types are non-numeric data. 

The exception is identity id_code which is a unique identifier that should always be selected''')

st.write(check_df)
object_match = check_df[check_df['data_type'] == 'object'].index

st.header('Non-numeric columns')
st.write('''Non-numeric columns are often comments
 or free text that are difficult to analyse statistically. 
 
 To perform statistical analysis or graphing
 the data must be numeric. Select the non-numeric columns to remove from the dataset
 ''')

object_data = data.loc[:,[i for i in object_match if i not in [('identity','id_code')]]]

st.markdown('### Object columns to drop')
st.write('Below is a selection of the identified non-numeric columns, expand this dataframe to view the "object" datatype columns')


if not object_data.empty:
    st.dataframe(object_data)

#data.drop(columns = drop_cols, inplace=True)

selected_drops = st.sidebar.multiselect('drop object (test name, column name)', [c for c in object_data.columns])

#######
# Drop time columns
#######

st.write("""Datetime64[ns] columns are dates, often the response date when the test was conducted. 

These are not a part of the test and can be removed""")

timestamp_match = check_df[check_df['data_type'] == 'datetime64[ns]'].index
timestamp_data = data.loc[:,[i for i in timestamp_match if i not in [('identity','id_code')]]]

if not timestamp_data.empty:
    st.dataframe(timestamp_data)

selected_drops.extend(st.sidebar.multiselect('drop time (test name, column name)', [c for c in timestamp_data.columns]))


selected_drops.extend(st.sidebar.multiselect('drop other (test name, column name)', [c for c in data.columns if c not in selected_drops]))

data.drop(columns = selected_drops, inplace=True)
    


st.header("NMF Finally!")


X = data.iloc[:,1:]
st.write(X)
st.write('Impute here')

X.fillna(1)

component_err = {}

for c in range(1,len(X.columns)):
    model = NMF(n_components=c, init='random', max_iter=100000)
    W = model.fit_transform(X.iloc[:,:-1])
    H = model.components_
    Err = model.reconstruction_err_
    component_err[c] = Err
#pd.DataFrame({k:[v] for k,v in component_err.items()}).plot(kind = 'bar', title = 'Error')\
#.legend(bbox_to_anchor=[1.5, 0.5])

st.write('reduction in error per component')
st.bar_chart(pd.DataFrame({k:[v] for k, v in component_err.items()}).T)
num_comp = st.slider('Select number of components', min_value = 1, max_value = len(X.columns) - 1)

model = NMF(n_components=num_comp, init='random', max_iter=100000, random_state=43)
W = model.fit_transform(X)
H = model.components_
Err = model.reconstruction_err_

hm_cols = [c[0]+'_'+c[1] for c in X.columns]
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

number_clusters = st.number_input('Number of clusters', step = 1)

kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit(W)

st.header('KMeans clustering')
st.write(kmeans.labels_)

df_W['group'] = kmeans.labels_

st.header('Cluster distribution')

fig = sns.pairplot(df_W, hue = 'group')

st.pyplot(fig)

@st.cache
def convert_df(data):
    data['group'] = kmeans.labels_
  # Cache the conversion to prevent computation on every rerun
    return data.to_csv().encode('utf-8')

csv = convert_df(df_W)

st.download_button(
    label="Press to Download",
    data=csv,
    file_name='start_df.csv',
    mime='text/csv'
)