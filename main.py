#This is Python script that runs the Streamlit app

from os import write
import streamlit as st
import numpy as np
import pandas as pd
import base64
from feature_select import FeatureSelector
import matplotlib.pyplot as plt 
from scipy import stats
import seaborn as sns

st.title("Feature Selector")
st.write("A simple interface for removing unnecessary features")
st.header("**1. Import**")
st.write("Let's start by adding some data.")

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="chosen_features.csv">Export Data as CSV</a>'
    return href 

@st.cache(suppress_st_warning=True,show_spinner=True)
def selector(df,target,algo,cat_or_cont):
    fs = FeatureSelector(df = df, target_col = target)
    if algo == "MRMR":
        rank = fs.mrmr()
    elif cat_or_cont == "Continuous":
        rank = fs.mutual_info_regress()
        rank = rank['Feature'].tolist()
    else:
        rank = fs.mutual_info_class()
        rank = rank['Feature'].tolist()
    # rank = rank.set_index('Feature')
    return rank

uploaded_file = st.file_uploader("Choose a file",type='csv',accept_multiple_files=False)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)
    data = data.dropna()
    data = data.select_dtypes(['number'])
    features_list = data.columns

    st.sidebar.header('Control Panel')

    target = st.sidebar.selectbox("Choose your target feature",features_list,\
        help = "Features will be selected with the ultimate goal of estimating this target feature.")

    cat_or_cont = st.sidebar.selectbox("Is the target categorical or continuous?",("Categorical", "Continuous"),index=1,\
        help = "This will help determine which type of algorithm to run - classification or regression.")

    max_num = int(len(features_list))
    num_features = st.sidebar.number_input("Number of selected features",min_value=1,max_value=max_num,value=10,\
        help="Consider the purpose of this feature selection. When in doubt, select a higher number of features.")

    must_haves = st.sidebar.multiselect("Choose which features to KEEP, if any", features_list,\
        help = "Force certain features to be included in the chosen features list.")

    remove = st.sidebar.multiselect("Choose which features to EXCLUDE, if any", features_list,\
        help = "Remove certain features from the chosen features list.")

    with st.sidebar.beta_expander('Advanced Options'):
        algo = st.selectbox("Choose feature selection algorithm",["MRMR","Mutual Info"],\
            help = "MRMR (default): good for redundant data, Mutual Info: good for non-linear data with few redundant variables")

    #Checks if target variable exists in keep or remove lists
    if (target in remove) or (target in must_haves):
        st.write("### Oops! The target variable cannot appear in the list of tags to be removed or kept. Please change the target or remove the target variable from the remove list or kept list.",)
        st.stop()

    #Checks if keep and remove lists overlap
    if len(set(must_haves) & set(remove)) > 0:
        st.write("### Oops! The keep and remove features overlap.")
        st.stop()
    
    #Filters dataframe by removing the remove features before running selector
    data = data.drop(columns=remove)

    # st.write("Here's a summary of your data")
    with st.beta_expander('Expand to see the raw data summary'):
        st.write('Number of starting features: **{}**'.format(len(features_list)-1))
        st.write('Number of data points: **{}**'.format(len(data)))
        st.write(data.describe())

    #Runs the selector function which is cached above
    chosen = selector(data,target,algo,cat_or_cont)

    #Reduces the full list to the top-N, set by num_features input
    # chosen_final = set(chosen.iloc[:num_features,:].index)

    chosen_final = set(chosen[:num_features])

    #Adds must haves
    if must_haves is not None:
        chosen_final = chosen_final | set(must_haves)
    
    st.header("**2. Results**")
    st.write('The dataset has been reduced from the **{}** original features to the following **{}** most relevant features for explaining **{}**.'.format(len(features_list)-1,len(chosen_final),target))

    #Sorts the list with the added must haves and reformats the dataframe for display
    # chosen_export = chosen.loc[list(chosen_final)].sort_values(by=['Score'],ascending=False).reset_index()

    chosen_export = [feature for feature in chosen if feature in chosen_final]
    chosen_export_df = pd.DataFrame(chosen_export,columns=['Feature'])
    st.dataframe(chosen_export_df)

    with st.beta_expander('Expand to see feature distributions'):
        # st.write("Here you can view each feature's histogram and a density plot against the target feature.")
        focus = st.selectbox("Choose which feature to compare against the target",chosen_export)
        x = data[focus]
        y = data[target]
        zx = np.abs(stats.zscore(x))
        zy = np.abs(stats.zscore(y))
        xind = np.where(zx < 3)[0]
        yind = np.where(zy < 3)[0]
        indices = list(set(xind) & set(yind) )
        x = x.iloc[indices]
        y = y.iloc[indices]
        plot_dict = {focus:x,target:y}
        plot_df = pd.DataFrame(plot_dict)
        fig, ax = plt.subplots(1,3,figsize = (12,4))
        ax[0].hist(x,bins="scott",color="#F63366")
        ax[0].set_xlabel(focus)
        ax[0].set_title("Selected Feature")
        sns.kdeplot(ax=ax[1],data=plot_df,x=focus, y=target,color="#F63366")
        ax[1].set_title("Selected vs Target")
        ax[2].hist(y,bins="scott",color="#F63366",orientation="horizontal")
        ax[2].set_xlabel(target)
        ax[2].set_title("Target Feature")
        # ax[1].scatter(x,y)
        fig.tight_layout()
        st.pyplot(fig)

    st.write("")

    st.header("**3. Export**")
    #Generate export link
    st.write("Now you can export your chosen features below.")
    st.markdown(get_table_download_link(chosen_export_df), unsafe_allow_html=True)

    # st.header("**3. Export**")
    # data_filtered = data[chosen_export_df['Feature']]
    # st.write("Now you can export the data below with only the selected features.")
    # st.markdown(get_table_download_link(data_filtered), unsafe_allow_html=True)




# chosen = data.iloc[1,:num_features]
# if st.sidebar.button("Run Feature Selector"):
#     st.write('Here are your results')
#     st.markdown(get_table_download_link(data), unsafe_allow_html=True)
#     chosen = choose_features(data,target)
    # st.bar_chart(chosen[:num_features,0],chosen[:num_features,1])

# fig, ax = plt.subplots()
# ax.plot(chosen)
# st.pyplot(fig)

# st.beta_container()
# st.beta_expander('Expander')
# with st.beta_expander('Expand'):
#     st.write('Juicy deets')
