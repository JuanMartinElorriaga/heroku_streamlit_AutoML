# AUTOML graphic interface for GM clients who wnat to find value in Analytics
from datetime import date
import janitor
#from datacleaner import autoclean
import streamlit as st
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import plotly.express as px
import matplotlib.pyplot as plt
#plt.style.use(['dark_background'])
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing   import StandardScaler
#from sklearn.metrics         import confusion_matrix
# TPOT
from tpot import TPOTClassifier
from tpot import TPOTRegressor

# Insert GM logo
# To center the image, I create 3 cols and leave 1 and 3 empty
col1, col2, col3 = st.beta_columns([2,6,1])
with col1:
    st.write("")
with col2:
    #path_image = "/Users/juanmartinelorriaga/Documents/repositorios/Marketing_GM/CRO/reports/figures/gm_icon.jpg"
    #st.image(path_image, width=250) # image from local
    try:
        st.markdown("<img src='https://raw.githubusercontent.com/JuanMartinElorriaga/datascience/master/gm_icon.jpg' style='width:300px;height:300px;'>", unsafe_allow_html=True) # image from URL
        #st.markdown("<img src='https://statics.turecibo.com/media/custom/login/gm2dev_logo.jpg' style='width:300px;height:300px;'>", unsafe_allow_html=True) # image from URL
        #st.markdown("<img src='https://images.app.goo.gl/L8SjbghgTi8jibQG6' style='width:300px;height:300px;'>", unsafe_allow_html=True) # image from URL
    except Exception:
        pass
with col3:
    st.write("")

df = None
# load data function
@st.cache
def load_data(file, sep=",", header='infer'): 
    df = pd.read_csv(file, low_memory=False, sep=sep, header=header).clean_names().remove_empty() # janitor
    #df = autoclean(pd.read_csv(file, low_memory=False, sep=sep)) # autoclean 

    # parse "dia" to datetime and then to ordinal to train
    #df["indice_de_dia"] = pd.to_datetime(df["indice_de_dia"])
    #df["indice_de_dia"] = df["indice_de_dia"].map(dt.datetime.toordinal)
    #df.set_index('indice_de_dia', inplace=True)
    num_cols  = df.select_dtypes(["float", "int"])
    text_cols = df.select_dtypes(["object"])
    return df.sort_index(), num_cols, text_cols


################################################################################
# UI header
# background color
#st.markdown(f"<span style=“background-color:#121922”>", unsafe_allow_html=True)

# add title
#stt.set_theme({'primary': '#FF00E7'})
#st.title("GM Dashboard for Data Analysis and ML")
# add sidebar title

check_sample_box = st.sidebar.checkbox("Use sample dataset!") 

#check_box_sample = st.sidebar.checkbox(label="Sample dataset")
st.markdown(
    f'<h2 style="color: #FB5906; font-size: big">{"Welcome to the GM Analytics board!"}</h2>',
    unsafe_allow_html=True)
#st.markdown(
#        f'<h1 style="color: #F68054; font-size: big">{"Welcome to the GM Dashboard!"}</h1>',
#        unsafe_allow_html=True)
st.markdown(
        f'<h4 style="color: #FB5906; font-size: big">{"Please, follow the steps to gain some info from your input data. The sidebar will help you to proceed"}</h4>',
        unsafe_allow_html=True)
#st.markdown("#### Welcome to the GM Dashboard! Please, follow the steps to gain some info from your input data.")
st.write(" ")
st.markdown("##### * Note: if you want to see the functionalities with a sample, click on the \"Use sample dataset!\" checkbox *")
################################################################################

# add widget for file input
uploaded_file = st.file_uploader("Upload a .csv, .tsv or .txt file to analyze", type=["csv", "tsv", "txt"])
header_box = st.checkbox("Check if your file has a header") 
# sample loading and pre-processing
if check_sample_box and not uploaded_file:
    # When using sample dataset, disable drag and drop
    #uploaded_file = st.empty()
    #header_box = st.empty()
    st.markdown("#### Good choice! This is a sample from a random e-commerce. Use this set to see the available functionalities, including exploratory analysis and ML predictions. Use the sidebar to follow through.")
    st.write(" ")
    st.write(" ")
    st.markdown("- Check **\"Display data table\"** to see the table given. If want to focus on a few features, use the **\"Which features do you want to see?\"** option.")
    st.markdown("- After that, you have the option to do an extensive profiling (check **\"Display exploratory analysis\"**. It may take a while...). ")
    st.markdown("- Also, the **\"Line Plot Settings\"** will allow you to display the features you want in a graph. It supports multiple features simultaneously, if you wish to compare.")
    st.markdown("- Finally, the Prediction settings will show you some predictive potential of the dataset chosen. \n First, choose the feature that you want to predict. Then, choose the list of predictors to predict your option. The results will be displayed for you.")
    # Load the sample dataset
    df, num_cols, text_cols = load_data(file=r'https://github.com/JuanMartinElorriaga/datascience/blob/master/Ecommerce.csv?raw=true', sep=",")
# file loading and pre-processing
elif uploaded_file:
    # Disable button if dataset is given
    #check_sample_box = False   
    if str(uploaded_file.type).endswith("csv"):
        if header_box:
            df, num_cols, text_cols = load_data(uploaded_file, sep=",", header='infer')
        else:
            df, num_cols, text_cols = load_data(uploaded_file, sep=",", header=None)
    elif str(uploaded_file.type).endswith("txt"):
        if header_box:
            df, num_cols, text_cols = load_data(uploaded_file, sep=" ", header='infer')
        else:
            df, num_cols, text_cols = load_data(uploaded_file, sep=" ", header=None)
    elif str(uploaded_file.type).endswith("tsv"):
        if header_box:
            df, num_cols, text_cols = load_data(uploaded_file, sep="\t", header='infer')
        else:
            df, num_cols, text_cols = load_data(uploaded_file, sep="\t", header=None)

# if there is a df, run the rest of code
if df is not None:
    # add checkbox
    #st.sidebar.title("Table settings")
    st.sidebar.markdown(
        f'<h2 style="color: #F68054; font-size: big">{"Table Settings"}</h2>',
        unsafe_allow_html=True)    
    check_box_df = st.sidebar.checkbox(label="Display data table")
    # Selection of features to show in table
    feature_selection = st.sidebar.multiselect(label="Which features do you want to see? (Default: all)", 
                                           options=df.columns.tolist())
    if check_box_df:
        st.write(" ")
        st.write(" ")
        check_box_stats = st.sidebar.checkbox(label="Display exploratory analysis")


    # plot settings
    #st.sidebar.title("Line plot settings")
    st.sidebar.markdown(
        f'<h2 style="color: #F68054; font-size: big">{"Line Plot Settings"}</h2>',
        unsafe_allow_html=True)
    st.sidebar.text('Note: we recommend to uncheck "Display exploratory analysis" before plotting')
    check_box_plot = st.sidebar.checkbox(label="Display line plots")
    # filter features for plotting
    features_plot_selection = st.sidebar.multiselect(label="Features to plot in timeline", 
                                                    options=num_cols.columns.tolist())
    
    #features_heat_selection = st.sidebar.multiselect(label="Features to plot in heatmap", 
    #                                                options=num_cols.columns.tolist())
    
    # Variable to predict and predictors
    #st.sidebar.title("Prediction settings")
    st.sidebar.markdown(
        '<h2 style="color: #F68054; font-size: big">Prediction Settings</h2>',
        unsafe_allow_html=True)
    # Initial settings to X and y
    X = None
    y = None    
    
    y = st.sidebar.selectbox(label="Feature to predict", 
                             options=["<select an option>"] + [c for c in num_cols.columns])
    if y != "<select an option>":
        y_type = df[y].dtype
    else:
        y_type = None
    # remove variable previously selected as "y" from predictors
    predictors = [c for c in num_cols.columns.tolist() if c != str(y)]
    X =  st.sidebar.multiselect(label="Features used as predictors", 
                                options=predictors)
    #print list of predictors
    #predictors_str = str(X)[1:-1].replace("\'", "")
    # detect type of model to run ML based on y type
    if y_type in ["float", "int"]:
        model_type = "regression"
    elif y_type in ["category"]: 
        model_type = "classification"
    else:
        model_type = None
    st.write(" ")
    if y_type and model_type and X:
        st.write(f"Your feature to predict is a type {y_type}, so we are setting a {model_type.upper()} MODEL. These are your variables selected as predictors: ")

    # Create and train the model
    # ML prediction
    if X and y != "<select an option>":
        X_train , X_test , y_train , y_test = train_test_split(df[X] , df[y] ,test_size = 0.25)
        st.write(X_train.head(3).reset_index(drop=True)) # df.head() to show predictors DF
        st.markdown("**Right now, the model is running some tests to figure out which algorithm best suits your dataset. You will see the results in the next minute or two ...**")
        # Differentiate between regression and classification
        if model_type == "regression":
            model = TPOTRegressor(generations= None, 
                      n_jobs=-1, 
                      max_time_mins=0.1, 
                      population_size=50, 
                      verbosity=2, 
                      random_state=42)
        elif model_type == "classification":
            model = TPOTClassifier(generations= None, 
                      n_jobs=-1, 
                      max_time_mins=0.1, 
                      population_size=50, 
                      verbosity=2, 
                      random_state=42)
        
        model.fit(X_train, y_train)

        # List of algorithms
        algorithms = [i[:i.find("(")] for i in list(model.evaluated_individuals_.keys())][:10]
        # cv scores
        cv_scores = [list(i.values())[-1].round(2) for i in list(model.evaluated_individuals_.values())][:10]
        # Merge and sort both lists
        alg_dict = {algorithms[i]: cv_scores[i] for i in range(len(algorithms))}
        sorted_alg_dict = {k: v for k, v in sorted(alg_dict.items(), reverse=True, key=lambda item: item[1])}
        # Turn dict into DataFrame
        df_algs = pd.DataFrame(list(sorted_alg_dict.items()), columns = ['algorithms','cv_score'])

        st.write(" ")
        st.markdown(f"**Best score:** {df_algs['cv_score'][0]}")
        st.markdown(f"**Best model:** {df_algs['algorithms'][0]}")
        #if model.score(X_test, y_test) > 0.70:
        #    st.write("**Insight:** This is really powerful! It means that your variable can be easily predicted using the others chosen as predictors.")
        #elif model.score(X_test, y_test) > 0.55:
        #    st.write("**Insight:** You are on to something there. Maybe you should try adding some extra predictors to the model.")
        #else:
        #    st.write("**Insight:** The coefficient shows a low value. There does not seem to be a clear linear correspondence between the variables chosen.")
            
            # Plot comparative model of algorithms
        plotly_tpot = px.bar(df_algs, 
                            x = "algorithms",
                            y = "cv_score",
                            title = "Best algorithms for prediction",
                            opacity=0.8,
                            template = "ggplot2")
        st.plotly_chart(plotly_tpot)
        if plotly_tpot:
            st.markdown(f'<h3 style="color: #FB5906; font-size: big">{"How to read this: "}</h3>',unsafe_allow_html=True)            
            st.write("- You are seeing a graph displaying the algorithms tested with their respective *score*. This has been made to evaluate how precise your prediction can potentially be. If the score is low, it means that the predictors chosen can predict the **\"Feature to predict\"**.")
            st.write("- This score shows the difference between the *real values versus the predicted ones*. The algorithm with the smallest bar is the winner, since it has the minimum predictive error.")

    # df with filters applied
    df_features = df[feature_selection]

    # Plots
    if features_plot_selection:
        plotly_line = px.line(df, 
                            x = df.index, 
                            y = features_plot_selection,
                            title = str(features_plot_selection)[1:-1].replace('\'','').replace(',',' +') + " timeline",
                            template = "ggplot2")

# show df according to selected settings
    if check_box_df:
        if feature_selection:
            st.dataframe(df_features)
            if check_box_stats:
                pr = ProfileReport(df_features, explorative=True)
                st_profile_report(pr)

        else:
            st.dataframe(df)
            if check_box_stats:
                pr = ProfileReport(df, explorative=True)
                st_profile_report(pr)

    if check_box_plot:
        if features_plot_selection:
            st.plotly_chart(plotly_line)

        
