# Graphic interface for GM clients who wnat a Data-Focused CRO
#from datetime import date
#import janitor
#from datacleaner import autoclean
#from google.cloud import bigquery
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#from sklearn.preprocessing   import StandardScaler
#from sklearn.metrics         import confusion_matrix

################################################################################

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

#df = None
df_path = r"https://raw.githubusercontent.com/JuanMartinElorriaga/datascience/master/GM2dev/df_mini_google_analytics.csv"

# load data function
@st.cache
def load_data(file, sep=",", header=0): 
    df = pd.read_csv(df_path, low_memory=False, sep=sep, header=header)#.clean_names().remove_empty() # janitor
    return df.sort_index()

# Load DataFrame
df = load_data(df_path, sep=",", header=0)
################################################################################
# "Begin" button
check_box_begin = st.sidebar.checkbox("Begin!") 

if not check_box_begin:
    st.markdown(
        f'<h2 style="color: #FB5906; font-size: big">{"Welcome to the GM Analytics board!"}</h2>',
        unsafe_allow_html=True)
#st.markdown(
#        f'<h1 style="color: #F68054; font-size: big">{"Welcome to the GM Dashboard!"}</h1>',
#        unsafe_allow_html=True)
    st.markdown(
        f'<h4 style="color: #FB5906; font-size: big">{"We want to show you how we use Analytics to improve your Conversion Rate performance. The sidebar will help you to proceed..."}</h4>',
        unsafe_allow_html=True)
#st.markdown("#### Welcome to the GM Dashboard! Please, follow the steps to gain some info from your input data.")
    st.markdown("When you are ready, check the **\"Begin!\"** button!")
################################################################################

# add widget for file input
#uploaded_file = st.file_uploader("Upload a .csv, .tsv or .txt file to analyze", type=["csv", "tsv", "txt"], )
#st.write(str(uploaded_file))
# sample loading and pre-processing
if check_box_begin:
    check_box_step_one = st.sidebar.checkbox("Step 1: Descriptive Analytics")
    check_box_step_two = None
    check_box_step_three = None
    #check_box_step_four = None
    if not check_box_step_one:
        # Display useful information for the user
        st.markdown("Great! Let's begin then. Before digging into a dataset, we would like to show you our analytics roadmap _(you can skip this step and dive into Step 2 if you prefer)_ ")
        # add image: "types of Analytics"
        st.markdown("<img src='https://raw.githubusercontent.com/JuanMartinElorriaga/datascience/master/types_5.jpg' style='width:600px;height:400px;'>", unsafe_allow_html=True) # image from URL
        st.markdown("This is the roadmap we will try to conquer with your data. Here is a brief summary of each stage of the process:")
        # Descriptive Analytics
        st.markdown(
            f'<h3 style="color: #FB5906; font-size: big">{"Descriptive Analytics"}</h3>',
            unsafe_allow_html=True)
        st.markdown("- How do you improve something? Anything. Any process, any action. You start but understanding your baseline. This is _always_ the first step of the process. In order to extract value from your data, we first need to understand how it behaves. By doing this,we kill two birds with one stone: we gain some insights from historical data, and also prepare the ground for further stages.")
        st.markdown("- Descriptive analytics can help to identify the areas of strength and weakness in an organization. Using a range of historic data and benchmarking, decision-makers obtain a holistic view of performance and trends on which to base business strategy.")
        st.markdown("- In its simplest form, descriptive analytics answers the question: **\"What happened?\"**")
        # Predictive Analytics
        st.markdown(
            f'<h3 style="color: #FB5906; font-size: big">{"Predictive Analytics"}</h3>',
            unsafe_allow_html=True)
        st.markdown("- This is where we start to become proactive. Not always the past information is enough for a business to prosper; we need to anticipate outcomes and behaviors based upon data, and not on a hunch or assumptions.")
        st.markdown("- We apply advanced **AI** methods, such as _Machine Learning_, _Deep Learning_ and _Data Mining_ along with _advanced statistics_ to bring together the management, information technology, and modeling business process to make predictions about the future.")

        st.markdown("- In its simplest form, descriptive analytics answers the question: **\"What will happen?\"**")
        st.markdown(
            f'<h3 style="color: #FB5906; font-size: big">{"Prescriptive Analytics"}</h3>',
            unsafe_allow_html=True)
        st.markdown("- We finally reach the mountain peak. By factoring information about possible situations or scenarios, available resources, past performance and current performance, we suggests a course of action or strategy.")
        st.markdown("- We rely on everything we colected from previous steps, but go even further: using the predictive analytics estimation of what is likely to happen, we can recommend what future course to take.")
        st.markdown("- It can be used to make decisions on _any time horizon_, from immediate to long term. ")
        st.markdown("- In its simplest form, descriptive analytics answers the question: **\"What to do next?\"**")
        st.write(" ")    
        st.write(" ")    
        st.markdown(
            f'<h3 style="color: #FB5906; font-size: big">{"Data-focused CRO"}</h3>',
            unsafe_allow_html=True)
        st.markdown("- Although many marketing organizations talk about the importance of mining big data for driving both tactical and strategic decisions, most have not been able to integrate the technique into their daily operations. This is where Analytics and Data Science come into play.")
        st.markdown("- Unshackled from the constraints of pre-packaged analytics reports, we can take raw data about website traffic, combine it with customer purchase data, geographic and demographic information, and even PPC and email tracking/click data, to create a profile of customers most likely to respond to certain marketing messages and promotions, at what time of day, in which season, and for what products or services. These insights can literally open up a **new world of optimization opportunities** for a robust conversion optimization.")
        st.write(" ")
        st.write(" ")
        st.markdown(
            f'<h4 style="color: #FB5906; font-size: big">{"Now we know what to look for. Let´s continue with Step 2, by loading a sample dataset to test a common pipeline. "}</h4>',
            unsafe_allow_html=True)

    # "Descriptive analytics" button
    if check_box_step_one:
        check_box_step_two = st.sidebar.checkbox("Step 2: Predictive Analytics")
        #df = load_data(df_path, sep=",", header=0) #FIXME: only works in local
        if not check_box_step_two:
            st.markdown(
                f'<h2 style="color: #FB5906; font-size: big">{"Descriptive Analytics"}</h2>',
                unsafe_allow_html=True)
            st.markdown("If we want to gain value from a **Data-focused CRO**, we first need a dataset! But don´t worry, we got you covered. We will load a sample dataset to show the kind of data we need, and also how the process goes.")
            st.markdown("This is how the data looks like: ")
            # Display head
            st.dataframe(df.head(5))
            # Description of each feature
            st.markdown("This is a classic **Google Analytics dataset**. Below is a brief description of each column: ")
            st.markdown("- _fullVisitorId_: A unique identifier for each user of the Google Merchandise Store.")
            st.markdown("- _channelGrouping_: The channel via which the user came to the Store.")
            st.markdown("- _date_: The date on which the user visited the Store.")
            st.markdown("- _device_: The specifications for the device used to access the Store.")
            st.markdown("- _geoNetwork_: This section contains information about the geography of the user.")
            st.markdown("- _socialEngagementType_: Engagement type, either \"Socially Engaged\" or \"Not Socially Engaged\".")
            st.markdown("- _totals_: This section contains aggregate values across the session.")
            st.markdown("- _trafficSource_: This section contains information about the Traffic Source from which the session originated.")
            st.markdown("- _visitId_: An identifier for this session. This is part of the value usually stored as the _utmb cookie. This is only unique to the user. For a completely unique ID, you should use a combination of fullVisitorId and visitId.")
            st.markdown("- _visitNumber_: The session number for this user. If this is the first session, then this is set to 1.")
            st.markdown("- _visitStartTime_: The timestamp expressedasPOSIXtime")
            st.markdown("- _hits_: This row and nested fields are populated for any and all types of hits. Provides a record of all page visits.")
            st.markdown("- _customDimensions_: This section contains any user-level or session-level custom dimensions that are set for a session. This is a repeated field and has an entry for each dimension that is set.")
            st.write(" ")
            # Pandas profiling
            st.markdown(
                f'<h3 style="color: #FB5906; font-size: big">{"Profiling report"}</h3>',
                unsafe_allow_html=True)
            st.markdown("First things first. We take the whole dataset and make an extensive **profiling report**. No matter the subject involved, this report is _priceless_: it returns the **\"big picture\"** of what we are available to work with, such as:  ")
            st.markdown("- **Essentials**: type, unique values, missing values, min, max, quantiles, etc.")
            st.markdown("- **Descriptive statistics**: mean, mode, standard deviation, sum, median absolute deviation, coefficient of variation, kurtosis, skewness")
            st.markdown("- **Histograms**, **text analysis** and **image analysis**.")
            st.markdown("- **Correlations** between variables.")
            st.markdown("- **File and Image analysis**")
            st.markdown("- **Text analysis**")
            st.write(" ")
            st.markdown("After this process, we have made the first step to turn the raw data into information, displaying historical patterns of behaviors and performance. We can continue to \"Step 2: Predictive Analytics\".")
            # Display PANDAS PROFILING
            st.markdown(
                f'<h3 style="color: #FB5906; font-size: big">{"Watch the profiling report for yourself!"}</h3>',
                unsafe_allow_html=True)
            # link to Github
            link = '[GO TO PROFILING REPORT!](https://htmlpreview.github.io/?https://github.com/JuanMartinElorriaga/datascience/blob/master/GM2dev/prof_report.html)'
            st.markdown(link, unsafe_allow_html=True)

            #HtmlFile = open("/Users/juanmartinelorriaga/Documents/repositorios/Marketing_GM/CRO/data/processed/prof_report.html", 'r', encoding='utf-8')
            #source_code = HtmlFile.read()
            #print(source_code)
            #components.html(source_code)

            #prof = ProfileReport(df, explorative=True, orange_mode=True)
            #st_profile_report(prof)

    if check_box_step_two:
        check_box_step_three = st.sidebar.checkbox("Step 3: Prescriptive Analytics")
        if not check_box_step_three:
            st.markdown(
                    f'<h2 style="color: #FB5906; font-size: big">{"Predictive Models"}</h2>',
                    unsafe_allow_html=True)
            st.markdown("Alright. Time to get proactive.")
            st.markdown("Optimisation of _e-commerce platforms_ requires effective modelling of interactions among the retailers, customers and the platforms themselves. One of the primary benefits for doing business online is that several aspects of customers behavior can be tracked with assistance of modern technology.")
            st.markdown("There is a rich list of predictive models in the area of e-commerce. Here, we will present you those that we consider as most relevant to the business: ")
            st.markdown("- _Conversion prediction_")
            st.markdown("- _Churn model_")
            st.markdown("- _System recommendation_")
            st.markdown("- _Inventory management_")
            st.markdown("- _Customer lifetime value_")
            st.markdown("- _Sentiment analysis_")
            st.markdown("- _Fraud detection_")
            st.markdown("- _Pricing model_")
            st.write(" ")
            st.markdown(
                    f'<h3 style="color: #FB5906; font-size: big">{"Conversion Prediction Model"}</h3>',
                    unsafe_allow_html=True)
            st.markdown("For this time, we will show you the process of a **Conversion Prediction Model**. Thousands of customers begin trials of our product each day. Some customers end up purchasing a subscription, while others do not. Given what we know about customers' behavior on the site, their exposure to our marketing efforts, prior history of trialing, interactions with customer support, and other factors, can we predict _which customers will subscribe_ within a certain time window of starting a trial? This model is made to answer that question.")
            st.write(" ")
            st.markdown("- We start with a raw Google Analytics dataset, containing historical data about past users, such as the device they used, browser, country, region, etc.")
            st.markdown("- This dataset contains a special column, previously calculated, that shows the total transaction revenue for each purchase.")
            st.markdown("- Since this revenue is of vital importance to every e-commerce local, we will tell the AI algorithm to learn how to predict it.")
            st.markdown("- As a result, we are expecting a list of users, with an individual predictive revenue for each one.")
            st.markdown("- _Note:_ Since this is just a demo to show you the process, we will skip the pre-process steps and go straight to the final results.")
        
            st.markdown(
                    f'<h4 style="color: #FB5906; font-size: big">{"Type of Model"}</h4>',
                    unsafe_allow_html=True)
            st.markdown("The kind of model depends on the nature of the variable to predict. Therefore, since we are now looking for a numeric value (future revenue of each individual), this is a **Regression model**.")
            st.markdown(
                    f'<h4 style="color: #FB5906; font-size: big">{"Variable to predict"}</h4>',
                    unsafe_allow_html=True)
            st.markdown("- Revenue of each customer.")
            st.markdown(
                    f'<h4 style="color: #FB5906; font-size: big">{"Predictors"}</h4>',
                    unsafe_allow_html=True)
            st.markdown("- These are the features used to _\"feed\"_ the model, and learn patterns about the variable to predict.")
            st.markdown("- In this case, there are multiple predictors, which can be seen in Step 1: Descriptive Analytics. Among them: Channel Grouping, total hits, kind of browser, country, total transactions, total pageviews, etc.")
            st.markdown(
                    f'<h4 style="color: #FB5906; font-size: big">{"How to determine the precision of the model?"}</h4>',
                    unsafe_allow_html=True)
            st.markdown("- Basically, we take each point of prediction and compare it to the actual value, present in the dataset. The average of all those errors is then taken as the final score of the model, to determine how precise the predictions actually are.")
            st.markdown(
                    f'<h4 style="color: #FB5906; font-size: big">{"Final Result"}</h4>',
                    unsafe_allow_html=True)
            st.dataframe({
                'fullVisitorId': ['0000040862739425590', '0000039460501403861', '0000085059828173212', '000026722803385797', '0000436683523507380'],
                'Predicted Revenue': [103.15, 191.20, 25.80, 0, 0]
            })
            st.markdown("This is a sample of the final matrix. Just like we planned, the algorithm calculates the predicted value of Revenue for each particular user, depending on his (her) particular behavior.")
            st.markdown("- The algorithm takes almost _3/4 parts_ of the dataset to learn common pattern and behaviors from users, and then predicts the last _1/4 part_. It is important to clear out that this _1/4 part_ is new data for the algorithm! That is the point of this: to be capable to predict on fresh data!")
            st.markdown(
                    f'<h4 style="color: #FB5906; font-size: big">{"Take a look at Step 4: Prescriptive Analytics for a more complete description of outputs."}</h4>',
                    unsafe_allow_html=True)
            

    # DESCRIPTIVE ANALYSIS
        if check_box_step_three:
            st.markdown(
                    f'<h2 style="color: #FB5906; font-size: big">{"Customer Revenue Model: Prescriptive Actions"}</h2>',
                    unsafe_allow_html=True)
            st.markdown("- This last section is made to answer your most pragmatic questions, such as **\"What are the final results, and what should I do with all this new information?**\"")
            st.markdown("- The prescriptive section takes the results from both the predictions and the exploratory analysis and presents the knowledge gained from the historical data in a clear and easy to digest way.")
            st.markdown("- This _\"new knowledge\"_ is then turned into _qualitative recommendations_ after being studied by our Marketing experts. ")
            st.markdown(
                    f'<h3 style="color: #FB5906; font-size: big">{"Model Insights (examples)"}</h3>',
                    unsafe_allow_html=True)
            st.markdown(
                    f'<h4 style="color: #FB5906; font-size: big">{"The 80/20 Rule"}</h4>',
                    unsafe_allow_html=True)
            st.markdown("The _80/20 rule_ has proven true for many businesses–only a small percentage of customers produce most of the revenue. As such, marketing teams are challenged to make appropriate investments in promotional strategies.")
            st.markdown("- In this case, the result shows an even more distant proportion! ")
            st.markdown(
                    f'<h4 style="color: #FB5906; font-size: big">{"Grouping Channels"}</h4>',
                    unsafe_allow_html=True)
            st.markdown("- The TOP 5 Grouping Channels represent 97% of total values. Respectively:")
            df_channels = pd.DataFrame({'Channels': ['Organic Search', 'Social', 'Direct', 'Referral', 'Paid Search'],
                          'Percentage':  [42.99, 24.39, 15.42, 11.89, 2.55],
                          'Observation': ['But 3rd in terms of Revenue','','','But almost 40% in Visits and Revenue','']
                          })
            df_channels
            fig_channels = px.bar(df_channels, x='Channels', y='Percentage', color='Channels', color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_channels)

            st.markdown(
                    f'<h4 style="color: #FB5906; font-size: big">{"Browsers"}</h4>',
                    unsafe_allow_html=True)
            st.markdown("- Chrome has the highest frequency but the highest value of transactions was made on Firefox")
            df_browsers = pd.DataFrame({
                                        'Browser': ["Chrome", "Safari", "Firefox", "Internet Explorer", "Edge"],
                                        'Count': [43682, 12672, 2479, 1343, 703]
                                        })
            df_browsers
            fig_browsers = px.bar(df_browsers, x='Browser', y='Count', color='Browser', color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_browsers)

            st.markdown(
                    f'<h4 style="color: #FB5906; font-size: big">{"Operational Systems"}</h4>',
                    unsafe_allow_html=True)
            st.markdown("- The TOP 5 Operational Systems correspond to 96% of total values. Respectively:")    
            df_OS = pd.DataFrame({'OS': ['Windows', 'Macintosh', 'Android', 'iOS', 'Linux'],
                          'Percentage':  [38.75, 28.04, 14.15, 11.75, 3.91]
                          })
            df_OS
            fig_OS = px.bar(df_OS, x='OS', y='Percentage', color='OS', color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_OS)

            st.markdown(
                    f'<h4 style="color: #FB5906; font-size: big">{"Devices"}</h4>',
                    unsafe_allow_html=True)
            st.markdown("- A clear majority prefer to access through Desktop devices")
            df_devices = pd.DataFrame({'Device': ['Desktop', 'Mobile', 'Tablet'],
                          'Percentage':  [73.5,23.12, 3.38]
                          })
            df_devices
            fig_devices = px.bar(df_devices, x='Device', y='Percentage', color='Device', color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_devices)
            st.markdown(
                    f'<h4 style="color: #FB5906; font-size: big">{"Regions"}</h4>',
                    unsafe_allow_html=True)
            st.markdown("- The TOP 5 regions are equivalent to almost 70% of total set:")
            df_regions = pd.DataFrame({'Regions': ['Northern America', 'Southeast Asia', 'Northern Europe', 'Southern Asia', 'Western Europe'],
                          'Percentage':  [44.18, 08.29, 6.73, 6.33, 6.23]
                          })
            df_regions
            st.markdown("- Also, USA is the country with most frequent access to the local store")
            fig_regions = px.bar(df_regions, x='Regions', y='Percentage', color='Regions', color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_regions)

            st.markdown(
                    f'<h4 style="color: #FB5906; font-size: big">{"Most frequent cities"}</h4>',
                    unsafe_allow_html=True)
            st.write(" ")
            df_cities = pd.DataFrame({
                                    'City': ["Mountain View", "New York", "San Francisco", "Sunnyvale", "London"],
                                    'Count': [2867, 1910, 1432 , 895, 879],
                                    'Observation': ['has 19% of visits but just 16% of revenues', 'responsible for 14% of visits and 31% of revenues', '3.5% of visits but has a high significance in revenues', '', '']
                                     })
            df_cities
            fig_cities = px.bar(df_cities, x='City', y='Count', color='City', color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_cities)

            st.markdown(
                    f'<h4 style="color: #FB5906; font-size: big">{"Dates"}</h4>',
                    unsafe_allow_html=True)
            st.markdown("- The months with highest access are October and November.")
            st.markdown("- On the weekend the trafic is lower than other days.")
            st.markdown("- The 5 days with highest number of accesses is 1 and 5")
            st.markdown("- Considering the full count of dates, the days with highest accesses are almost all in november/2016")
            st.markdown("- From 17 to 20 hours have the highest numbers of visits")

            st.markdown(
                    f'<h3 style="color: #FB5906; font-size: big">{"Feature Importance"}</h3>',
                    unsafe_allow_html=True)
            df_cities = pd.DataFrame({
                                    'City': ["Mountain View", "New York", "San Francisco", "Sunnyvale", "London"],
                                    'Count': [2867, 1910, 1432 , 895, 879],
                                    'Observation': ['has 19% of visits but just 16% of revenues', 'responsible for 14% of visits and 31% of revenues', '3.5% of visits but has a high significance in revenues', '', '']
                                     })
            df_feature_imp = pd.DataFrame({
                                    'Features': ["Pageviews ", "Hits ", "Visit Start Time ", "Visit Number ", "Network Domain ", "City ", "Metro ", "Region ", "Operating System ", "Country ", "Source ", "referral Path ", "Channel Grouping ", "Device Category ", "Keyboard ", "adWords Click Info ", "Source medium ", "Direct ", "New Visits ", "Continent ", "Browser ", "Bounces ", "Sub Continent "],
                                    'Importance': [1106, 1105, 1058, 585, 509, 426, 321, 305, 288, 248, 222, 178, 137, 118, 103, 101, 96, 91, 89, 72, 71, 70, 40]
            })
            fig_features = px.bar(df_feature_imp, x='Importance', y='Features', color='Features', width=800, height=700, color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_features)
            #st.markdown("<img src='https://raw.githubusercontent.com/JuanMartinElorriaga/datascience/master/Screen Shot 2021-05-11 at 14.07.05.png' style='width:700px;height:700px;'>", unsafe_allow_html=True) # image from URL
            st.markdown("- As it shows in the image, the algorithm states that **total pageviews, total hits and visit start time are the most crucial factors that determine the future value of revenue**.")
    
