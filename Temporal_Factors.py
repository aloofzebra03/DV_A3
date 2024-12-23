#!/usr/bin/env python
# coding: utf-8

# # ANALYSES BASED ON TEMPORAL FACTORS

# In[1]:


# Importing the Libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[2]:


# Loading the dataset
loyalty = pd.read_csv('Dataset/Updated_Customer_Loyalty_History.csv')


# In[3]:


# View of the dataset
loyalty


# In[4]:


# Converting enrollment date and cancellation date to pandas date time object.
loyalty['Enrollment Date'] = pd.to_datetime(loyalty['Enrollment Date'])
loyalty['Cancellation Date'] = pd.to_datetime(loyalty['Cancellation Date'])


# In[5]:


def applySeasons(month):
    """
    Determine the season based on the given month.
    Args:
        month (int): The month as an integer (1 for January, 2 for February, ..., 12 for December).
    Returns:
        str: The season corresponding to the given month. Possible values are 'Winter', 'Spring', 'Summer', and 'Fall'.
    """
    
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'


# In[6]:


# Creating the enrollments dataframe
# enrollments = pd.DataFrame({
#     "Enrollment_Month": loyalty['Enrollment Date'].dt.month,
#     "Enrollment_Year": loyalty['Enrollment Date'].dt.year,
#     "Enrollment_Quarter": loyalty['Enrollment Date'].dt.quarter,
#     "Enrollment_Season": loyalty['Enrollment Date'].dt.month.apply(applySeasons),
#     "Cancellation_Month": loyalty['Cancellation Date'].dt.month,
#     "Cancellation_Year": loyalty['Cancellation Date'].dt.year,
#     "Cancellation_Quarter": loyalty['Cancellation Date'].dt.quarter,
#     "Cancellation_Season": loyalty['Cancellation Date'].dt.month.apply(applySeasons)
# })     
enrollments = pd.DataFrame({
    "Enrollment_Month": loyalty['Enrollment Date'].dt.month,
    "Enrollment_Year": loyalty['Enrollment Date'].dt.year,
    "Enrollment_Quarter": loyalty['Enrollment Date'].dt.quarter,
    "Enrollment_Season": loyalty['Enrollment Date'].dt.month.apply(applySeasons),
})     


# In[7]:


enrollments


# ## First Run : Investigating year on year enrollments

# ### This iteration tries to investigate the relationship between the year of enrollments and the number of enrollments into the loyalty program. It tries to see if there is a periodic or cyclical trend in the data

# ### Data
# #### Used the `enrollments` dataframe

# ### Visualizations
# - #### Radial Area Chart : Shows values as distances from center. Higher the distance, higher the value.
# - #### LOESS Trend Curve : Plots a trend line.

# In[8]:


def plot_radial_area_chart(data: pd.DataFrame):
    """
    Plots a radial area chart for year-wise enrollments using Plotly.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data. 
                         It must have a column 'Enrollment_Year'.
    """
    try:
        # Aggregate enrollments by year
        enrollments = data.groupby('Enrollment_Year').size()

        # Sort years for consistent order
        years = sorted(data['Enrollment_Year'].unique())
        enrollments = enrollments.reindex(years, fill_value=0)

        # Convert years to strings for better radial plotting
        year_labels = [str(year) for year in years]

        # Create an interactive radial area chart
        fig = go.Figure()

        # Add radial area data
        fig.add_trace(go.Scatterpolar(
            r=enrollments.values,
            theta=year_labels,
            fill='toself',
            name='Enrollments',
            mode='lines+markers',
            marker=dict(color='blue'),
            line=dict(color='blue')
        ))

        # Layout adjustments
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    title="Number of Enrollments",
                    angle=0
                )
            ),
            title=dict(text="Radial Area Chart for Year-wise Enrollments"),
            legend=dict(title="Legend")
        )

        # Show the plot
        fig.show()

    except Exception as e:
        print(f"Error in plot_radial_area_chart:", e)


# In[9]:


plot_radial_area_chart(enrollments)


# In[10]:


import statsmodels.api as sm
def plot_loess_trend_all_years(data: pd.DataFrame):
    """
    Plots an interactive LOESS curve (trend line) for enrollments and cancellations across all years.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    """
    try:
        # Prepare data
        enrollments = data.groupby('Enrollment_Year').size()

        # Reindex for consistent order
        years = sorted(data['Enrollment_Year'].unique())
        enrollments = enrollments.reindex(years, fill_value=0)

        # Apply LOESS smoothing
        x_numeric = range(len(years))  # Numeric x-values for smoothing
        loess_enrollments = sm.nonparametric.lowess(enrollments.values, x_numeric, frac=0.5)

        # Create interactive Plotly chart
        fig = go.Figure()

        # Add original enrollments and LOESS curve
        fig.add_trace(go.Scatter(
            x=years, y=enrollments.values, mode='markers+lines',
            name='Enrollments (Original)', line=dict(dash='dash', color='skyblue')
        ))
        fig.add_trace(go.Scatter(
            x=years, y=loess_enrollments[:, 1], mode='lines',
            name='Enrollments (LOESS)', line=dict(color='blue')
        ))



        # Layout adjustments
        fig.update_layout(
            title="LOESS Trend Curve for All Years",
            xaxis=dict(title="Year", tickmode='linear', tickvals=years),
            yaxis=dict(title="Count"),
            legend=dict(title="Legend")
        )

        fig.show()

    except Exception as e:
        print(f"Error in plot_loess_trend_all_years:", e)


# In[11]:


plot_loess_trend_all_years(enrollments)


# ### Conclusions
# - #### From the plots, we see that the enrollments are the lowest in 2012 possibly because the program started in 2012.
# - #### The enrollments are nearly steady from the years 2013 - 2015. It then slightly increases in the years 2016 and 2017.
# - #### But there is a steep increase in the number of enrollments in 2018. This could possibly be because of the "2018 Promotion" run by the airline to enroll more people into the loyalty program.
# 

# # Second Run : Plotting Per Year Per Month Enrollment Data

# ### Data
# #### Used the `enrollments` dataframe. Then did a groupby on `Enrollment_Year` and `Enrollment_Month` to get the monthly counts for every year.

# ### Visualizations
# - #### A interactive line plot plotted in plotly
# - #### A interactive stacked area chart using raw enrollments plotted using plotly. 
# - #### A stacked area chart using normalized enrollment data.

# In[12]:


def plot_monthly_enrollments_by_year_plotly(data: pd.DataFrame):
    """
    Plots the monthly enrollments for individual years using Plotly.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the time series data.
    """
    try:
        # Group data by year and month and count enrollments
        monthly_counts = data.groupby(['Enrollment_Year', 'Enrollment_Month']).size().reset_index(name='Enrollments')

        # Create Plotly line plot
        fig = px.line(
            monthly_counts,
            x='Enrollment_Month',
            y='Enrollments',
            color='Enrollment_Year',
            markers=True,
            title="Monthly Enrollments by Year",
            labels={
                'Month': 'Month',
                'Enrollments': 'Enrollments',
                'Year': 'Year'
            }
        )
        fig.update_xaxes(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        )
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",  # Horizontal legend
                x=0.5,  # Center horizontally
                xanchor="center",
                y=1.1,  # Place above the plot
                yanchor="bottom"
            )
        )
        fig.show()
    except Exception as e:
        print("Error in plot_monthly_enrollments_by_year_plotly:", e)


# In[13]:


plot_monthly_enrollments_by_year_plotly(enrollments)


# In[14]:


def plot_seasonality_area_chart(data: pd.DataFrame):
    """
    Plots an area chart to visualize normalized seasonality patterns across years.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the time series data.
    """
    try:
        # Normalize enrollments by year to highlight seasonal variations
        monthly_counts = data.groupby(['Enrollment_Year', 'Enrollment_Month']).size().reset_index(name='Enrollments')
        yearly_totals = monthly_counts.groupby('Enrollment_Year')['Enrollments'].transform('sum')
        monthly_counts['Normalized_Enrollments'] = monthly_counts['Enrollments'] / yearly_totals

        # Map months to their names for the plot
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_counts['Month_Name'] = monthly_counts['Enrollment_Month'].map(lambda x: month_names[x - 1])

        # Create an area chart using Plotly
        fig = px.area(
            monthly_counts,
            x='Month_Name',
            y='Enrollments',
            color='Enrollment_Year',
            title='Seasonal Variation in Enrollments',
            labels={
                'Month_Name': 'Month',
                'Enrollments': 'Enrollments',
                'Enrollment_Year': 'Year'
            }
        )
        fig.update_layout(xaxis=dict(categoryorder='array', categoryarray=month_names))
        fig.show()
    except Exception as e:
        print("Error in plot_normalized_seasonality_area_chart:")
        print(e)


# In[15]:


plot_seasonality_area_chart(enrollments)


# In[16]:


def plot_normalized_seasonality_area_chart(data: pd.DataFrame):
    """
    Plots an area chart to visualize normalized seasonality patterns across years.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the time series data.
    """
    try:
        # Normalize enrollments by year to highlight seasonal variations
        monthly_counts = data.groupby(['Enrollment_Year', 'Enrollment_Month']).size().reset_index(name='Enrollments')
        yearly_totals = monthly_counts.groupby('Enrollment_Year')['Enrollments'].transform('sum')
        monthly_counts['Normalized_Enrollments'] = monthly_counts['Enrollments'] / yearly_totals

        # Map months to their names for the plot
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_counts['Month_Name'] = monthly_counts['Enrollment_Month'].map(lambda x: month_names[x - 1])

        # Create an area chart using Plotly
        fig = px.area(
            monthly_counts,
            x='Month_Name',
            y='Normalized_Enrollments',
            color='Enrollment_Year',
            title='Normalized Seasonal Variation in Enrollments',
            labels={
                'Month_Name': 'Month',
                'Normalized_Enrollments': 'Normalized Enrollments',
                'Enrollment_Year': 'Year'
            }
        )
        fig.update_layout(xaxis=dict(categoryorder='array', categoryarray=month_names))
        fig.show()
    except Exception as e:
        print("Error in plot_normalized_seasonality_area_chart:")
        print(e)


# In[17]:


plot_normalized_seasonality_area_chart(enrollments)


# ### Knowledge
# - ### Monthly Trends: Enrollments peak consistently from May to July, likely aligning with summer travel demand.
# - ### Normalized Data: Seasonal variations remain consistent across years, with proportional peaks during summer months.
# - ### Yearly Differences: Total enrollments vary by year, with 2018 showing the highest enrollments and 2012 the lowest.
# - ### Operational Insights: Peaks in summer suggest increased travel demand, useful for resource allocation and promotional planning.
# - ### Stable Patterns: The consistent seasonality indicates reliable trends for forecasting and optimizing operations.

# # Third Run : Plotting Enrollment Data by Grouping it According to Seasons.

# ### Data
# #### Used the `enrollments` dataframe and did a groupby on `Enrollment Year` and `Enrollment Season` to get the seasonal enrollment count for every year.

# In[18]:


def plot_seasonal_enrollments_area_chart(data: pd.DataFrame):
    """
    Plots the seasonal enrollments as an area chart for individual years using Plotly.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the time series data.
    """
    try:
        # Group data by year and season and count enrollments
        seasonal_counts = data.groupby(['Enrollment_Year', 'Enrollment_Season']).size().reset_index(name='Enrollments')
        
        # Define the order of seasons for proper visualization
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        seasonal_counts['Enrollment_Season'] = pd.Categorical(seasonal_counts['Enrollment_Season'], categories=season_order, ordered=True)
        
        # Create Plotly area chart for raw enrollments
        fig = px.area(
            seasonal_counts,
            x='Enrollment_Season',
            y='Enrollments',
            color='Enrollment_Year',
            title="Seasonal Enrollments by Year (Raw)",
            labels={
                'Enrollment_Season': 'Season',
                'Enrollments': 'Enrollments',
                'Enrollment_Year': 'Year'
            }
        )
        fig.update_layout(showlegend=True)
        fig.show()
    except Exception as e:
        print("Error in plot_seasonal_enrollments_area_chart (raw):", e)


# In[19]:


plot_seasonal_enrollments_area_chart(enrollments)


# In[20]:


def plot_normalized_seasonal_enrollments_area_chart(data: pd.DataFrame):
    """
    Plots the normalized seasonal enrollments as an area chart for individual years using Plotly.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the time series data.
    """
    try:
        # Group data by year and season and count enrollments
        seasonal_counts = data.groupby(['Enrollment_Year', 'Enrollment_Season']).size().reset_index(name='Enrollments')
        
        # Normalize enrollments by year
        yearly_totals = seasonal_counts.groupby('Enrollment_Year')['Enrollments'].transform('sum')
        seasonal_counts['Normalized_Enrollments'] = seasonal_counts['Enrollments'] / yearly_totals
        
        # Define the order of seasons for proper visualization
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        seasonal_counts['Enrollment_Season'] = pd.Categorical(seasonal_counts['Enrollment_Season'], categories=season_order, ordered=True)
        
        # Create Plotly area chart for normalized enrollments
        fig = px.area(
            seasonal_counts,
            x='Enrollment_Season',
            y='Normalized_Enrollments',
            color='Enrollment_Year',
            title="Seasonal Enrollments by Year (Normalized)",
            labels={
                'Enrollment_Season': 'Season',
                'Normalized_Enrollments': 'Normalized Enrollments',
                'Enrollment_Year': 'Year'
            }
        )
        fig.update_layout(showlegend=True)
        fig.show()
    except Exception as e:
        print("Error in plot_normalized_seasonal_enrollments_area_chart (normalized):", e)


# In[21]:


plot_normalized_seasonal_enrollments_area_chart(enrollments)


# ### Knowledge:
# 
# - #### Raw Area Chart:
# 	- ##### There is a slight increase in enrollments during Fall and Summer compared to Spring and Winter.
# 	- ##### However, the difference is not pronounced, suggesting that seasonal effects are minor compared to other factors influencing enrollments.
# - #### Normalized Area Chart:
# 	- ##### After normalizing by year, the seasonal variation becomes slightly more apparent because it removes the effect of varying yearly totals.
# 	- ##### Fall and Summer show higher proportions of enrollments consistently across most years.
# - #### Weak Seasonal Variation:
# 	- ##### The trend does not show drastic peaks or troughs, indicating that other factors (e.g., pricing, promotions, or broader economic trends) might be more significant drivers of enrollments.
# 
# 
# ### The seasonal variation in enrollments exists but is weak. It suggests that while seasons may have some influence (e.g., increased travel in Spring and Summer), they are not the primary factor determining enrollments. External variables or year-specific events may play a larger role.

# # Fourth Run : Quarterly Temporal Grouping

# ### Data
# #### Used the `enrollments` dataframe and did a groupby on `Enrollment_Year` and `Enrollment_Quarter` to get the quarter - wise counts of enrollment data for every year.

# ### Visualizations
# - #### Stacked bar chart using raw counts of quarterly enrollment data.
# - #### Stacked bar chart using normalized counts of quarterly enrollment data.

# In[22]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

def plot_quarterly_enrollments_subplots(data: pd.DataFrame, normalized=False):
    """
    Plots the quarterly enrollments as separate circular bar charts for each year using Plotly subplots.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the time series data.
    normalized (bool): Whether to plot normalized enrollments.
    """
    try:
        # Group data by year and quarter and count enrollments
        quarterly_counts = data.groupby(['Enrollment_Year', 'Enrollment_Quarter']).size().reset_index(name='Enrollments')

        # Normalize enrollments if specified
        if normalized:
            yearly_totals = quarterly_counts.groupby('Enrollment_Year')['Enrollments'].transform('sum')
            quarterly_counts['Enrollments'] = quarterly_counts['Enrollments'] / yearly_totals

        # Define the order of quarters for proper visualization
        quarter_order = [1, 2, 3, 4]
        quarterly_counts['Enrollment_Quarter'] = pd.Categorical(quarterly_counts['Enrollment_Quarter'], categories=quarter_order, ordered=True)
        
        # Map quarters to their labels (e.g., Q1, Q2, etc.)
        quarter_labels = ['Q1', 'Q2', 'Q3', 'Q4']
        quarterly_counts['Quarter_Label'] = quarterly_counts['Enrollment_Quarter'].map(lambda x: quarter_labels[int(x) - 1])
        
        # Get unique years for subplots
        years = quarterly_counts['Enrollment_Year'].unique()
        
        # Create subplots with one polar chart per year
        fig = make_subplots(
            rows=1, cols=len(years),  # Arrange all plots in a single row
            specs=[[{'type': 'polar'} for _ in years]],  # Set each subplot as polar
            subplot_titles=[str(year) for year in years],  # Titles for each year
            horizontal_spacing=0.05
        )
        
        # Add polar plots for each year
        for i, year in enumerate(years):
            year_data = quarterly_counts[quarterly_counts['Enrollment_Year'] == year]
            fig.add_trace(
                go.Barpolar(
                    r=year_data['Enrollments'],
                    theta=year_data['Quarter_Label'],
                    marker=dict(
                        color=year_data['Enrollments'],
                        colorscale='Viridis',  # Gradient color map for subtle changes
                        cmin=quarterly_counts['Enrollments'].min(),
                        cmax=quarterly_counts['Enrollments'].max(),
                        colorbar=dict(
                            title="Enrollments" if not normalized else "Normalized Enrollments",
                            orientation='h',  # Horizontal orientation
                            x=0.5,  # Centered horizontally
                            y=1.15,  # Position above the plots
                            xanchor='center'  # Align the center horizontally
                        ),
                    ),
                    name=str(year)
                ),
                row=1, col=i + 1  # Add to the correct subplot
            )
        
        # Update layout for better aesthetics
        fig.update_layout(
            title="Quarterly Enrollments by Year (Normalized)" if normalized else "Quarterly Enrollments by Year (Raw)",
            polar=dict(
                angularaxis=dict(
                    direction="clockwise",  # Set clockwise direction
                    rotation=90  # Start at the top
                ),
                radialaxis=dict(title="Enrollments (Normalized)" if normalized else "Enrollments", showticklabels=False)
            ),
            showlegend=False
        )
        
        # Show the final figure
        fig.show()
    except Exception as e:
        print("Error in plot_quarterly_enrollments_subplots:", e)


# In[23]:


plot_quarterly_enrollments_subplots(enrollments, False)


# In[24]:


plot_quarterly_enrollments_subplots(enrollments, True)


# ### Knowledge
# - #### There is no trend, except that Q1 has the lowest enrollments across all quarters.
# - #### Also, it is reconfirmed, that 2018 had the highest enrollment while 2012 had the lowest.
# - #### Also, in 2012 majority of the enrollments occured in Q3 and Q4.

# ### Final Findings.
# - #### First Run : 2012 lowest, with the enrollments being nearly the same across all the years except in 2018 when the enrollments spiked.
# - #### Second Run : Number of enrollments are highest in May and July possibly due to holiday seasons and lowest in January.
# - #### Third Run : The number of enrollments in the summer and fall season are high and low in spring and winter. Possible reason is holiday season during the summer and resumption of business activities in the winter.
# - #### Fourth Run : The quarter of the year has no bearing on the enrollments.

# 
