#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score


# In[2]:


df_history = pd.read_csv('Updated_Customer_Loyalty_History.csv')


# In[3]:


df_history.info()


# In[4]:


df_history['Salary'].isnull().sum()


# In[5]:


df_history[df_history['Education']=='College'].shape


# In[6]:


df_history['Salary'] = df_history['Salary'].fillna(0)


# # Iteration 1: Plotting Salary bands v/s Enrollment Numbers

# + Assumption that salary should be a big factor in determining whether an individual is enrolling in the program or no

# In[7]:


df_history.info()


# In[8]:


# Define salary bands based on actual values
data_subset = df_history.copy()
salary_bins = [-1, 30000, 50000, 75000, 100000, np.inf] #To include all salaries including 0
salary_labels = ['Low (<30K)', 'Lower-Mid (30K-50K)', 'Upper-Mid (50K-75K)', 'High (75K-100K)', 'Very High (>100K)']

# Categorize salary into bands
data_subset['Salary_Band'] = pd.cut(data_subset['Salary'], bins=salary_bins, labels=salary_labels)


# In[9]:


# Group by Salary_Band to count enrollments
enrollment_by_band = data_subset.groupby('Salary_Band')['Salary'].count().reset_index()
enrollment_by_band.rename(columns={'Salary': 'Number of Enrollments'}, inplace=True)

# Extract categories and enrollment values
categories = enrollment_by_band['Salary_Band'].tolist()
values = enrollment_by_band['Number of Enrollments'].tolist()

# Compute angles
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()

# Correctly close the circle for both angles and values
angles.append(angles[0])
values.append(values[0])

# Radial Bar Chart with Boxed Categories
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Radial Bar Chart with Boxed Categories
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

bars = ax.bar(
    angles[:-1],  # Angles for the bars
    values[:-1],  # Heights of the bars
    color=sns.color_palette("husl", len(categories)), 
    align='center',
    alpha=0.8
)

# Format plot
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks([])  # Remove default category labels

# Add boxed category labels
for angle, category in zip(angles[:-1], categories):
    x_offset = 1.05  # Slightly outside the circle
    ax.text(
        angle, max(values) * x_offset,  # Position slightly outside the bars
        category,
        ha='center',
        va='center',
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white')
    )

# Add title with increased spacing
ax.set_title(
    "Radial View of Enrollments by Salary Band", 
    va='bottom', 
    fontsize=16, 
    pad=30  # Increase padding to separate title from the plot
)

plt.show()


# + We see that people in the Upper-Mid and High bands contribute the most to the total number of enrollments.
# + But suprisingly people from the lowest salary band also contribute a lot.
# + This can be due to missing context of the total population in each band.
# + Also we haven't check the distribution of the salary clmnn.
# + These things are done in the next iteration.

# # Iteration 2

# In[10]:


plt.figure(figsize=(8, 6))
df_history['Salary'].plot(kind='box')

# Adding labels and title
plt.title('Box Plot of Salary Distribution', fontsize=16, pad=20)
plt.ylabel('Salary', fontsize=14)
plt.xlabel('Distribution', fontsize=14)
plt.show()


# + From this plot it can be clearly seen that the salries are heavily concentrated in the lower range(<100K CAD)
# + This means that even slight changes in the band definitions could heavily impact the inferences drawn.
# + This means that using salary bands is not a reliable metric especially in the absence of standard definitions of the bands.
# + So in future iterations we analyze based on other demographic factors.

# In[11]:


# Count total population per salary band
total_population = data_subset['Salary_Band'].value_counts().sort_index()

# Count enrolled population per salary band (non-NaN in Enrollment Date)
enrolled_population = data_subset[~data_subset['Enrollment Date'].isnull()]['Salary_Band'].value_counts().sort_index()
not_enrolled_population = data_subset[data_subset['Enrollment Date'].isnull()]['Salary_Band'].value_counts().sort_index()

# Prepare data for plotting
bands = total_population.index.astype(str).tolist()
total_counts = total_population.values.tolist()

# Match enrolled counts to bands to avoid mismatches
enrolled_counts = [enrolled_population.get(band, 0) for band in bands]
not_enrolled_counts = [not_enrolled_population.get(band, 0) for band in bands]

import numpy as np
import matplotlib.pyplot as plt

# Back-to-Back Bar Chart with Improved Font Sizes and Visibility
fig, ax = plt.subplots(figsize=(12, 8))  # Increased figure size for better clarity

# Create horizontal bars for total and enrolled populations
x = np.arange(len(bands))  # Position for each band
width = 0.4  # Bar width
ax.barh(
    x - width / 2, 
    total_counts, 
    height=width, 
    label='Total Population', 
    color='gray', 
    alpha=0.7
)
ax.barh(
    x + width / 2, 
    enrolled_counts, 
    height=width, 
    label='Enrolled Population', 
    color='blue', 
    alpha=0.7
)

# Add labels and title with increased font size
ax.set_yticks(x)
ax.set_yticklabels(bands, fontsize=14)  # Font size for Salary Band labels
ax.set_xlabel('Count', fontsize=16)  # Font size for x-axis label
ax.set_title("Horizontal Stacked Bar Chart: Enrolled vs Total Population", fontsize=18)  # Font size for title
ax.legend(fontsize=14)  # Font size for legend

# Customize ticks for better readability
ax.tick_params(axis='x', labelsize=12)  # Font size for x-axis ticks
ax.tick_params(axis='y', labelsize=14)  # Font size for y-axis ticks

# Tight layout for better visibility
plt.tight_layout()
plt.show()



# + Looking at this chart we can clearly see that the total population count of a band and the total number of people enrolled in the band is the same.
# + This means that our data doesn't contain data for flyers that didn't enroll in the program.
# + This is something that has to be verified.

# # Iteration 3

# In[12]:


import matplotlib.pyplot as plt
import numpy as np

# Example data
categories = ['Total Population', 'Enrolled', 'Non-Enrolled']
values = [df_history.shape[0], -sum(enrolled_counts), -sum(not_enrolled_counts)]  # Use negative values for decreases

# Calculate cumulative values for the waterfall
cumulative = np.cumsum([0] + values[:-1])

# Create the waterfall chart
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['gray', 'blue', 'red']  # Define colors for each category

for i, (category, value) in enumerate(zip(categories, values)):
    ax.bar(category, value, bottom=cumulative[i], color=colors[i])
    # Add annotations
    ax.text(
        i, cumulative[i] + value / 2, f'{value:+}', ha='center', va='center', 
        color='white', fontsize=12  # Set fontsize for annotations
    )

# Add labels and title
ax.set_ylabel('Count', fontsize=14)  # Set fontsize for y-axis label
ax.set_title('Waterfall Chart: Total, Enrolled, and Non-Enrolled Population', fontsize=16)  # Set fontsize for title
ax.axhline(0, color='black', linewidth=1, linestyle='--')  # Add a baseline for reference

# Customize ticks
ax.tick_params(axis='x', labelsize=12)  # Set fontsize for x-axis ticks
ax.tick_params(axis='y', labelsize=12)  # Set fontsize for y-axis ticks

plt.tight_layout()
plt.show()


# + We don't have any data on the salary of people who did not enroll in the loyalty program.
# + So therefore in future iterations we only find patterns in the number of people that have enrolled.
# + To do that we first cluster based on other demographic clmns.

# # Iteration 4: Using other Demographic Variables

# In[13]:


df_history.info()


# In[14]:


data_subset = df_history[['Gender','Marital Status','Education','Loyalty Number']].groupby(['Gender','Marital Status','Education']).count()


# In[15]:


data_subset = data_subset.rename(columns = {'Loyalty Number':'Total Enrollments'}).reset_index()


# In[16]:


data_subset


# + There are three demographic variables other than salary to analyze:Gender,Education and Marital Status.
# + In order to do so Hierarchial Clustering has been used.

# In[17]:


import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Apply hierarchical clustering
Z = linkage(data_subset['Total Enrollments'].values.reshape(-1, 1), method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(Z, labels=data_subset.index.values, leaf_rotation=0, leaf_font_size=10)  # Set leaf_rotation to 0
plt.title('Hierarchical Clustering Dendrogram (Total Enrollments)', fontsize=14)
plt.xlabel('Index Order', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.show()


# In[18]:


import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# Assign cluster numbers (k=2 clusters)
data_subset['Cluster'] = fcluster(Z, t=2, criterion='maxclust')


# In[19]:


data_subset.groupby(['Cluster'])['Total Enrollments'].mean()


# In[20]:


data_subset


# In[21]:


import pandas as pd
import plotly.graph_objs as go

# Define fontsize and plot size variables
fontsize = 16  # Base font size
title_fontsize = fontsize + 10  # Larger font size for the title
plot_width = 900
plot_height = 700

# Use your provided dataset
data_subset = pd.DataFrame({
    "Gender": ["Female", "Female", "Female", "Female", "Female", "Female", "Female", "Female", "Female", "Female",
               "Female", "Female", "Female", "Female", "Female", "Male", "Male", "Male", "Male", "Male",
               "Male", "Male", "Male", "Male", "Male", "Male", "Male", "Male", "Male", "Male"],
    "Marital Status": ["Divorced", "Divorced", "Divorced", "Divorced", "Divorced", "Married", "Married", "Married",
                       "Married", "Married", "Single", "Single", "Single", "Single", "Single", "Divorced", "Divorced",
                       "Divorced", "Divorced", "Divorced", "Married", "Married", "Married", "Married", "Married",
                       "Single", "Single", "Single", "Single", "Single"],
    "Education": ["Bachelor", "College", "Doctor", "High School or Below", "Master", "Bachelor", "College", "Doctor",
                  "High School or Below", "Master", "Bachelor", "College", "Doctor", "High School or Below", "Master",
                  "Bachelor", "College", "Doctor", "High School or Below", "Master", "Bachelor", "College", "Doctor",
                  "High School or Below", "Master", "Bachelor", "College", "Doctor", "High School or Below", "Master"],
    "Total Enrollments": [822, 188, 96, 58, 104, 3569, 690, 227, 272, 105, 885, 1206, 49, 71, 58, 821, 188, 85, 66, 88,
                          3528, 746, 235, 245, 103, 831, 1220, 42, 69, 50],
    "Cluster": [2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
})

# Encode categorical data into unique integer mappings and retain original labels
encoded_data = data_subset.copy()
categorical_columns = ["Gender", "Marital Status", "Education"]

# Create mappings for each categorical column
label_mappings = {}
for col in categorical_columns:
    unique_categories = encoded_data[col].unique()
    label_mappings[col] = {category: idx for idx, category in enumerate(unique_categories)}
    encoded_data[col] = encoded_data[col].map(label_mappings[col])

# Define consistent axis ranges for uniform dimensions
dimensions = [
    dict(
        label="<b>Gender</b>",
        values=encoded_data["Gender"],
        tickvals=list(label_mappings["Gender"].values()),
        ticktext=list(label_mappings["Gender"].keys()),
        range=[min(encoded_data["Gender"]), max(encoded_data["Gender"])]  # Consistent range
    ),
    dict(
        label="<b>Marital Status</b>",
        values=encoded_data["Marital Status"],
        tickvals=list(label_mappings["Marital Status"].values()),
        ticktext=list(label_mappings["Marital Status"].keys()),
        range=[min(encoded_data["Marital Status"]), max(encoded_data["Marital Status"])]  # Consistent range
    ),
    dict(
        label="<b>Education</b>",
        values=encoded_data["Education"],
        tickvals=list(label_mappings["Education"].values()),
        ticktext=list(label_mappings["Education"].keys()),
        range=[min(encoded_data["Education"]), max(encoded_data["Education"])]  # Consistent range
    ),
    dict(
        label="<b>Total Enrollments</b>",
        values=data_subset["Total Enrollments"],
        range=[min(data_subset["Total Enrollments"]), max(data_subset["Total Enrollments"])]  # Consistent range
    ),
]

# Create traces for each cluster
traces = []
cluster_colors = {1: "blue", 2: "red"}

for cluster, color in cluster_colors.items():
    cluster_mask = data_subset["Cluster"] == cluster  # Filter data for the cluster
    traces.append(
        go.Parcoords(
            visible=False,  # Initially hidden
            line=dict(
                color=data_subset.loc[cluster_mask, "Cluster"],
                colorscale=[[0, color], [1, color]],
                showscale=False,
            ),
            dimensions=[
                dict(dimensions[0], values=encoded_data.loc[cluster_mask, "Gender"]),
                dict(dimensions[1], values=encoded_data.loc[cluster_mask, "Marital Status"]),
                dict(dimensions[2], values=encoded_data.loc[cluster_mask, "Education"]),
                dict(dimensions[3], values=data_subset.loc[cluster_mask, "Total Enrollments"]),
            ],
        )
    )

# Add "Show All" trace
traces.append(
    go.Parcoords(
        visible=True,  # Initially visible
        line=dict(
            color=data_subset["Cluster"],
            colorscale=[[0, "blue"], [1, "red"]],
            showscale=False,
        ),
        dimensions=dimensions,  # Use consistent dimensions
    )
)

# Create buttons
buttons = [
    dict(
        label=f"Cluster {cluster}",
        method="update",
        args=[
            {"visible": [i == cluster - 1 for i in range(len(traces) - 1)] + [False]},
            {"title": f"Cluster {cluster} - Parallel Coordinates Plot"},
        ],
    )
    for cluster in cluster_colors.keys()
]

# Add "Show All" button
buttons.append(
    dict(
        label="Show All",
        method="update",
        args=[
            {"visible": [False] * (len(traces) - 1) + [True]},
            {"title": "All Clusters - Parallel Coordinates Plot"},
        ],
    )
)

# Create layout
layout = go.Layout(
    title=dict(
        text="Interactive Parallel Coordinates Plot with Uniform Axes",
        font=dict(size=title_fontsize, family="Arial"),
        x=0.5,  # Center the title
        xanchor="center",
        y=0.95,  # Adjust vertical position
    ),
    font=dict(size=fontsize),
    width=plot_width,
    height=plot_height,
    margin=dict(l=100, r=180, t=120, b=50),  # Adjusted margins for better spacing
    updatemenus=[
        dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=1.2,
            y=1.0,
            xanchor="left",
            yanchor="top",
        )
    ],
)

# Create figure
fig = go.Figure(data=traces, layout=layout)

# Display the figure
fig.show()


# # Iteration 5

# + In this iteration we want to deepdive into cluster 2.
# + In order to do this we divide the entire data into 3 clusters instead of 2.
# + It is clear from the dendrogram above the clusters 2 and 3 finally combine to form the cluster 2 from above.

# In[22]:


data_subset['Cluster'] = fcluster(Z, t=3, criterion='maxclust')
data_subset.groupby(['Cluster'])['Total Enrollments'].mean()


# In[23]:


data_subset.groupby(['Cluster'])['Total Enrollments'].mean().plot(kind='bar', color='blue', figsize=(12, 6))
plt.xlabel('Cluster', fontsize=14)
plt.xticks(rotation=0)
plt.ylabel('Average Total Enrollments', fontsize=14)
plt.title('Average Total Enrollments by Cluster', fontsize=16)
plt.show()


# + According to the dendrogram cluster 1 should be the same as the one in the previuos iteration.
# + Clusters 2 and 3 are the ones that make up the cluster 2 from the prvious iteration and so they are the ones that need to be studied further.
# + It can clearly be seen that cluster 3 is far superior in terms of average number of enrollments as compared to cluster 2 and hence the demographic features of cluster 3 will be studies further.

# In[24]:


filtered_data = data_subset[data_subset["Cluster"].isin([2, 3])]


# In[25]:


import plotly.graph_objects as go

bluish_color_scheme = "Blues"

def create_enhanced_sunburst_chart(cluster, color_scale):
    cluster_data = filtered_data[filtered_data["Cluster"] == cluster]

    labels = []
    parents = []
    values = []

    # Root (Cluster)
    labels.append(f"Cluster {cluster}")
    parents.append("")
    values.append(cluster_data["Total Enrollments"].sum())

    # Gender layer
    genders = cluster_data["Gender"].unique()
    for gender in genders:
        gender_total = cluster_data[cluster_data["Gender"] == gender]["Total Enrollments"].sum()
        labels.append(f"{gender} (C{cluster})")
        parents.append(f"Cluster {cluster}")
        values.append(gender_total)

    # Marital Status layer
    for gender in genders:
        gender_data = cluster_data[cluster_data["Gender"] == gender]
        marital_statuses = gender_data["Marital Status"].unique()
        for status in marital_statuses:
            status_total = gender_data[gender_data["Marital Status"] == status]["Total Enrollments"].sum()
            labels.append(f"{status} (C{cluster}, {gender})")
            parents.append(f"{gender} (C{cluster})")
            values.append(status_total)

    # Education layer
    for gender in genders:
        gender_data = cluster_data[cluster_data["Gender"] == gender]
        marital_statuses = gender_data["Marital Status"].unique()
        for status in marital_statuses:
            status_data = gender_data[gender_data["Marital Status"] == status]
            educations = status_data["Education"].unique()
            for education in educations:
                education_total = status_data[status_data["Education"] == education]["Total Enrollments"].sum()
                labels.append(f"{education} (C{cluster}, {gender}, {status})")
                parents.append(f"{status} (C{cluster}, {gender})")
                values.append(education_total)

    # Create the sunburst chart
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        insidetextorientation="radial",
        marker=dict(colors=values, colorscale=color_scale, line=dict(color="black", width=1)),  # Black boundaries
        hoverinfo="label+value+percent entry",
    ))

    # Update traces to emphasize visibility
    fig.update_traces(
        textfont=dict(size=16, family="Arial, sans-serif", color="black")  # Black text inside the sunburst
    )

    # Update layout for better contrast and visibility
    fig.update_layout(
        title={
            "text": f"Sunburst Chart for Cluster {cluster}",
            "x": 0.5,
            "xanchor": "center",
            "font": dict(size=24, weight="bold", color="black"),  # Black title font
        },
        margin=dict(t=50, l=20, r=20, b=20),
        paper_bgcolor="white",  # White background
        sunburstcolorway=["#636efa", "#ef553b", "#00cc96", "#ab63fa"],  # Custom contrasting colors
    )

    return fig

# Generate enhanced chart
fig_cluster_3 = create_enhanced_sunburst_chart(3, bluish_color_scheme)

# Save as a high-resolution image
# fig_cluster_3.write_image("sunburst_chart_high_res.png", width=1920, height=1080, scale=2)

# Show the chart
fig_cluster_3.show()


# + Over here it can clearly be see that in Cluster 3 Gender has no effect on the total enrollmens.
# + In both genders single college going students seem to be contributing the most to the enrollment numbers.
# + But this number will still be quite less as compared to Married people with a bachelor's degree.

# 
