# Short_Customer_Segmentation
This project includes a short customer segmentation of a small mall using clustering and exploratory analysis.

Check a file [Mall_Customer_Segmentation.pdf](https://github.com/claudia13062013/Short_Customer_Segmentation/files/13890488/Mall_Customer_Segmentation.pdf) to see a presentation of this analysis or go below to see slices of that presentation with a source code and plots.

## Table of contents :
* [Introduction](#introduction-)
* [Descriptive and Exploratory Analysis](#descriptive-and-exploratory-analysis-)
* [K-Means Clustering](#k-means-clustering-)
* [Conclusions](#conclusions-)

## Introduction :
Data set used in this project is in a file : 'Mall_Customers.csv'

Source code of this data analysis in :
- jupyter lab file: 'mall_customer_segmentation_clustering.ipynb'
- python file: mall_customer_segmentation_clustering.py

![Picture](https://github.com/claudia13062013/Short_Customer_Segmentation/blob/main/plots/introduction.jpg)

## Descriptive and Exploratory Analysis :
Firstly, descriptive analysis to get to know the data - distributions, amount, statistic values, visualisations
```python
# plots to visualize data : 
ax = sns.countplot(x='Gender', data=data, palette='pastel')
for container in ax.containers:
    ax.bar_label(container)
plt.show()

fig, axs = plt.subplots(ncols=3)
sns.histplot(x='Age', data=data, color='purple', ax=axs[0])
sns.histplot(x='Annual Income (k$)', data=data, color='purple', ax=axs[1])
sns.histplot(x='Spending Score (1-100)', data=data, color='purple', ax=axs[2])
```

![Picture](https://github.com/claudia13062013/Short_Customer_Segmentation/blob/main/plots/gender_plot.png)

![Picture](https://github.com/claudia13062013/Short_Customer_Segmentation/blob/main/plots/features_plot.png)

Data set does not have any missing data and significant outliers.

I used the one-hot-encoding method for 'Gender' values 

I used a Standard Scaler and then plot a correlation heatmap

```python
# preparing data to analysis :
df = data.drop(columns='CustomerID')
one_hot_encoded_data = pd.get_dummies(df, columns = ['Gender'],dtype=int) 
to_scal_data = one_hot_encoded_data.drop(columns=['Gender_Female', 'Gender_Male'])

# standardization to the same scale :
scaler = StandardScaler()
scaled_data = scaler.fit_transform(to_scal_data)
d_ready = np.append(scaled_data, one_hot_encoded_data['Gender_Female'].values.reshape(200, 1), axis=1)
d_ready = np.append(d_ready, one_hot_encoded_data['Gender_Male'].values.reshape(200, 1), axis=1)
```
```python
dfull_scaled = pd.DataFrame(d_ready, columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Female', 'Gender_Male'])
corr_p = dfull_scaled.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(corr_p, cmap="crest", vmax=0.9, fmt='.1f', annot=True)
plt.show()

plt.subplots(figsize=(12, 9))
corr_s = dfull_scaled.corr('spearman', numeric_only=False)
sns.heatmap(corr_s, cmap="crest", vmax=0.9, fmt='.1f', annot=True)
plt.show()

# testing if the 'Age' and 'Spending score' correlation is statistically significant: 
corr_bi = stats.pointbiserialr(dfull_scaled['Age'], dfull_scaled['Spending Score (1-100)'])
print(corr_bi)
age_spend = stats.spearmanr(dfull_scaled['Age'], dfull_scaled['Spending Score (1-100)'])
print(age_spend.pvalue)
corr_bi2 = stats.spearmanr(dfull_scaled['Age'], dfull_scaled['Annual Income (k$)'])
print(corr_bi2.pvalue)
corr_bi3 = stats.spearmanr(dfull_scaled['Spending Score (1-100)'], dfull_scaled['Annual Income (k$)'])
print(corr_bi3.pvalue)
```

Spearman's Correlation:

![Picture](https://github.com/claudia13062013/Short_Customer_Segmentation/blob/main/plots/correl_spearman.png)


The correlation between 'Age' feature and 'Spending Score(1-100)' feature is statistically significant:
SignificanceResult(statistic=-0.3272268460390901, pvalue=2.2502957035652467e-06)

Here is the scatter plot to see this negative correlation:
![Picture](https://github.com/claudia13062013/Short_Customer_Segmentation/blob/main/plots/scatter_spending_age.png)

## K-Means Clustering :
To make customers segmentation was used the k-means clustering method.
A right parameter 'k' for a clustering was decided with a help of 'elbow' scores.

```python
# K-Means Clustering :
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

# looking for right parameter k :
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10)).fit(data_2col)
visualizer.show()

```
![download](https://github.com/claudia13062013/Short_Customer_Segmentation/assets/97663507/eed2a7fe-ac75-43ef-b19e-203c2fa434b5)

```python
visualizer = KElbowVisualizer(model, k=(2,10), metric='calinski_harabasz', timings=False)

visualizer.fit(data_2col)
visualizer.poof()
```
![download](https://github.com/claudia13062013/Short_Customer_Segmentation/assets/97663507/18a965a5-7343-4c0b-a42c-24765b89474c)

```python
kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0).fit(data_2col)

sns.scatterplot(data=data, x="Age", y="Spending Score (1-100)", hue=kmeans.labels_)

plt.show()
```

![Picture](https://github.com/claudia13062013/Short_Customer_Segmentation/blob/main/plots/clustering.jpg)

```python
# 3 dimensional clustering :
data3d = data.drop(columns=['CustomerID', 'Gender'])
kmeans3d = KMeans(n_clusters = 3, init = 'k-means++',  random_state=42)
y = kmeans3d.fit_predict(scaled_data)
data3d['cluster'] = y

color_list = ['deeppink', 'blue', 'red', 'orange', 'darkviolet', 'brown']
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
# Data for 3-dimensional scattered points :
for i in range(data3d.cluster.nunique()):
    label = "cluster=" + str(i+1)
    ax.scatter3D(data3d[data3d.cluster==i]['Spending Score (1-100)'], data3d[data3d.cluster==i]['Annual Income (k$)'], data3d[data3d.cluster==i]['Age'], c=color_list[i], label=label)

ax.set_xlabel('Spending Score (1-100)')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Age')
plt.legend()
plt.title("Kmeans Clustering Of Mall's Customers")
plt.show()
```
I think that 3 subgroups of the data better show the main segmentations of customers:
- Red Cluster with customers with low spending score but various age and mostly high annual income
- Pink Cluster with customers with low or average annual income, age below 40 and average spending score
- Blue Cluster with customers high spending score, age below 30 and high or average annual income
  
![Picture](https://github.com/claudia13062013/Short_Customer_Segmentation/blob/main/plots/3dclustering.jpg)

## Conclusions :

![Picture](https://github.com/claudia13062013/Short_Customer_Segmentation/blob/main/plots/conclusions.jpg)

# Thank you for reading !
