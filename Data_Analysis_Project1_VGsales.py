import pandas as pn
import matplotlib.pyplot as mplt
import seaborn as sb

# Loading the dataset
file_path = 'G:\\DAP\\vgsales.csv'
ds = pn.read_csv(file_path)

# Displaying the first few rows of the dataset 
print(ds.head())

# Checking for missing values 
missing_values = ds.isnull().sum()

# Displaying the count of missing values for each column
print("Missing values in each column:")
print(missing_values)

# Removing the rows with missing values
ds = ds.dropna()

# Checking for duplicates 
duplicates = ds.duplicated()

# Displaying the number of duplicate rows
print("Number of duplicate rows:", duplicates.sum())

# Removing duplicate rows
ds = ds.drop_duplicates()

# Checking for outliers in the 'Global_Sales' column

# Calculate the first quartile (Q1) and third quartile (Q3)
Q1 = ds['Global_Sales'].quantile(0.25)
Q3 = ds['Global_Sales'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identifying outliers
outliers = ds[(ds['Global_Sales'] < lower_bound) | (ds['Global_Sales'] > upper_bound)]

# Displaying the outliers
print("Outliers in the 'Global_Sales' column:")
print(outliers)

# Get summary statistics for numerical columns
summary_stats = ds.describe()

# Export the cleaned and processed data to CSV for Tableau
ds.to_csv('processed_vgsales.csv', index=False)

# Display summary statistics to understand the distribution of the data
print("Summary Statistics:")
print(summary_stats)

# Visualuzing a pi chart to find out games in which genre were sold the most
sales_by_genre = ds.groupby('Genre')['Global_Sales'].sum().reset_index()
sales_by_genre = sales_by_genre.sort_values(by='Global_Sales', ascending=False)
mplt.figure(figsize=(10, 8))
mplt.pie(sales_by_genre['Global_Sales'], labels=sales_by_genre['Genre'], autopct='%1.1f%%', startangle=140)
mplt.title('Total Global Sales by Genre')
mplt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
mplt.show()

# Visualizing sales in North America by date
# Converting the 'Date' column to datetime format
ds['Date'] = pn.to_datetime(ds['Date'])

# Setting the 'Date' column as the index 
ds.set_index('Date', inplace=True)

# Resampling by month and calculate the total sales for each month
monthly_sales = ds['NA_Sales'].resample('M').sum()

# Plotting the monthly sales trend to visualize how sales change over time
mplt.figure(figsize=(12, 6))
monthly_sales.plot()
mplt.title('NA sales by date')
mplt.xlabel('Date')
mplt.ylabel('NA sales (millions)')
mplt.show()



# Grouping the games sold by platform in Japan
category_sales = ds.groupby('Platform')['JP_Sales'].sum().reset_index()

# Sorting the sales
category_sales = category_sales.sort_values(by='JP_Sales', ascending=False)

# Ploting sales by product category to see which platfrom generate the most sales

mplt.figure(figsize=(10, 6))
mplt.bar(category_sales['Platform'], category_sales['JP_Sales'], color='skyblue')
mplt.xlabel('Platform')
mplt.ylabel('Sales in Japan (millions)')
mplt.title('Total Sales by Platform in Japan')
mplt.xticks(rotation=45)
mplt.show()

# Correlation matrix between Global, Japan and North America sales
sales = ds[['JP_Sales', 'NA_Sales', 'Global_Sales']]

# Calculating the correlation matrix
correlation_matrix = sales.corr()
print(correlation_matrix)

# Ploting the correlation matrix
mplt.figure(figsize=(8, 6))
sb.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
mplt.title('Correlation Matrix')
mplt.show()