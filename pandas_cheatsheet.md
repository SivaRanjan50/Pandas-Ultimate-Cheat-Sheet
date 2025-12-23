# Pandas-Ultimate-Cheat-Sheet

# 📋 TABLE OF CONTENTS
    Installation & Setup
    Data Structures
    Reading & Writing Data
    Viewing & Inspecting Data
    Selection & Filtering
    Data Cleaning
    Data Transformation
    Grouping & Aggregation
    Merging & Joining
    Time Series
    Visualization
    Performance Tips

# 📦 INSTALLATION & SETUP
## installation
```bash
# Using pip
pip install pandas

# Using conda
conda install pandas

# Install with optional dependencies
pip install "pandas[excel,html,parquet]"
```

## Import Convention
```python
import pandas as pd
import numpy as np
```

## Version Check
```python
print(pd.__version__)
```

# 📊 DATA STRUCTURES
## Series - 1D labeled array
```python
# Create Series
s = pd.Series([1, 3, 5, np.nan, 6, 8])
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s = pd.Series({'a': 1, 'b': 2, 'c': 3})

# Series operations
s.values      # Array of values
s.index       # Index object
s.dtype       # Data type
s.shape       # Shape (n,)
s.size        # Number of elements
s.name        # Name attribute
s.rename("new_name")  # Rename
```

## DataFrame - 2D labeled data structure
```python
# Create DataFrame from dict
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': ['p', 'q', 'r']
})

# Create DataFrame from list of dicts
df = pd.DataFrame([
    {'A': 1, 'B': 4, 'C': 'p'},
    {'A': 2, 'B': 5, 'C': 'q'},
    {'A': 3, 'B': 6, 'C': 'r'}
])

# Create with index
df = pd.DataFrame(data, index=['row1', 'row2', 'row3'])

# From NumPy array
df = pd.DataFrame(np.random.randn(3, 4), columns=['A', 'B', 'C', 'D'])

# From CSV/Excel/SQL (see Reading Data section)
```

## DataFrame Attributes
```python
df.shape           # (rows, columns)
df.columns         # Column names
df.index           # Row index
df.dtypes          # Data types per column
df.info()          # Summary info
df.describe()      # Statistical summary
df.values          # NumPy array
df.T               # Transpose
df.memory_usage()  # Memory usage per column
```

# 📁 READING & WRITING DATA
## Reading Files
```python
# CSV
df = pd.read_csv('file.csv')
df = pd.read_csv('file.csv', sep=',', header=0, index_col=0)
df = pd.read_csv('file.csv', usecols=['col1', 'col2'])
df = pd.read_csv('file.csv', nrows=1000)  # Read only first 1000 rows
df = pd.read_csv('file.csv', dtype={'col1': str, 'col2': int})
df = pd.read_csv('file.csv', parse_dates=['date_column'])
df = pd.read_csv('file.csv', encoding='utf-8')

# Excel
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')
df = pd.read_excel('file.xlsx', sheet_name=0)  # First sheet
df = pd.read_excel('file.xlsx', usecols='A:C,E')  # Specific columns

# JSON
df = pd.read_json('file.json')
df = pd.read_json('file.json', orient='records')

# Parquet
df = pd.read_parquet('file.parquet')

# SQL
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM table', conn)
df = pd.read_sql_table('table_name', conn)
df = pd.read_sql_query('SELECT * FROM table', conn)

# HTML (web scraping)
df = pd.read_html('http://url.com/table.html')[0]

# Clipboard (from Excel/Sheets)
df = pd.read_clipboard()

# Multiple files
import glob
files = glob.glob('data/*.csv')
df_list = [pd.read_csv(f) for f in files]
df = pd.concat(df_list, ignore_index=True)
```

## Writing Files
```python
# CSV
df.to_csv('output.csv', index=False)
df.to_csv('output.csv', sep=';', encoding='utf-8')

# Excel
df.to_excel('output.xlsx', sheet_name='Sheet1', index=False)
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')

# JSON
df.to_json('output.json')
df.to_json('output.json', orient='records')

# Parquet
df.to_parquet('output.parquet')

# SQL
df.to_sql('table_name', conn, if_exists='replace', index=False)

# Pickle (fast Python serialization)
df.to_pickle('output.pkl')
df = pd.read_pickle('output.pkl')

# HTML
df.to_html('output.html')
```

# 👀 VIEWING & INSPECTING DATA
## Viewing Data
```python
# First/last n rows
df.head()          # First 5 rows
df.head(10)        # First 10 rows
df.tail()          # Last 5 rows
df.tail(3)         # Last 3 rows

# Sample rows
df.sample(5)       # Random 5 rows
df.sample(frac=0.1)  # 10% random sample
df.sample(n=5, random_state=42)  # Reproducible sample

# View specific rows
df.iloc[0]         # First row
df.iloc[[0, 2, 4]] # Multiple rows by position
df.loc['row_name'] # Row by index label
```

## Data Info
```python
# Basic info
df.info()          # Data types, non-null counts
df.shape           # (rows, columns)
df.columns         # Column names
df.index           # Index
df.dtypes          # Data type per column
df.memory_usage(deep=True)  # Memory usage
                            (returns in bytes)

# Statistics
df.describe()      # Summary statistics for numeric columns
df.describe(include='all')  # All columns
df.describe(include=[np.number])  # Only numeric
df.describe(include=[object])     # Only object/string
df.describe(percentiles=[0.1, 0.5, 0.9])

# Counts
df.count()         # Non-null values per column
df.nunique()       # Unique values per column
df.value_counts()  # What it does: Counts the frequency of each unique value in a Series.
df['col'].value_counts(normalize=True)  # Proportions(Shows the proportion of each value instead of raw counts.)
                                        # Use case: When you want to understand the distribution as percentages rather than absolute numbers.
df['col'].value_counts(dropna=False)    # Include NaN(null/None) values in the count.
                                        # (By default, value_counts() excludes NaN values.)
        
```

## Checking Values
```python
# Unique values
df['col'].unique()        # Array of unique values
df['col'].nunique()       # Count of unique values
df['col'].value_counts()  # Frequency of each unique value

# Check for duplicates
df.duplicated()           # Boolean Series
df.duplicated().sum()     # Count duplicates
df[df.duplicated()]       # Show duplicate rows
df.duplicated(subset=['col1', 'col2'])  # Check specific columns

# Check for nulls
df.isnull()               # Boolean DataFrame
df.isna()                 # Alias for isnull()
df.notnull()              # Opposite of isnull()
df.isnull().sum()         # Count nulls per column
df.isnull().sum().sum()   # Total nulls in DataFrame
df[df['col'].isnull()]    # Rows where column is null
```

# 🎯 SELECTION & FILTERING
## Column Selection
```python
# Single column (returns Series)
df['col_name']
df.col_name        # Only if column name has no spaces/special chars

# Multiple columns (returns DataFrame)
df[['col1', 'col2']]

# Columns by data type
df.select_dtypes(include=['number'])      # Numeric columns
df.select_dtypes(include=['object'])      # String columns
df.select_dtypes(include=['datetime'])    # Datetime columns
df.select_dtypes(exclude=['number'])      # Non-numeric columns

# Columns matching pattern
df.filter(like='price')    # Columns containing 'price'
df.filter(regex='^A')      # Columns starting with 'A'
df.filter(regex='\d$')     # Columns ending with digit
```

## Row Selection (by label) - loc
```python
# Single row
df.loc['row_label']        # Returns Series
df.loc[['row_label']]      # Returns DataFrame

# Multiple rows
df.loc[['row1', 'row2', 'row3']]

# Row and column selection
df.loc['row_label', 'col_name']          # Scalar value
df.loc['row_label', ['col1', 'col2']]    # Multiple columns
df.loc[['row1', 'row2'], 'col_name']     # Multiple rows, single column
df.loc[['row1', 'row2'], ['col1', 'col2']]  # Both

# Slicing with labels
df.loc['row1':'row3']                    # Includes both endpoints
df.loc['row1':'row3', 'col1':'col3']     # Row and column slices

# Boolean indexing with loc
df.loc[df['col'] > 5]
df.loc[df['col'].isin([1, 2, 3])]
df.loc[(df['col1'] > 5) & (df['col2'] == 'A')]
```

## Row Selection (by position) - iloc
```python
# Single row
df.iloc[0]                 # First row (Series)
df.iloc[[0]]               # First row (DataFrame)

# Multiple rows
df.iloc[[0, 2, 4]]         # Rows 0, 2, 4

# Row and column selection
df.iloc[0, 1]              # Row 0, Column 1 (scalar)
df.iloc[0, [1, 3]]         # Row 0, Columns 1 and 3
df.iloc[[0, 2], 1]         # Rows 0 and 2, Column 1
df.iloc[[0, 2], [1, 3]]    # Rows 0,2 and Columns 1,3

# Slicing
df.iloc[0:5]               # Rows 0 to 4 (5 rows)
df.iloc[0:5, 1:4]          # Rows 0-4, Columns 1-3
df.iloc[::2]               # Every other row
df.iloc[::-1]              # Reverse order
```

## Boolean Indexing
```python
# Single condition
df[df['col'] > 5]
df[df['col'] == 'value']
df[df['col'].isin(['A', 'B', 'C'])]

# Multiple conditions
df[(df['col1'] > 5) & (df['col2'] == 'A')]  # AND
df[(df['col1'] > 5) | (df['col2'] == 'A')]  # OR
df[~(df['col'] == 'A')]                     # NOT

# String methods
df[df['col'].str.contains('pattern')]
df[df['col'].str.startswith('A')]
df[df['col'].str.endswith('Z')]
df[df['col'].str.match('^A.*Z$')]  # Regex
df[df['col'].str.len() > 10]

# Query method (SQL-like)
df.query('col1 > 5 and col2 == "A"')
df.query('col1 in [1, 2, 3]')
df.query('col1 > @threshold')  # Use variable

# Between
df[df['col'].between(10, 20)]           # Inclusive
df[df['col'].between(10, 20, inclusive='neither')]

# Null checks
df[df['col'].isnull()]
df[df['col'].notnull()]
```

## Random Selection
```python
# Random rows
df.sample(n=5)              # 5 random rows
df.sample(frac=0.1)         # 10% of rows
df.sample(n=5, random_state=42)  # Reproducible

# Random columns
df.sample(n=3, axis=1)      # 3 random columns

# Weights
df.sample(n=5, weights='col_name')  # Weighted by column
```

# 🧹 DATA CLEANING
## Handling Missing Data
```python
# Check for missing values
df.isnull()
df.isna()
df.notnull()
df.isnull().sum()
df.isnull().sum().sum()

# Drop missing values
df.dropna()                 # Drop rows with any NaN
df.dropna(axis=1)           # Drop columns with any NaN
df.dropna(how='all')        # Drop rows where all values are NaN
df.dropna(thresh=3)         # Keep rows with at least 3 non-NaN
df.dropna(subset=['col1', 'col2'])  # Only check specific columns

# Fill missing values
df.fillna(0)                # Fill with 0
df.fillna(method='ffill')   # Forward fill
df.fillna(method='bfill')   # Backward fill
df.fillna(df.mean())        # Fill with column mean
df.fillna(df.median())      # Fill with column median
df.fillna(df.mode().iloc[0]) # Fill with mode

# Interpolation
df.interpolate()            # Linear interpolation
df.interpolate(method='time')  # Time-based interpolation
df.interpolate(limit=2)     # Max consecutive interpolations
df.interpolate(limit_direction='both')  # Interpolate in both directions

# Replace specific values
df.replace(-999, np.nan)    # Replace -999 with NaN
df.replace([-999, -1000], np.nan)
df.replace({-999: np.nan, 'N/A': None})
df.replace({'col1': {-999: np.nan}, 'col2': {'N/A': None}})
```

## Handling Duplicates
```python
# Find duplicates
df.duplicated()                     # Boolean Series
df.duplicated(subset=['col1', 'col2'])  # Check specific columns
df.duplicated(keep='first')         # Mark all but first as duplicates
df.duplicated(keep='last')          # Mark all but last as duplicates
df.duplicated(keep=False)           # Mark all duplicates

# Remove duplicates
df.drop_duplicates()                # Remove duplicate rows
df.drop_duplicates(subset=['col1', 'col2'])  # Based on columns
df.drop_duplicates(keep='first')    # Keep first occurrence
df.drop_duplicates(keep='last')     # Keep last occurrence
df.drop_duplicates(keep=False)      # Remove all duplicates
df.drop_duplicates(inplace=True)    # Modify in place
```

## Data Type Conversion
```python
# Check types
df.dtypes
df['col'].dtype

# Convert types
df['col'] = df['col'].astype(int)
df['col'] = df['col'].astype(float)
df['col'] = df['col'].astype(str)
df['col'] = df['col'].astype('category')
df['col'] = pd.to_numeric(df['col'])  # Convert to numeric, errors='coerce'

# Datetime conversion
df['date_col'] = pd.to_datetime(df['date_col'])
df['date_col'] = pd.to_datetime(df['date_col'], format='%Y-%m-%d')
df['date_col'] = pd.to_datetime(df['date_col'], errors='coerce')

# Convert all columns
df = df.convert_dtypes()  # Convert to best possible dtypes
df = df.infer_objects()   # Infer better dtypes
```

## String Operations
```python
# Case conversion
df['col'].str.lower()
df['col'].str.upper()
df['col'].str.title()
df['col'].str.capitalize()

# Strip whitespace
df['col'].str.strip()
df['col'].str.lstrip()
df['col'].str.rstrip()

# Split strings
df['col'].str.split(',')
df['col'].str.split(',', expand=True)  # Split into columns
df['col'].str.split(',', n=1)          # Split into 2 parts only

# Replace patterns
df['col'].str.replace('old', 'new')
df['col'].str.replace(r'\d+', 'X', regex=True)  # Regex

# Extract patterns
df['col'].str.extract(r'(\d+)')        # Extract first number
df['col'].str.extractall(r'(\d+)')     # Extract all numbers
df['col'].str.findall(r'\d+')          # Find all matches

# Check patterns
df['col'].str.contains('pattern')
df['col'].str.startswith('A')
df['col'].str.endswith('Z')
df['col'].str.match('^A.*Z$')          # Full string match

# String length
df['col'].str.len()

# Join strings
df['col1'].str.cat(df['col2'], sep='-')
df[['col1', 'col2']].agg('-'.join, axis=1)
```

# 🔄 DATA TRANSFORMATION
## Adding/Removing Columns
```python
# Add new column
df['new_col'] = values
df['new_col'] = df['col1'] + df['col2']
df['new_col'] = df['col'].apply(lambda x: x*2)
df = df.assign(new_col=df['col1'] + df['col2'])  # Returns new DataFrame
df = df.assign(new_col1=val1, new_col2=val2)     # Multiple columns

# Insert column at specific position
df.insert(2, 'new_col', values)  # Insert at position 2 (0-indexed)

# Rename columns
df.rename(columns={'old_name': 'new_name'})
df.rename(columns=str.lower)      # All columns to lowercase
df.rename(columns=lambda x: x.replace(' ', '_'))
df.columns = ['col1', 'col2', 'col3']  # Replace all names
df.columns = df.columns.str.lower()    # Convert all to lowercase

# Remove columns
df.drop('col_name', axis=1)
df.drop(['col1', 'col2'], axis=1)
df.drop(columns=['col1', 'col2'])
del df['col_name']                     # In-place deletion
```

## Adding/Removing Rows
```python
# Add rows
df.append(new_row)                     # Deprecated in pandas 1.4.0
pd.concat([df1, df2])                  # Preferred method
pd.concat([df1, df2], ignore_index=True)

# Insert row at specific position
df.loc['new_index'] = row_values

# Remove rows
df.drop('row_label')
df.drop(['row1', 'row2'])
df.drop(index='row_label')
df.drop(index=['row1', 'row2'])
```

## Reshaping Data
```python
# Melt (wide to long)
pd.melt(df, id_vars=['id'], value_vars=['col1', 'col2'])
pd.melt(df, id_vars=['id'], value_vars=['col1', 'col2'], 
        var_name='variable', value_name='value')

# Pivot (long to wide)
df.pivot(index='id', columns='variable', values='value')

# Pivot table (with aggregation)
df.pivot_table(index='col1', columns='col2', values='col3', aggfunc='mean')
df.pivot_table(values='value', index='date', columns='type', 
               aggfunc='sum', fill_value=0)

# Stack/Unstack
df.stack()        # Columns to rows
df.unstack()      # Rows to columns

# Crosstab
pd.crosstab(df['col1'], df['col2'])
pd.crosstab(df['col1'], df['col2'], normalize=True)  # Proportions
pd.crosstab(df['col1'], df['col2'], values=df['col3'], aggfunc='mean')
```

## Applying Functions
```python
# Apply function to Series
df['col'].apply(lambda x: x*2)
df['col'].apply(np.sqrt)
df['col'].apply(len)  # For strings/lists

# Apply function to DataFrame (row-wise or column-wise)
df.apply(np.sum, axis=0)      # Column-wise (default)
df.apply(np.sum, axis=1)      # Row-wise
df.apply(lambda x: x.max() - x.min())  # Column-wise

# Applymap (element-wise)
df.applymap(lambda x: len(str(x)))
df.applymap(str.upper)        # For string columns

# Vectorized operations (fastest)
df['new_col'] = df['col1'] + df['col2']
df['new_col'] = np.log(df['col'])
df['new_col'] = np.where(df['col'] > 0, 'Positive', 'Non-positive')

# Using NumPy ufuncs
np.sqrt(df['col'])
np.log(df['col'] + 1)
```

## Binning & Discretization
```python
# Cut (bins based on values)
pd.cut(df['col'], bins=5)
pd.cut(df['col'], bins=[0, 10, 20, 30, 40])
pd.cut(df['col'], bins=[0, 10, 20, 30, 40], labels=['A', 'B', 'C', 'D'])

# qcut (bins based on quantiles)
pd.qcut(df['col'], q=4)                     # Quartiles
pd.qcut(df['col'], q=[0, 0.25, 0.5, 0.75, 1])  # Custom quantiles
pd.qcut(df['col'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

## Sorting
```python
# Sort by values
df.sort_values('col')
df.sort_values(['col1', 'col2'])
df.sort_values('col', ascending=False)
df.sort_values(['col1', 'col2'], ascending=[True, False])

# Sort by index
df.sort_index()
df.sort_index(ascending=False)

# Get n largest/smallest
df.nlargest(5, 'col')
df.nsmallest(5, 'col')
```

# 📊 GROUPING & AGGREGATION
## Basic Grouping
```python
# Group by single column
grouped = df.groupby('col')

# Group by multiple columns
grouped = df.groupby(['col1', 'col2'])

# Group by index level
df.groupby(level=0)  # First level of MultiIndex

# Iterate over groups
for name, group in df.groupby('col'):
    print(name)
    print(group)

# Get specific group
df.groupby('col').get_group('group_name')

# Group sizes
df.groupby('col').size()
df.groupby('col').count()
```

## Aggregation
```python
# Single aggregation function
df.groupby('col')['value'].sum()
df.groupby('col')['value'].mean()
df.groupby('col')['value'].std()
df.groupby('col')['value'].min()
df.groupby('col')['value'].max()
df.groupby('col')['value'].count()
df.groupby('col')['value'].nunique()

# Multiple aggregation functions
df.groupby('col')['value'].agg(['sum', 'mean', 'std'])
df.groupby('col')['value'].agg([np.sum, np.mean, np.std])

# Different aggregations per column
df.groupby('col').agg({'value1': 'sum', 'value2': 'mean'})
df.groupby('col').agg({'value1': ['sum', 'mean'], 'value2': 'std'})

# Custom aggregation
df.groupby('col')['value'].agg(lambda x: x.max() - x.min())
df.groupby('col')['value'].agg(['sum', lambda x: x.quantile(0.95)])

# Named aggregations (pandas 0.25+)
df.groupby('col').agg(
    total=('value', 'sum'),
    average=('value', 'mean'),
    stdev=('value', 'std')
)
```

## Transformation
```python
# Transform (same shape as original)
df.groupby('col')['value'].transform('sum')
df.groupby('col')['value'].transform(lambda x: x - x.mean())
df['group_mean'] = df.groupby('col')['value'].transform('mean')

# Normalize within group
df['normalized'] = df.groupby('col')['value'].transform(
    lambda x: (x - x.mean()) / x.std()
)
```

## Filtering Groups
```python
# Filter groups based on condition
df.groupby('col').filter(lambda x: x['value'].mean() > 100)
df.groupby('col').filter(lambda x: len(x) > 5)
df.groupby('col').filter(lambda x: x['value'].sum() > 1000)
```

## Window Operations
```python
# Rolling window
df['rolling_mean'] = df['value'].rolling(window=3).mean()
df['rolling_sum'] = df['value'].rolling(window=7).sum()
df['rolling_std'] = df['value'].rolling(window=5).std()

# Expanding window
df['expanding_mean'] = df['value'].expanding().mean()
df['expanding_sum'] = df['value'].expanding().sum()

# Exponential weighted moving average
df['ewm'] = df['value'].ewm(span=10).mean()
```

# 🔗 MERGING & JOINING
## Merge (SQL-like joins)
```python
# Inner join (default)
pd.merge(df1, df2, on='key')
pd.merge(df1, df2, left_on='key1', right_on='key2')

# Outer join
pd.merge(df1, df2, on='key', how='outer')

# Left join
pd.merge(df1, df2, on='key', how='left')

# Right join
pd.merge(df1, df2, on='key', how='right')

# Merge on multiple keys
pd.merge(df1, df2, on=['key1', 'key2'])

# Merge with indicator
pd.merge(df1, df2, on='key', how='outer', indicator=True)

# Merge with suffixes
pd.merge(df1, df2, on='key', suffixes=('_left', '_right'))
```

## Concatenation
```python
# Stack DataFrames vertically
pd.concat([df1, df2, df3])
pd.concat([df1, df2], ignore_index=True)      # Reset index
pd.concat([df1, df2], keys=['A', 'B'])        # Add keys
pd.concat([df1, df2], axis=0)                 # Same as default

# Stack DataFrames horizontally
pd.concat([df1, df2], axis=1)
pd.concat([df1, df2], axis=1, join='inner')   # Inner join

# Concatenate with different indexes
pd.concat([df1, df2], axis=1, join_axes=[df1.index])  # Keep df1 index
```

## Join (index-based)
```python
# Join on index
df1.join(df2)                     # Left join by default
df1.join(df2, how='inner')
df1.join(df2, how='outer')
df1.join(df2, how='right')

# Join on column
df1.set_index('key').join(df2.set_index('key'))
df1.join(df2.set_index('key'), on='key')  # Join on column
```

## Combine
```python
# Combine first (keep first non-null)
df1.combine_first(df2)

# Combine with function
df1.combine(df2, lambda s1, s2: s1 if s1.sum() > s2.sum() else s2)
```


# ⏰ TIME SERIES
## Date Range
```python
# Create date range
pd.date_range('2023-01-01', '2023-12-31', freq='D')
pd.date_range(start='2023-01-01', periods=365, freq='D')
pd.date_range(end='2023-12-31', periods=12, freq='M')

# Frequencies
# 'D' - daily, 'B' - business day, 'W' - weekly
# 'M' - month end, 'MS' - month start
# 'Q' - quarter end, 'QS' - quarter start
# 'A' - year end, 'AS' - year start
# 'H' - hourly, 'T' or 'min' - minutely, 'S' - secondly

# Custom frequencies
pd.date_range('2023-01-01', periods=10, freq='2D')  # Every 2 days
pd.date_range('2023-01-01', periods=5, freq='W-MON')  # Every Monday
```

## Datetime Properties
```python
# Extract datetime components
df['date'].dt.year
df['date'].dt.month
df['date'].dt.day
df['date'].dt.hour
df['date'].dt.minute
df['date'].dt.second
df['date'].dt.weekday        # Monday=0, Sunday=6
df['date'].dt.dayofweek      # Same as weekday
df['date'].dt.dayofyear
df['date'].dt.quarter
df['date'].dt.is_month_start
df['date'].dt.is_month_end
df['date'].dt.is_quarter_start
df['date'].dt.is_quarter_end
df['date'].dt.is_year_start
df['date'].dt.is_year_end

# Format dates
df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
df['month_year'] = df['date'].dt.to_period('M')
```

## Resampling
```python
# Downsampling (to lower frequency)
df.resample('D').sum()          # Daily to weekly
df.resample('W').mean()         # Daily to weekly mean
df.resample('M').agg({'col1': 'sum', 'col2': 'mean'})

# Upsampling (to higher frequency)
df.resample('H').ffill()        # Daily to hourly, forward fill
df.resample('H').interpolate()  # Interpolate

# Offset
df.resample('W-MON').sum()      # Weekly resample starting Monday

# Resample with groupby
df.groupby('category').resample('M')['value'].sum()
```

## Time Shifting
```python
# Shift values
df['prev_day'] = df['value'].shift(1)     # Shift down
df['next_day'] = df['value'].shift(-1)    # Shift up

# Shift dates
df.shift(periods=1, freq='D')             # Shift dates by 1 day

# Percentage change
df['pct_change'] = df['value'].pct_change()  # 1 period
df['pct_change_7d'] = df['value'].pct_change(periods=7)

# Difference
df['diff'] = df['value'].diff()           # Difference from previous
df['diff_7d'] = df['value'].diff(periods=7)
```

## Window Functions for Time Series
```python
# Rolling window
df['rolling_7d'] = df['value'].rolling(window=7).mean()
df['rolling_30d'] = df['value'].rolling(window=30, min_periods=5).mean()

# Expanding window
df['expanding_mean'] = df['value'].expanding().mean()

# Exponential weighted
df['ewm_span'] = df['value'].ewm(span=10).mean()
df['ewm_halflife'] = df['value'].ewm(halflife=5).mean()
```

## Time Zone Handling
```python
# Localize (add timezone)
df['date'].dt.tz_localize('UTC')
df['date'].dt.tz_localize('US/Eastern')

# Convert timezone
df['date'].dt.tz_convert('US/Pacific')

# Remove timezone
df['date'].dt.tz_localize(None)
```

# 📈 VISUALIZATION
## Basic Plotting
```python
import matplotlib.pyplot as plt

# Line plot
df.plot()                     # All numeric columns
df['col'].plot()
df.plot(x='date', y='value')
df.plot(kind='line')

# Bar plot
df.plot(kind='bar')
df.plot(kind='barh')          # Horizontal bar
df['col'].value_counts().plot(kind='bar')

# Histogram
df['col'].plot(kind='hist', bins=30)
df.plot(kind='hist', alpha=0.5)  # Multiple columns

# Box plot
df.plot(kind='box')
df[['col1', 'col2', 'col3']].plot(kind='box')

# Scatter plot
df.plot(kind='scatter', x='col1', y='col2')

# Area plot
df.plot(kind='area', alpha=0.5)

# Pie chart
df['col'].value_counts().plot(kind='pie')

# Customization
ax = df.plot(figsize=(10, 6), title='Title', grid=True)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.legend(loc='best')
plt.tight_layout()
plt.show()
```

## Plotting with Seaborn
```python
import seaborn as sns

# Distribution plots
sns.histplot(df['col'], kde=True)
sns.kdeplot(df['col'])
sns.boxplot(x='category', y='value', data=df)
sns.violinplot(x='category', y='value', data=df)

# Relationship plots
sns.scatterplot(x='col1', y='col2', data=df)
sns.scatterplot(x='col1', y='col2', hue='category', data=df)
sns.regplot(x='col1', y='col2', data=df)  # With regression line

# Categorical plots
sns.barplot(x='category', y='value', data=df)
sns.countplot(x='category', data=df)
sns.pointplot(x='category', y='value', data=df)

# Matrix plots
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
sns.clustermap(df.corr(), standard_scale=1)

# Pair plots
sns.pairplot(df, hue='category')
sns.pairplot(df, vars=['col1', 'col2', 'col3'])

# Facet grids
g = sns.FacetGrid(df, col='category', col_wrap=3)
g.map(plt.hist, 'value')
g.map(sns.scatterplot, 'col1', 'col2')

# Style
sns.set_style('whitegrid')
sns.set_palette('husl')
```

# ⚡ PERFORMANCE TIPS
## Optimization Techniques
```python
# Use vectorized operations (fastest)
df['new'] = df['col1'] + df['col2']           # ✓ Fast
df['new'] = df.apply(lambda row: row['col1'] + row['col2'], axis=1)  # ✗ Slow

# Use .loc/.iloc for assignment
df.loc[condition, 'new_col'] = value          # ✓ Fast
df[condition]['new_col'] = value              # ✗ Creates copy

# Use categorical data for strings with few unique values
df['category_col'] = df['category_col'].astype('category')

# Use appropriate data types
df['int_col'] = df['int_col'].astype(np.int32)  # Instead of int64

# Use query() for complex filtering
df.query('col1 > 5 and col2 == "A"')           # Often faster

# Use pd.eval() for complex expressions
pd.eval('df1 + df2 * df3')

# Chunk processing for large files
chunk_size = 10000
for chunk in pd.read_csv('large.csv', chunksize=chunk_size):
    process(chunk)
```

## Memory Optimization
```python
# Check memory usage
df.memory_usage(deep=True)
df.info(memory_usage='deep')

# Reduce numeric precision
df['float_col'] = df['float_col'].astype(np.float32)  # Instead of float64
df['int_col'] = pd.to_numeric(df['int_col'], downcast='integer')

# Use sparse data structures
df = df.to_sparse()

# Delete unused columns
del df['unused_col']
```

## Parallel Processing
```python
# Using multiprocessing
import multiprocessing as mp

def process_chunk(df_chunk):
    return df_chunk.apply(complex_function)

with mp.Pool(processes=4) as pool:
    results = pool.map(process_chunk, np.array_split(df, 4))

# Using swifter (for apply operations)
# pip install swifter
import swifter
df['new'] = df['col'].swifter.apply(complex_function)

# Using dask for out-of-core computation
# pip install dask[dataframe]
import dask.dataframe as dd
ddf = dd.read_csv('large.csv')
result = ddf.groupby('col').mean().compute()
```

## Caching Results
```python
# Cache with joblib
from joblib import Memory
memory = Memory(location='./cachedir')

@memory.cache
def expensive_computation(df):
    return df.groupby('col').apply(complex_operation)

# Use hashing for memoization
import hashlib

def hash_dataframe(df):
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()

cache = {}
key = hash_dataframe(df)
if key not in cache:
    cache[key] = expensive_computation(df)
result = cache[key]
```


# 🎯 QUICK REFERENCE CARD
## Most Common Operations
```python
# Read CSV
df = pd.read_csv('file.csv')

# View data
df.head(), df.tail(), df.sample(5)

# Basic info
df.info(), df.describe(), df.shape

# Select data
df['col'], df[['col1', 'col2']], df.loc[rows], df.iloc[positions]

# Filter
df[df['col'] > 5], df.query('col1 > 5 and col2 == "A"')

# Group & aggregate
df.groupby('col')['value'].agg(['mean', 'sum', 'count'])

# Handle missing values
df.dropna(), df.fillna(value), df.isnull().sum()

# Merge data
pd.merge(df1, df2, on='key'), pd.concat([df1, df2])

# Save data
df.to_csv('output.csv', index=False)
```

## Common Patterns
```python
# Create DataFrame from list of dicts
data = [{'col1': 1, 'col2': 'A'}, {'col1': 2, 'col2': 'B'}]
df = pd.DataFrame(data)

# Add calculated column
df['new'] = df['col1'] * df['col2']

# Filter and modify
df.loc[df['col'] > 5, 'new_col'] = 'high'

# Group and transform
df['group_mean'] = df.groupby('category')['value'].transform('mean')

# Pivot table
pivot = df.pivot_table(index='date', columns='type', values='value', aggfunc='sum')

# Time series resampling
df.set_index('date').resample('D').sum()

# Apply function to column
df['col'] = df['col'].apply(lambda x: x*2)

# String operations
df['col'] = df['col'].str.upper().str.strip()

# Check for duplicates
duplicates = df.duplicated().sum()
```

## Troubleshooting Common Errors
```python
# SettingWithCopyWarning
df.loc[condition, 'col'] = value  # ✓ Correct
df[condition]['col'] = value      # ✗ Wrong

# KeyError
if 'col' in df.columns:           # Check before accessing
    value = df['col']

# MemoryError
# Use chunksize when reading
# Downcast numeric types
# Delete unused columns

# DtypeWarning when reading CSV
# Specify dtypes explicitly
df = pd.read_csv('file.csv', dtype={'col1': str, 'col2': int})

# Performance issues
# Use vectorized operations instead of apply()
# Use .loc/.iloc instead of chained indexing
# Use categorical dtype for string columns with few unique values
```

# Useful Methods to Remember
```
# Inspection: head(), tail(), info(), describe(), shape
# Selection: [], loc[], iloc[], query()
# Cleaning: dropna(), fillna(), drop_duplicates(), replace()
# Transformation: groupby(), pivot_table(), melt(), apply()
# Combining: merge(), concat(), join()
# Time series: resample(), shift(), rolling()
# Visualization: plot(), hist(), scatter(), boxplot()
```









