# Best Selling Books Data Analysis
# This script processes and analyzes the Goodreads Best Selling Books dataset
# as per the assignment requirements

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import re
from datetime import datetime

# Set the aesthetic style of the plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# 1. Load the Dataset
print("1. Loading the Dataset")
# Assuming the dataset is downloaded from Kaggle and saved as 'books.csv'
# Load the dataset with better error handling
try:
    df = pd.read_csv(
        'data/books.csv',
        encoding='utf-8',   # Ensure proper encoding
        on_bad_lines='skip'  # Skip problematic rows
    )
    print("Dataset loaded successfully!")
except pd.errors.ParserError as e:
    print(f"Parsing error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

# Display dataset shape and first few rows
print("Dataset shape:", df.shape)
print("\nFirst 5 rows of the dataset:")
print(df.head())


# 2. Data Cleaning
print("\n2. Data Cleaning")

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Handle missing values
# For numeric columns, fill with median
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# For categorical columns, fill with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after handling:")
print(df.isnull().sum())

# Check for and remove duplicates
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate entries: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Duplicates removed. New shape: {df.shape}")

# Ensure correct data types
# Convert ratings to float if they're not already
rating_cols = ['average_rating', 'ratings_count', 'text_reviews_count']
for col in rating_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert 'num_pages' to int if it exists
if 'num_pages' in df.columns:
    df['num_pages'] = pd.to_numeric(df['num_pages'], errors='coerce')
    df['num_pages'] = df['num_pages'].fillna(0).astype(int)

print("\nDataset info after type conversion:")
print(df.info())

# 3. Exploratory Data Analysis (EDA)
print("\n3. Exploratory Data Analysis")

# Analyze the distribution of ratings
plt.figure(figsize=(10, 6))
sns.histplot(df['average_rating'], kde=True)
plt.title('Distribution of Average Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.savefig('rating_distribution.png')
plt.close()

# If 'genre' column exists, explore relationship between genres and ratings
# Note: The actual dataset might have different column names for genres
if 'genres' in df.columns:
    # Extract the first genre for each book for simplicity
    df['primary_genre'] = df['genres'].apply(lambda x: x.split(',')[0].strip() if isinstance(x, str) else 'Unknown')

    # Group by genre and calculate mean ratings
    genre_ratings = df.groupby('primary_genre')['average_rating'].mean().sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    genre_ratings.plot(kind='bar')
    plt.title('Average Ratings by Genre')
    plt.xlabel('Genre')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('genre_ratings.png')
    plt.close()

    print("\nTop 5 genres by average rating:")
    print(genre_ratings.head())

# Identify top 10 books with highest ratings
top_books = df.sort_values('average_rating', ascending=False).head(10)
print("\nTop 10 books with highest average ratings:")
print(top_books[['title', 'authors', 'average_rating']])

# Investigate correlation between number of pages and average rating
if 'num_pages' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='num_pages', y='average_rating', data=df, alpha=0.6)
    plt.title('Relationship Between Book Length and Average Rating')
    plt.xlabel('Number of Pages')
    plt.ylabel('Average Rating')
    plt.savefig('pages_vs_rating.png')
    plt.close()

    corr = df['num_pages'].corr(df['average_rating'])
    print(f"\nCorrelation between number of pages and average rating: {corr:.4f}")

# 4. Feature Engineering
print("\n4. Feature Engineering")

# Calculate median rating for each book
df['median_rating'] = df['average_rating']  # Placeholder, as we don't have actual median ratings
# Create rating_diff feature
df['rating_diff'] = df['average_rating'] - df['median_rating']

# Create is_bestseller feature
# Define as books with average rating in the top 10%
rating_threshold = df['average_rating'].quantile(0.9)
df['is_bestseller'] = (df['average_rating'] >= rating_threshold).astype(int)

print(f"\nNumber of bestsellers (rating >= {rating_threshold:.2f}): {df['is_bestseller'].sum()}")

# 5. Advanced Data Processing
print("\n5. Advanced Data Processing")

# Extract publication year from publication date
# The format of the publication date column might vary, adapt as needed
if 'publication_date' in df.columns:
    # Extract year using regex to handle different date formats
    df['publication_year'] = df['publication_date'].str.extract(r'(\d{4})', expand=False)
    df['publication_year'] = pd.to_numeric(df['publication_year'], errors='coerce')

    print("\nPublication years range:")
    print(f"Min year: {df['publication_year'].min()}, Max year: {df['publication_year'].max()}")

# One-hot encoding for genres
# This will depend on how genres are stored in the dataset
if 'primary_genre' in df.columns:
    # Get dummies for primary genre
    genre_dummies = pd.get_dummies(df['primary_genre'], prefix='genre')
    df = pd.concat([df, genre_dummies], axis=1)

    print("\nFirst 5 rows after one-hot encoding genres:")
    genre_cols = [col for col in df.columns if col.startswith('genre_')]
    print(df[genre_cols].head())

# 6. Data Aggregation
print("\n6. Data Aggregation")

# Group by author and calculate average rating
author_ratings = df.groupby('authors')['average_rating'].mean().sort_values(ascending=False)
top_authors_by_rating = author_ratings.head(5)

print("\nTop 5 authors with highest average ratings:")
print(top_authors_by_rating)

# Identify authors with most books
author_books_count = df.groupby('authors').size().sort_values(ascending=False)
top_authors_by_books = author_books_count.head(5)

print("\nTop 5 authors with most books:")
print(top_authors_by_books)

# 7. Data Visualization
print("\n7. Data Visualization")

# Bar plot showing number of books per genre
if 'primary_genre' in df.columns:
    plt.figure(figsize=(12, 8))
    genre_counts = df['primary_genre'].value_counts()
    genre_counts.plot(kind='bar')
    plt.title('Number of Books per Genre')
    plt.xlabel('Genre')
    plt.ylabel('Number of Books')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('books_per_genre.png')
    plt.close()

# Scatter plot of book length vs ratings
if 'num_pages' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='num_pages', y='average_rating', data=df, hue='is_bestseller', alpha=0.6)
    plt.title('Relationship Between Book Length and Average Rating')
    plt.xlabel('Number of Pages')
    plt.ylabel('Average Rating')
    plt.legend(title='Bestseller')
    plt.savefig('pages_vs_rating_bestseller.png')
    plt.close()

# Pie chart showing distribution of books by genre
if 'primary_genre' in df.columns:
    plt.figure(figsize=(12, 10))
    top_genres = genre_counts.head(10)  # Top 10 genres for readability
    plt.pie(top_genres, labels=top_genres.index, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Distribution of Books by Top 10 Genres')
    plt.tight_layout()
    plt.savefig('genre_distribution_pie.png')
    plt.close()

# 8. Bonus Challenge: Machine Learning Model
print("\n8. Bonus Challenge: Machine Learning Model")

# Prepare features and target
if 'num_pages' in df.columns and 'primary_genre' in df.columns:
    # Select features for prediction
    features = genre_cols.copy()  # Genre one-hot encoded columns
    if 'num_pages' in df.columns:
        features.append('num_pages')
    if 'publication_year' in df.columns:
        features.append('publication_year')

    # Remove rows with missing values in selected features
    model_df = df.dropna(subset=features + ['average_rating'])

    X = model_df[features]
    y = model_df['average_rating']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nRandom Forest Model Performance:")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R-squared: {r2:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nTop 10 most important features:")
    print(feature_importance.head(10))

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

# Summary of Findings
print("\n=== Summary of Findings ===")
print("1. Data cleaning involved handling missing values and ensuring correct data types.")
print(f"2. The average rating distribution is centered around {df['average_rating'].mean():.2f}.")
if 'primary_genre' in df.columns:
    print(f"3. The genre with highest average rating is {genre_ratings.index[0]}.")
if 'num_pages' in df.columns:
    print(
        f"4. Correlation between book length and rating is {corr:.4f}, suggesting a {'positive' if corr > 0 else 'negative'} relationship.")
print(f"5. {df['is_bestseller'].sum()} books were classified as bestsellers based on our threshold.")
if 'num_pages' in df.columns and 'primary_genre' in df.columns:
    print(
        f"6. Our prediction model achieved an R-squared of {r2:.4f}, indicating {'good' if r2 > 0.7 else 'moderate' if r2 > 0.5 else 'poor'} predictive power.")
    print(f"7. The most important feature for predicting ratings is {feature_importance.iloc[0]['Feature']}.")