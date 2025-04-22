# Student Performance Analysis Project

## Overview
This project analyzes the factors affecting student academic performance using data science techniques. It implements a complete data science pipeline from data acquisition through exploratory analysis to predictive modeling, with the goal of understanding what influences student test scores and creating a model that can predict student performance.

## Dataset
The analysis uses the Student Performance Dataset, containing information about students' performances in math, reading, and writing examinations along with various demographic and socio-economic factors such as:
- Gender
- Race/ethnicity
- Parental level of education
- Lunch type (standard or free/reduced)
- Test preparation course completion

While smaller than 5,000 records, this dataset provides rich information for educational performance analysis and predictive modeling.

## Technologies Used
- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning modeling
- **Jupyter Notebook**: Development environment

## Project Structure
The project follows a comprehensive data science workflow:

1. **Data Acquisition**
   - Loading data from online source
   - Initial inspection of dataset structure

2. **Data Cleaning and Preprocessing**
   - Checking for missing values
   - Standardizing column names
   - Feature engineering (total and average scores)
   - Preparing categorical variables for encoding

3. **Exploratory Data Analysis (EDA)**
   - Statistical summaries of variables
   - Distribution analysis of categorical features
   - Examination of relationships between variables

4. **Data Visualization**
   - Score distributions
   - Correlation analysis through heatmaps
   - Comparative analysis across demographic groups
   - Subject performance visualization
   - Relationship exploration through pairplots

5. **Machine Learning Modeling**
   - Feature preparation and encoding
   - Training Linear Regression and Random Forest models
   - Model evaluation using multiple metrics
   - Cross-validation for reliability assessment
   - Feature importance analysis

6. **Findings and Conclusions**
   - Key insights about factors affecting student performance
   - Model performance comparison
   - Recommendations for future analysis

## Key Findings
- Strong correlations exist between math, reading, and writing scores
- Test preparation course completion positively impacts student performance
- Parental education level shows significant influence on academic outcomes
- Random Forest model outperforms Linear Regression in predicting student scores
- Feature importance analysis reveals the most influential factors for performance prediction

## Usage
1. Clone this repository
2. Install required dependencies:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Run the Jupyter notebook:
   ```
   jupyter notebook student_performance_analysis.ipynb
   ```

## Future Improvements
- Incorporate additional relevant datasets for more comprehensive analysis
- Implement advanced feature engineering
- Explore other machine learning algorithms and ensemble methods
- Perform hyperparameter tuning for model optimization
- Add interactive visualizations

## Contact
For questions or feedback about this project, please contact [MORGAN NSIKAK ADESHINA/07062492768].

## License
This project is available under the [NEWSCHOOLINNOVATION] license.
