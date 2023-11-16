from typing import Dict, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd

def linreg_total_sales(sales: pd.DataFrame, ad_costs: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """
    Function to perform linear regression on total sales
    """
    # Prepare data
    X = ad_costs.drop(columns=['day'])
    y = sales.groupby('day')['sales'].sum()

    # Create and train linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Calculate coefficients and R2 score
    coefficients = dict(zip(X.columns, model.coef_))
    coefficients['intercept'] = model.intercept_
    r_squared = r2_score(y, model.predict(X))

    return r_squared, coefficients

def linreg_category_sales(sales: pd.DataFrame, ad_costs: pd.DataFrame) -> Dict[str, Tuple[float, Dict[str, float]]]:
    """
    Function to perform linear regression on each category sales
    """
    # Merge dataframes
    df = pd.merge(sales, ad_costs, on='day')
    result = {}
    for category in df['category'].unique():
        # Fit model for each category
        model = LinearRegression().fit(
            df[df['category'] == category][['TV', 'Website banners', 'SMM', 'Google Ads']],
            df[df['category'] == category]['sales']
        )
        # Get coefficients and intercept
        coefficients = dict(zip(['TV', 'Website banners', 'SMM', 'Google Ads'], model.coef_))
        coefficients['intercept'] = model.intercept_
        # Calculate R^2 score
        r_squared = r2_score(
            df[df['category'] == category]['sales'],
            model.predict(df[df['category'] == category][['TV', 'Website banners', 'SMM', 'Google Ads']])
        )
        result[category] = (r_squared, coefficients)
    return result