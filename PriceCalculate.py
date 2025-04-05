from DataPreprocess import DataPreprocessor
from MMRPredict import MMRPredictor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class VehiclePriceCalculator:
    """
    Calculates whether a vehicle is underpriced or overpriced based on MMR prediction
    and value retention analysis.
    """
    
    def __init__(self, brand_focus=None):
        """
        Initialize the vehicle price calculator.
        
        Parameters:
        -----------
        brand_focus : str
            Specific brand to focus on for analysis
        """
        self.file_path = 'car_prices.csv' # built-in dataset for training
        self.brand_focus = brand_focus
        self.preprocessor = DataPreprocessor(self.file_path)
        self.mmr_predictor = MMRPredictor(model_type='gradient_boosting')
        self.avg_retention_by_brand = {}
        self.avg_retention_by_model = {}
        self.avg_retention_by_age = {}
    
        
    def segment(self):
        """
        Train the price calculator on historical vehicle data.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Historical vehicle data
            
        Returns:
        --------
        self
        """
        # Filter for focus brand if specified
        if self.brand_focus:
            df = df[df['brand'] == self.brand_focus].copy()
        
        # Load the untrained preprocessed data
        self.preprocessor.prepare_data()
        
        # Calculate average value retention by segment
        if 'brand' in self.preprocessor.columns and 'value_retention' in self.preprocessor.columns:
            self.avg_retention_by_brand = self.preprocessor.groupby('brand')['value_retention'].mean().to_dict()
        else:
            raise ValueError("Required columns 'brand' and 'value_retention' are missing in the data.")

        # Calculate average value retention by model
        if 'market_model' in self.preprocessor.columns and 'value_retention' in self.preprocessor.columns:
            self.avg_retention_by_model = self.preprocessor.groupby('market_model')['value_retention'].mean().to_dict()
        
        # Calculate average value retention by vehicle age
        if 'car_age' in self.preprocessor.columns and 'value_retention' in self.preprocessor.columns:
            self.avg_retention_by_age = self.preprocessor.groupby('car_age')['value_retention'].mean().to_dict()
        
    def train_mmr_predictor(self):
        """
        Train the MMR predictor model on the preprocessed data.
        
        Returns:
        --------
        self
        """
        self.mmr_predictor.preprocess_data()
        # Split the data into training and testing sets
        self.mmr_predictor.train_evaluate_optimize()
        
        return self.mmr_predictor.model
    
    def analyze_vehicle(self, vehicle_data, mmr=None, selling_price=None):
        """
        Analyze a single vehicle to determine if it's underpriced or overpriced.
        
        Parameters:
        -----------
        vehicle_data : pandas DataFrame or dict
            Vehicle attributes
        mmr : float
            Manheim Market Report value (optional - will be predicted if not provided)
        selling_price : float
            Current selling price
            
        Returns:
        --------
        dict
            Analysis results including predicted MMR, value retention, price status, etc.
        """
        # Convert dict to DataFrame if necessary
        if isinstance(vehicle_data, dict):
            vehicle_df = pd.DataFrame([vehicle_data])
        else:
            vehicle_df = vehicle_data.copy()
        
        # Add MMR if provided
        if mmr is not None:
            vehicle_df['mmr'] = mmr
            
        # Add selling price if provided
        if selling_price is not None:
            vehicle_df['selling_price'] = selling_price
        
        # Preoare the vehicle data
        vehicle_preprocessed = DataPreprocessor(vehicle_df)
        vehicle_preprocessed.prepare_data()
        
        # Predict MMR if not provided
        if mmr is None or 'mmr' not in vehicle_df.columns:
            vehicle_mmr = self.mmr_predictor.model
            vehicle_df['mmr'] = vehicle_mmr.predict(vehicle_df)
        
        # Calculate value retention if selling price is provided
        if 'selling_price' in vehicle_df.columns and 'mmr' in vehicle_df.columns:
            value_retention = vehicle_df['selling_price'].iloc[0] / vehicle_df['mmr'].iloc[0]
        else:
            value_retention = None
        
        # Get model and segment information
        model = vehicle_df['model'].iloc[0] if 'model' in vehicle_df.columns else None
        brand = vehicle_df['brand'].iloc[0] if 'brand' in vehicle_df.columns else None
        car_age = vehicle_df['vehicle_age'].iloc[0] if 'car_age' in vehicle_df.columns else None
        
        # Get average retention metrics for comparison
        model_avg_retention = self.avg_retention_by_model.get(model, None)
        brand_avg_retention = self.avg_retention_by_segment.get(brand, None)
        age_avg_retention = self.avg_retention_by_age.get(car_age, None)
        
        # Determine price status
        if value_retention is not None:
            if value_retention < 0.9:
                price_status = 'underpriced'
                price_deviation = (0.9 - value_retention) * 100  # Percentage below fair price
            elif value_retention > 1.1:
                price_status = 'overpriced'
                price_deviation = (value_retention - 1.1) * 100  # Percentage above fair price
            else:
                price_status = 'fair_price'
                price_deviation = 0
        else:
            price_status = None
            price_deviation = None
        
        # Calculate recommended price range
        if 'mmr' in vehicle_df.columns:
            mmr_value = vehicle_df['mmr'].iloc[0]
            min_fair_price = 0.9 * mmr_value
            max_fair_price = 1.1 * mmr_value
        else:
            min_fair_price = None
            max_fair_price = None
        
        # Return analysis results
        return {
            'vehicle_details': vehicle_df.iloc[0].to_dict(),
            'mmr': vehicle_df['mmr'].iloc[0] if 'mmr' in vehicle_df.columns else None,
            'selling_price': vehicle_df['selling_price'].iloc[0] if 'selling_price' in vehicle_df.columns else None,
            'value_retention': value_retention,
            'price_status': price_status,
            'price_deviation': price_deviation,
            'model_avg_retention': model_avg_retention,
            'segment_avg_retention': brand_avg_retention,
            'age_avg_retention': age_avg_retention,
            'min_fair_price': min_fair_price,
            'max_fair_price': max_fair_price
        }
    
    def analyze_inventory(self, inventory_df):
        """
        Analyze an entire inventory to identify underpriced and overpriced vehicles.
        
        Parameters:
        -----------
        inventory_df : pandas DataFrame
            Vehicle inventory data
            
        Returns:
        --------
        pandas DataFrame
            Inventory with price analysis
        """
        # Create a copy to avoid modifying the original DataFrame
        analyzed_inventory = inventory_df.copy()
        
        # Process each vehicle in the inventory
        results = []
        for i, row in analyzed_inventory.iterrows():
            vehicle_data = row.to_dict()
            analysis = self.analyze_vehicle(vehicle_data)
            results.append(analysis)
        
        # Create a DataFrame from the analysis results
        result_df = pd.DataFrame(results)
        
        # Add columns for value retention and price status
        analyzed_inventory['predicted_mmr'] = result_df['mmr']
        analyzed_inventory['value_retention'] = result_df['value_retention']
        analyzed_inventory['price_status'] = result_df['price_status']
        analyzed_inventory['min_fair_price'] = result_df['min_fair_price']
        analyzed_inventory['max_fair_price'] = result_df['max_fair_price']
        
        return analyzed_inventory
    
    def plot_value_retention_heatmap(self, df, segment=None):
        """
        Generate a heatmap of value retention by model year and age.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Vehicle data with value retention information
        segment : str
            Filter by vehicle segment
            
        Returns:
        --------
        matplotlib figure
            Value retention heatmap
        """
        # Filter data if segment is specified
        plot_df = df.copy()
        if segment:
            plot_df = plot_df[plot_df['segment'] == segment]
        
        # Create pivot table of value retention by year and age
        pivot = plot_df.pivot_table(
            values='value_retention',
            index='year',
            columns='vehicle_age',
            aggfunc='mean'
        )
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.2f', cbar_kws={'label': 'Value Retention'})
        plt.title(f'Value Retention by Model Year and Age{" for " + segment if segment else ""}')
        plt.xlabel('Vehicle Age (Years)')
        plt.ylabel('Model Year')
        
        return plt.gcf()
    
    def plot_price_status_distribution(self, df):
        """
        Generate a bar chart showing the distribution of price statuses.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Vehicle data with price status information
            
        Returns:
        --------
        matplotlib figure
            Price status distribution chart
        """
        # Count vehicles by price status
        status_counts = df['price_status'].value_counts()
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        sns.barplot(x=status_counts.index, y=status_counts.values)
        plt.title('Distribution of Price Statuses in Inventory')
        plt.xlabel('Price Status')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        return plt.gcf()
