from PriceCalculate import VehiclePriceCalculator
import pandas as pd
# Example usage:
if __name__ == "__main__":
    # Load sample data
    df = pd.read_csv('vehicle_data.csv')
    
    # Create a price calculator for a specific brand
    calculator = VehiclePriceCalculator(brand_focus='Toyota')
    
    # Train the calculator on historical data
    calculator.train_mmr_predictor(df)
    
    # Analyze a single vehicle
    vehicle = {
        'brand': 'Toyota',
        'model': 'Camry',
        'year': 2018,
        'mileage': 45000,
        'condition': 'Good',
        'vehicle_type': 'Sedan',
        'selling_price': 18500
    }
    
    analysis = calculator.analyze_vehicle(vehicle)
    print("\nVehicle Analysis:")
    print(f"MMR: ${analysis['mmr']:.2f}")
    print(f"Selling Price: ${analysis['selling_price']:.2f}")
    print(f"Value Retention: {analysis['value_retention']:.2f}")
    print(f"Price Status: {analysis['price_status']}")
    print(f"Recommended Price Range: ${analysis['min_fair_price']:.2f} - ${analysis['max_fair_price']:.2f}")
    
    # Analyze entire inventory (for example)
    inventory = pd.read_csv('inventory.csv') # this is an example dataset
    analyzed_inventory = calculator.analyze_inventory(inventory)
    
    # Count vehicles by price status
    price_status_counts = analyzed_inventory['price_status'].value_counts()
    print("\nInventory Analysis:")
    print(price_status_counts)
    
    # Plot value retention heatmap
    calculator.plot_value_retention_heatmap(df, segment='Sedan')
    
    # Plot price status distribution
    calculator.plot_price_status_distribution(analyzed_inventory)
