import pandas as pd
from skin_type_detection import predict_skin_type, display_image_with_prediction

# Load the dataset
file_path = 'C:/project sem 6/DL PROJECT/Real-TIme-Skin-Type-Detection-main/gayass/skincare_products.csv'
df = pd.read_csv(file_path)

# Data Cleaning and Preprocessing
# Ensure 'inr' column is a string
df['inr'] = df['inr'].astype(str)

# Remove the ₹ symbol and any surrounding whitespace, then convert to numeric
df['inr'] = pd.to_numeric(df['inr'].str.replace('₹', '', regex=False).str.strip(), errors='coerce')

def recommend_products(skin_type, budget):
    
    # Filter products by skin type suitability
    suitable_products = df[df['suitability'].str.contains(skin_type, case=False, na=False)]
    
    # Split by product type
    cleansers = suitable_products[suitable_products['product_type'].str.contains('Cleanser', case=False, na=False)]
    serums = suitable_products[suitable_products['product_type'].str.contains('Serum', case=False, na=False)]
    moisturizers = suitable_products[suitable_products['product_type'].str.contains('Moisturiser', case=False, na=False)]
    
    # Store best combination and minimum difference from budget
    best_combination = None
    min_budget_diff = float('inf')
    
    # Iterate over all product combinations within each category
    for cleanser in cleansers.itertuples():
        for serum in serums.itertuples():
            for moisturizer in moisturizers.itertuples():
                total_inr = cleanser.inr + serum.inr + moisturizer.inr
                
                # Check if total inr is within the budget
                if total_inr <= budget and (budget - total_inr) < min_budget_diff:
                    best_combination = (cleanser, serum, moisturizer)
                    min_budget_diff = budget - total_inr
    
    # Output the recommended products
    if best_combination:
        print("Recommended Products within Budget:")
        for product in best_combination:
            print(f"Product: {product.product_name}, Type: {product.product_type}, inr: {product.inr}")
    else:
        print("No combination found within budget.")

# Example usage: Combine both systems
image_path = "C:/project sem 6/DL PROJECT/Real-TIme-Skin-Type-Detection-main/gayass/skin-dataset/dry/dry_1c6eb0a63898fa231b61_jpg.rf.5fbdf5b7ed73ba991b5aa4d1dd005831.jpg"
budget = 3000

# Detect skin type
predicted_skin_type = predict_skin_type(image_path)
print(f"Predicted Skin Type: {predicted_skin_type}")

# Display image with prediction
display_image_with_prediction(image_path, predicted_skin_type)

# Recommend products based on detected skin type
recommend_products(predicted_skin_type, budget)
