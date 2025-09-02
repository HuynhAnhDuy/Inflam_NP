import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the CSV file
data_df = pd.read_csv('rf_pred_bestfold_5CV.csv')  # Adjust the path as needed

# Extract true and predicted labels
y_true = data_df['y_true'].values
y_pred = data_df['y_predicted'].values

# Create the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Calculate accuracy for each group
true_positive = cm[1, 1]  # True Positives for Activity/Toxicity
false_negative = cm[1, 0]  # False Negatives for Activity/Toxicity
false_positive = cm[0, 1]  # False Positives for Non-Activity/Toxicity
true_negative = cm[0, 0]  # True Negatives for Non-Activity/Toxicity


# Accuracy calculations
accuracy_toxic = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
accuracy_non_toxic = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0

# Create a confusion matrix display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-carcinogens', 'Carcinogens'])

# Plot the confusion matrix
plt.figure(figsize=(1, 1))  # Set figure size for better visibility
disp.plot(cmap='binary', values_format='d', colorbar=True)  # <-- đổi sang Greys
# Nếu muốn đen→trắng (đảo chiều), dùng: cmap='binary'

# Add accuracy annotations
plt.text(0, 0.2, f'Accuracy: {accuracy_non_toxic*100:.2f}%', ha='center', va='center', color='white', fontsize=11,fontweight='bold')
plt.text(1, 1.2, f'Accuracy: {accuracy_toxic*100:.2f}%', ha='center', va='center', color='white', fontsize=11,fontweight='bold')

# Add labels and title
plt.title('RF_E-state (5-fold CV)', fontsize=16, fontweight='bold', fontstyle='italic', family='sans-serif')
plt.xlabel('Predicted', fontsize=16, fontweight='bold', fontstyle='italic', family='sans-serif') 
plt.ylabel('True', fontsize=16, fontweight='bold', fontstyle='italic', family='sans-serif') 


# Adjust layout
plt.tight_layout()  # Automatically adjust subplot parameters for a better fit

# Save the plot
plt.savefig('rf_cm_5CV.svg', bbox_inches="tight")

