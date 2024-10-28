import matplotlib.pyplot as plt

# This data was obtained by running the 'VGG_PCR_Layers' model multiple times,
# each extracting the features from a different layer in the VGG16 model
# Layer names
layers = [
    "Block5_pool",
    "Block5_conv3",
    "Block5_conv2",
    "Block5_conv1",
    "Block4_pool",
    "Block4_conv3"
]

# MSE values
mse_values = [
    0.40781351923942566,
    0.40771856904029846,
    0.405080646276474,
    0.4044733941555023,
    0.40756890177726746,
    0.409424751996994
]

# R-squared values
r_squared_values = [
    0.16951928815300277,
    0.1695832340058866,
    0.17389115015932285,
    0.17463207424856286,
    0.16856491866548187,
    0.16511781140954657
]

# Create a figure and a single subplot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot R-squared values
ax1.plot(layers, r_squared_values, marker='o', linestyle='-', color='b', label='R-squared')
ax1.set_xlabel('VGG16 Layers')
ax1.set_ylabel('R-squared', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.legend(loc='upper left')
ax1.grid(True)

# Add the R-squared values to the plot
for i, value in enumerate(r_squared_values):
    ax1.annotate(f'{value:.3f}', (layers[i], r_squared_values[i]), textcoords="offset points", xytext=(10,2), ha='center')

ax1.set_xticklabels(layers, rotation=45)  # Rotate x-axis labels by 45 degrees

# Create a second y-axis to plot MSE values
ax2 = ax1.twinx()
ax2.plot(layers, mse_values, marker='o', linestyle='-', color='r', label='Mean Squared Error')
ax2.set_ylabel('MSE', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.legend(loc='upper right')

# Add the MSE values to the plot
for i, value in enumerate(mse_values):
    ax2.annotate(f'{value:.3f}', (layers[i], mse_values[i]), textcoords="offset points", xytext=(0,5), ha='center')

# Display the plot
plt.title('R-squared and Mean Squared Error (MSE) Values for VGG16 Layers')
plt.tight_layout()
plt.show()
