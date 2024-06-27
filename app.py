import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Define Langmuir, Hill, Quadratic Hill, and Michaelis-Menten equations
def langmuir_equation(x, Bmax, Kd):
    return (Bmax * np.array(x)) / (Kd + np.array(x))

def hill_equation(x, Bmax, Kd, n):
    return (Bmax * np.array(x)**n) / (Kd**n + np.array(x)**n)

def quadratic_hill_equation(x, Bmax, Kd, n, c):
    return (Bmax * (np.array(x)**n + c)) / (Kd**n + np.array(x)**n + c)

def michaelis_menten_equation(x, Vmax, Km):
    return (Vmax * np.array(x)) / (Km + np.array(x))

# Function to select equation based on user choice
def select_equation(equation_name):
    if equation_name == 'Langmuir':
        return langmuir_equation, ['Bmax', 'Kd']
    elif equation_name == 'Hill':
        return hill_equation, ['Bmax', 'Kd', 'n']
    elif equation_name == 'Quadratic Hill':
        return quadratic_hill_equation, ['Bmax', 'Kd', 'n', 'c']
    elif equation_name == 'Michaelis-Menten':
        return michaelis_menten_equation, ['Vmax', 'Km']
    else:
        raise ValueError(f"Equation '{equation_name}' not recognized.")

# Streamlit UI title and instructions
st.title('Equation Fitting')

# Number of datasets input by user
num_datasets = st.number_input('Number of datasets', min_value=1, value=1, step=1)

# Equation selection dropdown
equation_name = st.selectbox('Select Equation', ['Langmuir', 'Hill', 'Quadratic Hill', 'Michaelis-Menten'])

# Instructions based on selected equation
if equation_name == 'Langmuir':
    st.write("""
    ### Langmuir Equation:
    - Bmax: Maximum binding
    - Kd: Dissociation constant
    """)
elif equation_name == 'Hill':
    st.write("""
    ### Hill Equation:
    - Bmax: Maximum binding
    - Kd: Dissociation constant
    - n: Hill coefficient
    """)
elif equation_name == 'Quadratic Hill':
    st.write("""
    ### Quadratic Hill Equation:
    - Bmax: Maximum binding
    - Kd: Dissociation constant
    - n: Hill coefficient
    - c: Quadratic coefficient
    """)
elif equation_name == 'Michaelis-Menten':
    st.write("""
    ### Michaelis-Menten Equation:
    - Vmax: Maximum reaction rate
    - Km: Michaelis constant
    """)

datasets = []
for i in range(num_datasets):
    dataset_name = st.text_input(f'Enter name for Dataset {chr(65+i)}', f'Dataset {chr(65+i)}')

    st.subheader(f'Dataset {dataset_name} Inputs:')
    x_input = st.text_area(f'Enter space-separated x values for {dataset_name}', f'x1 x2 x3', key=f'x_input_{i}')
    y_input = st.text_area(f'Enter space-separated y values for {dataset_name}', f'y1 y2 y3', key=f'y_input_{i}')

    if x_input.strip() and y_input.strip():
        try:
            x_data = [float(x) for x in x_input.strip().split()]
            y_data = [float(y) for y in y_input.strip().split()]
            datasets.append((x_data, y_data, dataset_name))
        except ValueError:
            st.error(f'Error parsing data for {dataset_name}. Please enter numeric values separated by spaces.')

# Attempt to fit the selected equation for each dataset
for i, (xdata, ydata, dataset_name) in enumerate(datasets):
    st.header(f'{dataset_name} Fitting')
    try:
        # Select equation and corresponding parameters
        equation_func, param_names = select_equation(equation_name)
        num_params = len(param_names)
        
        # Initial guess for parameters
        p0 = [1.0] * num_params

        # Fit curve using curve_fit with increased maxfev
        popt, _ = curve_fit(equation_func, xdata, ydata, p0=p0, maxfev=2000)
        
        # Calculate R2 score
        y_pred = equation_func(xdata, *popt)
        r2 = r2_score(ydata, y_pred)

        st.success(f"{dataset_name} Fit successful:")
        for param_name, value in zip(param_names, popt):
            st.write(f"{param_name}: {value:.2f}")
        st.write(f"R2 Score: {r2:.2f}")

        # Plotting
        fig, ax = plt.subplots()
        x_curve = np.linspace(min(xdata), max(xdata), 100)
        y_curve = equation_func(x_curve, *popt)
        ax.plot(x_curve, y_curve, label='Fitted Curve', color='red')
        ax.scatter(xdata, ydata, label='Data Points', color='blue')
        ax.set_xlabel('X values')
        ax.set_ylabel('Y values')
        ax.set_title(f'{dataset_name} {equation_name} Fit')
        ax.legend()
        st.pyplot(fig)

    except RuntimeError as e:
        st.error(f"{dataset_name} Optimal parameters not found: {e}")

# Display instructions or additional UI elements as needed
