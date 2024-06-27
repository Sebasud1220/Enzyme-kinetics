import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from io import BytesIO

# Define equation functions
def langmuir_equation(x, Bmax, Kd):
    return (Bmax * x) / (Kd + x)

def hill_equation(x, Bmax, Kd, n):
    return (Bmax * x**n) / (Kd**n + x**n)

def quadratic_hill_equation(x, Bmax, Kd, n, c):
    return (Bmax * x**n) / (Kd**n + x**n + c*x**2)

def michaelis_menten_equation(x, Vmax, Km):
    return (Vmax * x) / (Km + x)

def irreversible_one_step(x, k, A0):
    return A0 * (1 - np.exp(-k * x))

def irreversible_one_step_reactant(x, k, A0):
    return A0 * np.exp(-k * x)

def reversible_one_step(x, k1, k2, A0, B0):
    k_obs = k1 + k2
    return (k1 * A0 * B0 / k_obs) * (1 - np.exp(-k_obs * x))

def pseudo_first_order(x, k_obs, C0):
    return C0 * np.exp(-k_obs * x)

# Define the select_equation function
def select_equation(equation_name):
    if equation_name == 'Langmuir':
        return langmuir_equation, ['Bmax', 'Kd']
    elif equation_name == 'Hill':
        return hill_equation, ['Bmax', 'Kd', 'n']
    elif equation_name == 'Quadratic Hill':
        return quadratic_hill_equation, ['Bmax', 'Kd', 'n', 'c']
    elif equation_name == 'Michaelis-Menten':
        return michaelis_menten_equation, ['Vmax', 'Km']
    elif equation_name == 'Irreversible One-Step (Product)':
        return irreversible_one_step, ['k', 'A0']
    elif equation_name == 'Irreversible One-Step (Reactant)':
        return irreversible_one_step_reactant, ['k', 'A0']
    elif equation_name == 'Reversible One-Step':
        return reversible_one_step, ['k1', 'k2', 'A0', 'B0']
    elif equation_name == 'Pseudo-First-Order':
        return pseudo_first_order, ['k_obs', 'C0']
    else:
        raise ValueError("Unknown equation selected")

# Function to create and save plot
def create_plot(xdata, ydata, x_curve, y_curve, dataset_name, equation_name, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.plot(x_curve, y_curve, label='Fitted Curve', color='red', linewidth=2)
    ax.scatter(xdata, ydata, label='Data Points', color='blue', edgecolor='k', s=100)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{dataset_name} {equation_name} Fit')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

# Streamlit UI title and instructions
st.title('HernandezLab Equation Fitting')

st.write("""
### Instructions:
- Enter the number of datasets you want to fit.
- For each dataset, provide a name and enter the x and y values separated by spaces.
- Choose the fitting equation from the dropdown menu.
- Click "Fit" to see the fitted curve and parameters.
""")

# Initial choice: kinetics or binding
fit_type = st.radio("Choose the type of fitting:", ('Binding', 'Kinetics'))

# Equation selection options based on initial choice
if fit_type == 'Binding':
    equation_options = ['Langmuir', 'Hill', 'Quadratic Hill', 'Michaelis-Menten']
else:
    equation_options = ['Michaelis-Menten', 'Irreversible One-Step (Product)', 'Irreversible One-Step (Reactant)', 'Reversible One-Step', 'Pseudo-First-Order']

# Number of datasets input by user
num_datasets = st.number_input('Number of datasets', min_value=1, value=1, step=1)

datasets = []
equations = []
for i in range(num_datasets):
    dataset_name = st.text_input(f'Enter name for Dataset {chr(65+i)}', f'Dataset {chr(65+i)}')

    st.subheader(f'Dataset {dataset_name} Inputs:')
    x_input = st.text_area(f'Enter space-separated x values for {dataset_name}', f'x1 x2 x3', key=f'x_input_{i}')
    y_input = st.text_area(f'Enter space-separated y values for {dataset_name}', f'y1 y2 y3', key=f'y_input_{i}')
    equation_name = st.selectbox(f'Select Equation for {dataset_name}', equation_options, key=f'equation_{i}')
    xlabel = st.text_input(f'X-axis label for {dataset_name}', 'X values')
    ylabel = st.text_input(f'Y-axis label for {dataset_name}', 'Y values')

    if x_input.strip() and y_input.strip():
        try:
            x_data = [float(x) for x in x_input.strip().split()]
            y_data = [float(y) for y in y_input.strip().split()]
            datasets.append((x_data, y_data, dataset_name, xlabel, ylabel))
            equations.append(equation_name)
        except ValueError:
            st.error(f'Error parsing data for {dataset_name}. Please enter numeric values separated by spaces.')

# Attempt to fit the selected equation for each dataset
for i, (xdata, ydata, dataset_name, xlabel, ylabel) in enumerate(datasets):
    st.header(f'{dataset_name} Fitting')
    try:
        # Select equation and corresponding parameters
        equation_func, param_names = select_equation(equations[i])
        num_params = len(param_names)
        
        # Initial guess for parameters
        p0 = [1.0] * num_params

        # Fit curve using curve_fit with increased maxfev
        popt, _ = curve_fit(equation_func, xdata, ydata, p0=p0, maxfev=2000)
        
        # Calculate R2 score
        y_pred = equation_func(np.array(xdata), *popt)
        r2 = r2_score(ydata, y_pred)

        st.success(f"{dataset_name} Fit successful:")
        for param_name, value in zip(param_names, popt):
            st.write(f"{param_name}: {value:.2f}")
        st.write(f"R2 Score: {r2:.2f}")

        # Plotting
        x_curve = np.linspace(min(xdata), max(xdata), 100)
        y_curve = equation_func(x_curve, *popt)
        fig = create_plot(xdata, ydata, x_curve, y_curve, dataset_name, equations[i], xlabel, ylabel)
        st.pyplot(fig)

        # Download plot
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.download_button(label="Download Plot", data=buf.getvalue(), file_name=f"{dataset_name}_fit.png", mime="image/png")

    except RuntimeError as e:
        st.error(f"{dataset_name} Optimal parameters not found: {e}")
