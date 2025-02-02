import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from io import BytesIO

# Define equation functions
def langmuir_equation(x, Bmax, Kd):
    return (Bmax * x) / (Kd + x)

def quadratic_binding_equation(x, Bmax, Kd, c):
    return (Bmax * x**2) / (Kd**2 + c*x**2)

def hill_equation(x, Bmax, Kd, n):
    return Bmax * (x**n) / (Kd**n + x**n)

def michaelis_menten_equation(x, Vmax, Km):
    return (Vmax * x) / (Km + x)

def irreversible_one_step(x, k, Amplitude):
    return Amplitude * (1 - np.exp(-k * x))

def reversible_association(x, kon, koff, A0, B0):
    kobs = kon * A0 * B0 + koff
    return A0 - A0 * np.exp(-kobs * x)

def pseudo_first_order_excess_B(x, k_obs, A0):
    return A0 * np.exp(-k_obs * x)

# Define the select_equation function
def select_equation(equation_name):
    if equation_name == 'Langmuir':
        return langmuir_equation, ['Bmax', 'Kd']
    elif equation_name == 'Quadratic Binding':
        return quadratic_binding_equation, ['Bmax', 'Kd', 'c']
    elif equation_name == 'Hill':
        return hill_equation, ['Bmax', 'Kd', 'n']
    elif equation_name == 'Michaelis-Menten':
        return michaelis_menten_equation, ['Vmax', 'Km']
    elif equation_name == 'Irreversible One-Step (Product)':
        return irreversible_one_step, ['k', 'Amplitude']
    elif equation_name == 'Reversible Association (A+B=C)':
        return reversible_association, ['kon', 'koff', 'A0', 'B0']
    elif equation_name == 'Pseudo-First-Order (Excess B over A)':
        return pseudo_first_order_excess_B, ['k_obs', 'A0']
    else:
        raise ValueError("Unknown equation selected")

# Function to create and save plot
def create_plot(xdata, ydata, x_curve, y_curve, dataset_name, equation_name, xlabel, ylabel, curve_color, curve_thickness, point_color, point_size, point_marker, line_style):
    fig, ax = plt.subplots()
    ax.plot(x_curve, y_curve, color=curve_color, linewidth=curve_thickness, linestyle=line_style)
    ax.scatter(xdata, ydata, color=point_color, edgecolor='k', s=point_size, marker=point_marker)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(f'{dataset_name} {equation_name} Fit')
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
    equation_options = ['Langmuir', 'Quadratic Binding', 'Hill']
else:
    equation_options = ['Michaelis-Menten', 'Irreversible One-Step (Product)', 'Reversible Association (A+B=C)', 'Pseudo-First-Order (Excess B over A)']

# Number of datasets input by user
num_datasets = st.number_input('Number of datasets', min_value=1, value=1, step=1)

datasets = []
equations = []
for i in range(num_datasets):
    with st.expander(f"Dataset {chr(65+i)} Inputs"):
        dataset_name = st.text_input(f'Enter name for Dataset {chr(65+i)}', f'Dataset {chr(65+i)}')

        col1, col2 = st.columns(2)
        with col1:
            x_input = st.text_area(f'Enter space-separated x values for {dataset_name}', f'x1 x2 x3', key=f'x_input_{i}')
        with col2:
            y_input = st.text_area(f'Enter space-separated y values for {dataset_name}', f'y1 y2 y3', key=f'y_input_{i}')

        selected_equations = st.multiselect(f'Select Equations for {dataset_name}', equation_options, key=f'equation_{i}')
        xlabel = st.text_input(f'X-axis label for {dataset_name}', 'X values')
        ylabel = st.text_input(f'Y-axis label for {dataset_name}', 'Y values')

        col3, col4 = st.columns(2)
        with col3:
            curve_color = st.color_picker(f'Select curve color for {dataset_name}', '#FF0000')
        with col4:
            point_color = st.color_picker(f'Select point color for {dataset_name}', '#0000FF')

        col5, col6 = st.columns(2)
        with col5:
            curve_thickness = st.slider(f'Select curve thickness for {dataset_name}', 1, 10, 2)
            line_style = st.selectbox(f'Select line style for {dataset_name}', ['-', '--', '-.', ':'])
        with col6:
            point_size = st.slider(f'Select point size for {dataset_name}', 20, 200, 100)
            point_marker = st.selectbox(f'Select point marker for {dataset_name}', ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*'])

        if x_input.strip() and y_input.strip():
            try:
                x_data = [float(x) for x in x_input.strip().split()]
                y_data = [float(y) for y in y_input.strip().split()]
                datasets.append((x_data, y_data, dataset_name, xlabel, ylabel, curve_color, curve_thickness, point_color, point_size, point_marker, line_style))
                equations.append(selected_equations)
            except ValueError:
                st.error(f'Error parsing data for {dataset_name}. Please enter numeric values separated by spaces.')

# Option to set initial parameters or use default ones
use_default_params = st.checkbox("Use default initial parameters", value=True)

# Option to fit different datasets in one plot
fit_single_plot = st.checkbox("Fit all datasets in one plot", value=False)

# Attempt to fit the selected equation for each dataset
if fit_single_plot:
    st.header('Combined Fitting')
    fig, ax = plt.subplots()
    for i, (xdata, ydata, dataset_name, xlabel, ylabel, curve_color, curve_thickness, point_color, point_size, point_marker, line_style) in enumerate(datasets):
        for equation_name in equations[i]:
            equation_func, param_names = select_equation(equation_name)
            num_params = len(param_names)

            # Set initial guesses based on the equation type
            if use_default_params:
                if equation_name == 'Langmuir':
                    p0 = [max(ydata), np.median(xdata)]
                elif equation_name == 'Quadratic Binding':
                    p0 = [max(ydata), np.median(xdata), 1.0]
                elif equation_name == 'Hill':
                    p0 = [max(ydata), np.median(xdata), 1.0]
                elif equation_name == 'Michaelis-Menten':
                    p0 = [max(ydata), np.median(xdata)]
                elif equation_name == 'Irreversible One-Step (Product)':
                    p0 = [1.0, max(ydata)]
                elif equation_name == 'Reversible Association (A+B=C)':
                    p0 = [0.1, 0.01, max(xdata), max(ydata)]
                elif equation_name == 'Pseudo-First-Order (Excess B over A)':
                    p0 = [0.1, max(ydata)]
            else:
                st.write(f"Set initial parameters for {equation_name}:")
                p0 = [st.number_input(f'Initial value for {param}', value=1.0) for param in param_names]

            # Fit curve using curve_fit
            popt, pcov = curve_fit(equation_func, xdata, ydata, p0=p0, maxfev=10000)
            y_pred = equation_func(np.array(xdata), *popt)
            r2 = r2_score(ydata, y_pred)
            se = np.sqrt(np.diag(pcov))

            st.success(f"{equation_name} Fit successful for {dataset_name}:")
            with st.expander(f"Fit details for {dataset_name} - {equation_name}"):
                for param_name, value, error in zip(param_names, popt, se):
                    st.write(f"{param_name}: {value:.2f} +/- {error:.2f}")
                st.write(f"R2 Score: {r2:.2f}")

            x_curve = np.linspace(min(xdata), max(xdata), 100)
            y_curve = equation_func(x_curve, *popt)

            ax.plot(x_curve, y_curve, label=f'{dataset_name} - {equation_name}', color=curve_color, linewidth=curve_thickness, linestyle=line_style)
            ax.scatter(xdata, ydata, color=point_color, edgecolor='k', s=point_size, marker=point_marker)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(f'Combined Dataset Fit')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

else:
    for i, (xdata, ydata, dataset_name, xlabel, ylabel, curve_color, curve_thickness, point_color, point_size, point_marker, line_style) in enumerate(datasets):
        for equation_name in equations[i]:
            equation_func, param_names = select_equation(equation_name)
            num_params = len(param_names)

            # Set initial guesses based on the equation type
            if use_default_params:
                if equation_name == 'Langmuir':
                    p0 = [max(ydata), np.median(xdata)]
                elif equation_name == 'Quadratic Binding':
                    p0 = [max(ydata), np.median(xdata), 1.0]
                elif equation_name == 'Hill':
                    p0 = [max(ydata), np.median(xdata), 1.0]
                elif equation_name == 'Michaelis-Menten':
                    p0 = [max(ydata), np.median(xdata)]
                elif equation_name == 'Irreversible One-Step (Product)':
                    p0 = [1.0, max(ydata)]
                elif equation_name == 'Reversible Association (A+B=C)':
                    p0 = [0.1, 0.01, max(xdata), max(ydata)]
                elif equation_name == 'Pseudo-First-Order (Excess B over A)':
                    p0 = [0.1, max(ydata)]
            else:
                st.write(f"Set initial parameters for {equation_name}:")
                p0 = [st.number_input(f'Initial value for {param}', value=1.0) for param in param_names]

            # Fit curve using curve_fit
            popt, pcov = curve_fit(equation_func, xdata, ydata, p0=p0, maxfev=10000)
            y_pred = equation_func(np.array(xdata), *popt)
            r2 = r2_score(ydata, y_pred)
            se = np.sqrt(np.diag(pcov))

            st.success(f"{equation_name} Fit successful for {dataset_name}:")
            with st.expander(f"Fit details for {dataset_name} - {equation_name}"):
                for param_name, value, error in zip(param_names, popt, se):
                    st.write(f"{param_name}: {value:.2f} +/- {error:.2f}")
                st.write(f"R2 Score: {r2:.2f}")

            x_curve = np.linspace(min(xdata), max(xdata), 100)
            y_curve = equation_func(x_curve, *popt)

            fig = create_plot(xdata, ydata, x_curve, y_curve, dataset_name, equation_name, xlabel, ylabel, curve_color, curve_thickness, point_color, point_size, point_marker, line_style)
            st.pyplot(fig)
