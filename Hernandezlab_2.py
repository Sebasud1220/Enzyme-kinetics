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
def create_plot(xdata, ydata, x_curve, y_curve, dataset_name, equation_name, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.plot(x_curve, y_curve, label='Fitted Curve', color='red', linewidth=2)
    ax.scatter(xdata, ydata, label='Data Points', color='blue', edgecolor='k', s=100)
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
    equation_options = ['Langmuir', 'Quadratic Binding']
else:
    equation_options = ['Michaelis-Menten', 'Irreversible One-Step (Product)', 'Reversible Association (A+B=C)', 'Pseudo-First-Order (Excess B over A)']

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
        
        # Improved initial guess for parameters based on equation type
        if equations[i] == 'Langmuir':
            p0 = [max(ydata), np.median(xdata)]  # Example initial guess for Bmax and Kd
        elif equations[i] == 'Quadratic Binding':
            p0 = [max(ydata), np.median(xdata), 1.0]  # Example initial guess for Bmax, Kd, and c
        elif equations[i] == 'Michaelis-Menten':
            p0 = [max(ydata), np.median(xdata)]  # Example initial guess for Vmax and Km
        elif equations[i] == 'Irreversible One-Step (Product)':
            p0 = [1.0, max(ydata)]  # Example initial guess for k and Amplitude
        elif equations[i] == 'Reversible Association (A+B=C)':
            p0 = [0.1, 0.01, max(xdata), max(ydata)]  # Example initial guess for kon, koff, A0, B0
        elif equations[i] == 'Pseudo-First-Order (Excess B over A)':
            p0 = [0.1, max(ydata)]  # Example initial guess for k_obs and A0
        else:
            raise ValueError("Unknown equation selected")

        # Fit curve using curve_fit with improved initial guesses and increased maxfev
        popt, pcov = curve_fit(equation_func, xdata, ydata, p0=p0, maxfev=2000)
        
        # Calculate R2 score
        y_pred = equation_func(np.array(xdata), *popt)
        r2 = r2_score(ydata, y_pred)
        
        # Calculate standard errors from covariance matrix
        se = np.sqrt(np.diag(pcov))

        st.success(f"{dataset_name} Fit successful:")
        for param_name, value, error in zip(param_names, popt, se):
            st.write(f"{param_name}: {value:.2f} +/- {error:.2f}")
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

        # Additional analysis for Pseudo-First-Order (Excess B over A)
        if equations[i] == 'Pseudo-First-Order (Excess B over A)':
            st.subheader('Pseudo-First-Order Kinetics Analysis')
            st.write("Provide initial concentrations and time-dependent C values to analyze kobs vs. [B].")
            initial_A = st.number_input('Initial concentration of A (A0)', min_value=0.0, step=0.1, value=1.0)
            initial_B = st.number_input('Initial concentration of B (B0)', min_value=0.0, step=0.1, value=10.0)
            
            # Example input for time-dependent C values
            st.write("Example: Enter time and corresponding C values separated by spaces (e.g., '0 0.5 1 1.5' for time and '0 0.2 0.5 0.8' for C)")
            time_input = st.text_area('Enter time values (separated by spaces)', '')
            c_values_input = st.text_area('Enter corresponding C values (separated by spaces)', '')

            if time_input.strip() and c_values_input.strip():
                try:
                    times = [float(t) for t in time_input.strip().split()]
                    c_values = [float(c) for c in c_values_input.strip().split()]
                    
                    # Fit kobs vs. [B] linear regression
                    B_values = initial_B - np.array(c_values)
                    kobs_values = np.log(initial_A / np.array(c_values)) / times
                    slope, intercept = np.polyfit(B_values, kobs_values, 1)
                    kon_estimated = slope
                    koff_estimated = -intercept

                    st.write(f"Estimated kon: {kon_estimated:.2f}")
                    st.write(f"Estimated koff: {koff_estimated:.2f}")

                    # Plot kobs vs. [B]
                    fig_kobs_B = plt.figure()
                    plt.plot(B_values, kobs_values, 'o', label='Data Points')
                    plt.plot(B_values, slope * B_values + intercept, '-', label=f'Linear Fit (kon={slope:.2f}, koff={intercept:.2f})')
                    plt.xlabel('[B]')
                    plt.ylabel('kobs')
                    plt.title('Pseudo-First-Order Kinetics Analysis')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig_kobs_B)

                except ValueError:
                    st.error('Error parsing time and/or C values. Please enter numeric values separated by spaces.')

    except RuntimeError as e:
        st.error(f"{dataset_name} Optimal parameters not found: {e}")
